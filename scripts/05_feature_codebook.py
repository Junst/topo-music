"""E3: put continuous representations on the same footing as a codec codebook.

The codec result cannot be compared to MERT directly -- one is quantised and the
other is not, and that difference alone could produce any topology difference we
saw. So every representation is quantised the same way: k-means with the same K
over frames from the same NSynth notes. EnCodec's codebook is itself close to a
learned k-means over encoder features, so this is the fair matching.

Three representations, which together separate "the objective changed the
topology" from "the input already had it":

    cqt   log-CQT frames. The floor. This is the L0 experiment that the ladder
          in DESIGN.md listed and that was never run: if chroma or octave
          structure shows up here, it is a property of the input, not of any
          model.
    mert  MERT-v1-330M hidden states, one layer at a time. Music SSL.
    (the codec arm already exists from scripts/03.)

Output matches scripts/03 exactly -- pitch/fam/src count tables, per-code mean
log-mel for the E2 spectral control, and the codebook -- so scripts/04 consumes
it unchanged.

One caveat that must travel with the cqt arm: its E2 control is degenerate. The
control descriptor (log-mel) and the representation under test (log-CQT) are
near-collinear, so "regress the spectrum out" removes the representation itself.
E2 is interpretable for the mert and codec arms only.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NSYNTH = Path("/lustre/dataset/musicdataset/marble/nsynth")
WIN = (0.2, 2.5)
NSYNTH_SR = 16000
MERT_SR = 24000


def cqt_frames(wav: np.ndarray) -> np.ndarray:
    """log-CQT, 12 bins/octave over 7 octaves from C1. [T, 84]"""
    import librosa
    C = np.abs(librosa.cqt(wav, sr=NSYNTH_SR, hop_length=512, fmin=librosa.note_to_hz("C1"),
                           n_bins=84, bins_per_octave=12))
    return librosa.amplitude_to_db(C, ref=np.max).T.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["cqt", "mert"], required=True)
    ap.add_argument("--layer", type=int, default=12, help="mert only")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--split", default="nsynth-test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kmeans-frames", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    name = a.feat if a.feat == "cqt" else f"mert_L{a.layer}"
    meta = json.loads((NSYNTH / a.split / "examples.json").read_text())
    keys = sorted(meta)
    if a.limit:
        keys = keys[:: max(1, len(keys) // a.limit)][: a.limit]

    pitches = sorted({meta[k]["pitch"] for k in keys})
    fams = sorted({meta[k]["instrument_family"] for k in keys})
    srcs = sorted({meta[k]["instrument_source"] for k in keys})
    p_ix = {p: i for i, p in enumerate(pitches)}
    f_ix = {f: i for i, f in enumerate(fams)}
    s_ix = {s: i for i, s in enumerate(srcs)}

    model = rs = None
    taps: dict[int, torch.Tensor] = {}
    if a.feat == "mert":
        import torchaudio
        from transformers import AutoModel
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M",
                                          trust_remote_code=True).to(a.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        rs = torchaudio.transforms.Resample(NSYNTH_SR, MERT_SR)

        # MERT ships custom remote code whose forward does not propagate
        # output_hidden_states under transformers 5.x -- it comes back None.
        # Forward hooks read the same tensors and do not depend on that
        # plumbing. Layer 0 is the feature projection, 1..24 the encoder layers.
        def tap(idx):
            def fn(_m, _i, o):
                taps[idx] = o[0] if isinstance(o, tuple) else o
            return fn
        model.feature_projection.register_forward_hook(tap(0))
        for i, lyr in enumerate(model.encoder.layers):
            lyr.register_forward_hook(tap(i + 1))
        if not 0 <= a.layer <= len(model.encoder.layers):
            raise SystemExit(f"--layer must be 0..{len(model.encoder.layers)}")

    lo, hi = int(WIN[0] * NSYNTH_SR), int(WIN[1] * NSYNTH_SR)

    def feats_for(batch_keys, want_mel: bool = False):
        wavs = []
        for k in batch_keys:
            x, sr = sf.read(NSYNTH / a.split / "audio" / f"{k}.wav", dtype="float32")
            assert sr == NSYNTH_SR
            wavs.append(x[lo:hi])
        if a.feat == "cqt":
            F = [cqt_frames(w) for w in wavs]
        else:
            w = torch.stack([rs(torch.from_numpy(x)) for x in wavs]).to(a.device)
            taps.clear()
            with torch.no_grad():
                model(w)
            h = taps[a.layer].float().cpu().numpy()                # [B, T, D]
            F = [h[i] for i in range(len(batch_keys))]
        if not want_mel:
            return F
        import librosa
        mels = []
        for w, f in zip(wavs, F):
            hop = max(1, len(w) // len(f))
            M = librosa.power_to_db(librosa.feature.melspectrogram(
                y=w, sr=NSYNTH_SR, n_fft=2048, hop_length=hop, n_mels=a.n_mels))
            mels.append(M.T)                                       # [T_m, n_mels]
        return F, mels

    # ---- pass 1: fit k-means on a frame subsample -------------------------
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    sub = keys[:: max(1, len(keys) // 512)][:512]
    pool = []
    for b0 in range(0, len(sub), a.batch):
        pool.extend(feats_for(sub[b0 : b0 + a.batch]))
    X = np.concatenate(pool, 0)
    if len(X) > a.kmeans_frames:
        X = X[rng.choice(len(X), a.kmeans_frames, replace=False)]
    print(f"[{name}] kmeans on {X.shape} ({time.time()-t0:.0f}s)", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed, batch_size=4096,
                         n_init=3, max_iter=300).fit(X)
    C = km.cluster_centers_.astype(np.float32)                    # [K, D]
    print(f"[{name}] kmeans done ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: assign every frame, accumulate label counts --------------
    K = a.k
    pc = np.zeros((1, K, len(pitches)), np.int32)
    fc = np.zeros((1, K, len(fams)), np.int32)
    sc = np.zeros((1, K, len(srcs)), np.int32)
    spec = np.zeros((1, K, a.n_mels), np.float32)
    for b0 in range(0, len(keys), a.batch):
        bk = keys[b0 : b0 + a.batch]
        FF, MM = feats_for(bk, want_mel=True)
        for k, F, M in zip(bk, FF, MM):
            T = min(len(F), len(M))
            lab = km.predict(F[:T])
            pi, fi, si = (p_ix[meta[k]["pitch"]], f_ix[meta[k]["instrument_family"]],
                          s_ix[meta[k]["instrument_source"]])
            u, cnt = np.unique(lab, return_counts=True)
            pc[0, u, pi] += cnt.astype(np.int32)
            fc[0, u, fi] += cnt.astype(np.int32)
            sc[0, u, si] += cnt.astype(np.int32)
            np.add.at(spec[0], lab, M[:T])
        if b0 % (a.batch * 40) == 0:
            print(f"{b0+len(bk)}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)

    out = Path(a.out or f"runs/probe_{name}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pitch_counts=pc, fam_counts=fc, src_counts=sc, spec_sums=spec,
        codebook=C[None],                       # [1, K, D], same layout as codecs
        pitches=np.array(pitches), families=np.array(fams), sources=np.array(srcs),
        codec=name, split=a.split, n_clips=len(keys),
    )
    print("wrote", out, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
