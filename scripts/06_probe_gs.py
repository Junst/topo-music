"""Close the last hole: is there chord/key-level topology?

NSynth is isolated single notes, so it can test octave equivalence (does C3 sit
near C4) but it cannot test the Tonnetz or the circle of fifths -- those are
relations among *keys and chords*, and a single note has neither. GiantSteps is
30 s clips labelled with one of 24 keys (12 tonics x major/minor), which is
exactly the object the circle of fifths organises.

The npz is written in the scripts/03 layout with the label axes reinterpreted:

    "pitches"  -> tonic pitch class, 0..11
    "families" -> mode, 0 = major, 1 = minor

so scripts/04 runs unchanged and its octave block computes, for tonic pitch
class, the chromatic circular distance and the circle-of-fifths distance. Note
that 04's `rho_abs_pitch` is then |delta pitch class|, which is NOT absolute
pitch height -- it must not be reported under that name for this arm.

Caveat that travels with every number here: key is a property of the piece, not
of a 20 ms frame, so every frame of a clip inherits the clip's label. Per-code
key preference is therefore diluted, and GiantSteps is entirely EDM, so genre
and production style are confounded with key. A weak result here is weaker
evidence than a weak result on NSynth.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GS = Path("/lustre/dataset/musicdataset/marble/GS")
CQT_SR, MERT_SR = 16000, 24000
TONIC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
ENH = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def parse_key(lab: str) -> tuple[int, int] | None:
    parts = lab.strip().split()
    if len(parts) != 2:
        return None
    t, mode = parts
    t = ENH.get(t, t)
    if t not in TONIC or mode.lower() not in ("major", "minor"):
        return None
    return TONIC[t], 0 if mode.lower() == "major" else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="cqt | mert_L<n> | encodec_32k | dac_44k | encodec_24k")
    ap.add_argument("--split", default="GS.test.jsonl")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kmeans-frames", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import librosa
    rows = []
    for line in (GS / a.split).read_text().splitlines():
        r = json.loads(line)
        km = parse_key(r["label"])
        if km is None:
            continue
        p = GS / Path(r["audio_path"]).relative_to("data/GS")
        if p.exists():
            rows.append((p, *km))
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    print(f"[{a.arm}] {len(rows)} clips", flush=True)

    is_codec = a.arm.startswith(("encodec", "dac"))
    model = taps = codec = None
    if is_codec:
        from topo.codecs import load
        codec = load(a.arm, device=a.device)
        native_sr = codec.sr
    elif a.arm.startswith("mert_L"):
        from transformers import AutoModel
        layer = int(a.arm.split("_L")[1])
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M",
                                          trust_remote_code=True).to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        taps = {}

        def tap(idx):
            def fn(_m, _i, o):
                taps[idx] = o[0] if isinstance(o, tuple) else o
            return fn
        model.feature_projection.register_forward_hook(tap(0))
        for i, lyr in enumerate(model.encoder.layers):
            lyr.register_forward_hook(tap(i + 1))
        native_sr = MERT_SR
    else:
        native_sr = CQT_SR

    def audio(p: Path) -> np.ndarray:
        x, sr = sf.read(p, dtype="float32", frames=int(a.seconds * 44100),
                        start=int(2.0 * 44100))
        if x.ndim > 1:
            x = x.mean(1)
        return librosa.resample(x, orig_sr=sr, target_sr=native_sr) if sr != native_sr else x

    def feats(paths, want_mel=False):
        ws = [audio(p) for p in paths]
        n = min(len(w) for w in ws)
        ws = [w[:n] for w in ws]
        if a.arm == "cqt":
            F = [librosa.amplitude_to_db(np.abs(librosa.cqt(
                w, sr=native_sr, hop_length=512, fmin=librosa.note_to_hz("C1"),
                n_bins=84, bins_per_octave=12)), ref=np.max).T.astype(np.float32) for w in ws]
        elif is_codec:
            wav = torch.from_numpy(np.stack(ws)).unsqueeze(1)
            F = [codec.encode(wav)[i].numpy() for i in range(len(ws))]   # [L,T'] each
        else:
            taps.clear()
            with torch.no_grad():
                model(torch.from_numpy(np.stack(ws)).to(a.device))
            h = taps[layer].float().cpu().numpy()
            F = [h[i] for i in range(len(ws))]
        if not want_mel:
            return F
        mels = []
        for w, f in zip(ws, F):
            T = f.shape[-1] if is_codec else len(f)
            hop = max(1, len(w) // max(T, 1))
            M = librosa.power_to_db(librosa.feature.melspectrogram(
                y=w, sr=native_sr, n_fft=2048, hop_length=hop, n_mels=a.n_mels))
            mels.append(M.T)
        return F, mels

    t0 = time.time()
    km_model = None
    if not is_codec:
        rng = np.random.default_rng(a.seed)
        sub = rows[:: max(1, len(rows) // 256)][:256]
        pool = []
        for b0 in range(0, len(sub), a.batch):
            pool.extend(feats([r[0] for r in sub[b0:b0 + a.batch]]))
        X = np.concatenate(pool, 0)
        if len(X) > a.kmeans_frames:
            X = X[rng.choice(len(X), a.kmeans_frames, replace=False)]
        from sklearn.cluster import MiniBatchKMeans
        km_model = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed,
                                   batch_size=4096, n_init=3, max_iter=300).fit(X)
        print(f"[{a.arm}] kmeans on {X.shape} ({time.time()-t0:.0f}s)", flush=True)

    L = codec.codebooks.shape[0] if is_codec else 1
    K = codec.codebooks.shape[1] if is_codec else a.k
    pc = np.zeros((L, K, 12), np.int32)
    fc = np.zeros((L, K, 2), np.int32)
    sc = np.zeros((L, K, 1), np.int32)
    spec = np.zeros((L, K, a.n_mels), np.float32)

    for b0 in range(0, len(rows), a.batch):
        br = rows[b0:b0 + a.batch]
        FF, MM = feats([r[0] for r in br], want_mel=True)
        for (_, ton, mode), F, M in zip(br, FF, MM):
            if is_codec:
                T = min(F.shape[-1], len(M))
                for l in range(L):
                    cj = F[l, :T]
                    u, cnt = np.unique(cj, return_counts=True)
                    pc[l, u, ton] += cnt.astype(np.int32)
                    fc[l, u, mode] += cnt.astype(np.int32)
                    sc[l, u, 0] += cnt.astype(np.int32)
                    np.add.at(spec[l], cj, M[:T])
            else:
                T = min(len(F), len(M))
                lab = km_model.predict(F[:T])
                u, cnt = np.unique(lab, return_counts=True)
                pc[0, u, ton] += cnt.astype(np.int32)
                fc[0, u, mode] += cnt.astype(np.int32)
                sc[0, u, 0] += cnt.astype(np.int32)
                np.add.at(spec[0], lab, M[:T])
        if b0 % (a.batch * 40) == 0:
            print(f"{b0+len(br)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    E = codec.codebooks if is_codec else km_model.cluster_centers_.astype(np.float32)[None]
    out = Path(a.out or f"runs/gs_{a.arm}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pitch_counts=pc, fam_counts=fc, src_counts=sc, spec_sums=spec,
        codebook=E, pitches=np.arange(12), families=np.arange(2),
        sources=np.arange(1), codec=f"gs_{a.arm}", split=a.split, n_clips=len(rows))
    print("wrote", out, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
