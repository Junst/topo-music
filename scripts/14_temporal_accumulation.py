"""How many frames of aggregation does tonal geometry need to appear?

This is the experiment the note/clip gap asks for directly. At 13 ms (one MERT
frame) the representation shows no octave equivalence; at 20 s it shows a
circle of fifths as clear as a chromagram's. Somewhere between those two the
geometry appears, and the shape of that curve distinguishes two explanations:

  - a threshold, i.e. tonal geometry needs enough frames to *estimate* a
    pitch-class distribution at all, in which case rho rises steeply once a bar
    or two is covered and then flattens;
  - slow averaging, i.e. the tonal signal is low-energy and only emerges as
    unrelated variance (attack transients, production, instrumentation) is
    cancelled, in which case rho climbs gradually with no knee.

Design. For each window length n, one window of n contiguous frames is drawn
per clip, pooled to a single embedding, and the 24 key centroids are built from
those. The number of clips per centroid is therefore identical at every n --
only the window length changes -- so nothing in the curve comes from centroid
sample size, which is the confound that made the split-matched ambient
necessary in script 13. R independent draws give a spread rather than a point.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from importlib.machinery import SourceFileLoader
m9 = SourceFileLoader("s09", str(Path(__file__).with_name(
    "09_clip_key_geometry.py"))).load_module()

GS, MERT_SR, CQT_SR, MUQ_SR = m9.GS, m9.MERT_SR, m9.CQT_SR, m9.MUQ_SR
PQ_SR, HCQT_SR, MATPAC_SR, PUPU_SR = m9.PQ_SR, m9.HCQT_SR, m9.MATPAC_SR, m9.PUPU_SR
# Window lengths in *seconds*, not frames: chroma/CQT run at 31.25 fps and MERT
# at 75, so a frame-indexed sweep would compare 32 ms against 13 ms and put the
# arms on different x axes. 0 = the whole clip.
SECS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 0.0]


def geometry(C, present, rng, n_perm):
    iu = np.triu_indices(len(present), 1)
    P = m9.kk_profiles()[present]
    kk = 1.0 - np.corrcoef(P)[iu]
    ton = present // 2
    f = (ton * 7) % 12
    dfif = m9.circ12(np.abs(f[:, None] - f[None, :]))[iu]
    dchr = m9.circ12(np.abs(ton[:, None] - ton[None, :]))[iu]
    dz = squareform(pdist(C))[iu]
    out = {}
    for nm, dd in [("kk", kk), ("fifths", dfif), ("chromatic", dchr)]:
        obs = float(spearmanr(dd, dz).statistic)
        null = np.array([spearmanr(dd, squareform(pdist(
            C[rng.permutation(len(present))]))[iu]).statistic
            for _ in range(n_perm)])
        out[nm] = dict(rho=obs, z=float((obs - null.mean()) / (null.std() + 1e-12)),
                       p=float((np.abs(null) >= abs(obs)).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import librosa
    rows = []
    for sp in a.splits:
        for line in (GS / sp).read_text().splitlines():
            r = json.loads(line)
            km = m9.parse_key(r["label"])
            p = GS / Path(r["audio_path"]).relative_to("data/GS")
            if km is not None and p.exists():
                rows.append((p, km[0] * 2 + km[1]))
    if a.max_clips:
        rows = rows[:a.max_clips]
    print(f"[{a.arm}] {len(rows)} clips", flush=True)

    # this script only handled the mert_L arms and the two spectral ones; the
    # codec and MuQ branches mirror script 09
    is_codec = a.arm.startswith(("encodec", "dac"))
    is_muq = a.arm.startswith("muq_L")
    is_matpac = a.arm.startswith("matpac_L")
    is_pupu = a.arm == "pupujepa"
    model = taps = codec = CB = None
    layer = None
    if is_codec:
        from topo.codecs import load
        codec = load(a.arm, device=a.device)
        native_sr = codec.sr
        CB = torch.from_numpy(codec.codebooks)
    elif is_matpac:
        from matpac.model import get_matpac
        layer = int(a.arm.split("_L")[1])
        model = get_matpac(checkpoint_path=m9.MATPAC_CKPT,
                           pull_time_dimension=False).to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        native_sr = MATPAC_SR
    elif is_pupu:
        from topo import pupujepa_feats as pj
        model, pj_cfg = pj.load(device=a.device)
        native_sr = PUPU_SR
    elif a.arm == "pq_stft":
        native_sr = PQ_SR
    elif a.arm == "hcqt":
        native_sr = HCQT_SR
    elif is_muq:
        from muq import MuQ
        layer = int(a.arm.split("_L")[1])
        model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        native_sr = MUQ_SR
    elif a.arm.startswith("mert_L"):
        from transformers import AutoModel
        layer = int(a.arm.split("_L")[1])
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M",
                                          trust_remote_code=True).to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        taps = {}

        def tap(i):
            def fn(_m, _in, o):
                taps[i] = o[0] if isinstance(o, tuple) else o
            return fn
        model.feature_projection.register_forward_hook(tap(0))
        for i, lyr in enumerate(model.encoder.layers):
            lyr.register_forward_hook(tap(i + 1))
        native_sr = MERT_SR
    else:
        native_sr = CQT_SR

    def frames(paths):
        ws = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32", frames=int(a.seconds * 44100),
                            start=int(2.0 * 44100))
            if x.ndim > 1:
                x = x.mean(1)
            ws.append(librosa.resample(x, orig_sr=sr, target_sr=native_sr)
                      if sr != native_sr else x)
        n = min(len(w) for w in ws)
        ws = [w[:n] for w in ws]
        if a.arm == "chroma":
            return [librosa.feature.chroma_cqt(y=w, sr=native_sr,
                                               hop_length=512).T for w in ws]
        if a.arm == "cqt":
            return [librosa.amplitude_to_db(np.abs(librosa.cqt(
                w, sr=native_sr, hop_length=512, fmin=librosa.note_to_hz("C1"),
                n_bins=84, bins_per_octave=12)), ref=np.max).T for w in ws]
        if a.arm == "hcqt":
            fmin = librosa.note_to_hz("C1")
            return [np.concatenate(
                [librosa.amplitude_to_db(np.abs(librosa.cqt(
                    w, sr=native_sr, hop_length=512, fmin=h * fmin,
                    n_bins=m9.HCQT_NB, bins_per_octave=m9.HCQT_BPO)),
                    ref=np.max) for h in m9.HCQT_H], 0).T for w in ws]
        if a.arm == "pq_stft":
            f_k = m9.PQ_FLOW * (2.0 ** (np.arange(m9.PQ_K) * m9.PQ_CENTS / 1200.0))
            out = []
            for w in ws:
                yt = torch.from_numpy(np.ascontiguousarray(w))
                ch = []
                for n_fft in m9.PQ_NFFTS:
                    spec = torch.stft(yt, n_fft=n_fft, hop_length=m9.PQ_HOP,
                                      win_length=n_fft,
                                      window=torch.hann_window(n_fft),
                                      center=True, return_complex=True,
                                      pad_mode="reflect")
                    mag = spec.abs()
                    idx = torch.from_numpy(
                        np.floor(f_k / (native_sr / n_fft) + 0.5).astype(np.int64)
                    ).clamp(0, mag.shape[0] - 1)
                    c = torch.log1p(mag[idx, :])
                    c = (c - c.mean()) / (c.std() + 1e-6)
                    ch.append(c.numpy().astype(np.float32))
                out.append(np.concatenate(ch, 0).T)
            return out
        if is_matpac:
            with torch.no_grad():
                _, lay = model(torch.from_numpy(np.stack(ws)).to(a.device))
            return list(lay[:, layer - 1].float().cpu().numpy())
        if is_pupu:
            return pj.tokens(model, pj_cfg, np.stack(ws), device=a.device)
        if is_codec:
            codes = codec.encode(torch.from_numpy(np.stack(ws)).unsqueeze(1))
            lat = sum(CB[l][codes[:, l]] for l in range(CB.shape[0]))
            return list(lat.numpy())
        x = torch.from_numpy(np.stack(ws)).to(a.device)
        with torch.no_grad():
            if is_muq:
                h = model(x, output_hidden_states=True).hidden_states[layer]
            else:
                taps.clear()
                model(x)
                h = taps[layer]
        return list(h.float().cpu().numpy())

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    pooled = None                          # [len(NS), repeats, n_clips, D]
    Y = np.array([r[1] for r in rows])
    for b0 in range(0, len(rows), a.batch):
        br = rows[b0:b0 + a.batch]
        F = frames([r[0] for r in br])
        if pooled is None:
            D = F[0].shape[1]
            fps = F[0].shape[0] / a.seconds
            NS = [0 if s_ == 0.0 else max(1, int(round(s_ * fps))) for s_ in SECS]
            pooled = np.zeros((len(NS), a.repeats, len(rows), D), np.float32)
        for j, f in enumerate(F):
            T = f.shape[0]
            for ni, n in enumerate(NS):
                for r in range(a.repeats):
                    if n == 0 or n >= T:
                        pooled[ni, r, b0 + j] = f.mean(0)
                    else:
                        s = int(rng.integers(0, T - n + 1))
                        pooled[ni, r, b0 + j] = f[s:s + n].mean(0)
        if b0 % (a.batch * 60) == 0:
            print(f"{b0+len(br)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    present = np.array(sorted(set(Y.tolist())))
    out = {"arm": a.arm, "n_clips": len(rows), "dim": int(pooled.shape[-1]),
           "repeats": a.repeats, "fps": float(fps), "seconds": SECS, "curve": []}
    for ni, (n, sec) in enumerate(zip(NS, SECS)):
        per = []
        for r in range(a.repeats):
            C = np.stack([pooled[ni, r][Y == k].mean(0) for k in present])
            per.append(geometry(C.astype(np.float64), present,
                                np.random.default_rng(a.seed + r), a.n_perm))
        rec = {"n_frames": n, "seconds": sec}
        for nm in ("kk", "fifths", "chromatic"):
            v = np.array([p[nm]["rho"] for p in per])
            rec[nm] = dict(mean=float(v.mean()), std=float(v.std()),
                           z_mean=float(np.mean([p[nm]["z"] for p in per])))
        out["curve"].append(rec)
        print(f"  {sec:>5}s n={n:<5} fifths={rec['fifths']['mean']:+.3f}"
              f"+-{rec['fifths']['std']:.3f}  kk={rec['kk']['mean']:+.3f}"
              f"  chrom={rec['chromatic']['mean']:+.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    p = Path(f"runs/accum_{a.arm}.json")
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
