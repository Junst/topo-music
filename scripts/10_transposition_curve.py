"""Transposition intervention: is +12 semitones special?

The correlation statistics so far are indirect. This is the direct test. For a
representation f and a semitone offset k, measure

    D(k) = mean over (instrument, velocity, p) of  d( f(x_p), f(x_{p+k}) )

Octave equivalence predicts a dip at k = 12 and k = 24 against their neighbours.
Pitch-height sensitivity alone predicts a monotone rise. Decomposed as

    D(k) = c + b*k - a*cos(2*pi*k/12)

    a > 0  12-semitone periodicity, i.e. chroma / octave structure
    b > 0  pitch-height sensitivity

per layer, this gives the figure the rho_abs / rho_chroma table was standing in
for. Reported on cosine *similarity* S(k), so beta2 > 0 means octave periodicity
and beta1 < 0 means similarity falls with pitch distance.

Two design points that decide whether the curve means anything:

**The anchor set is identical at every k.** Only pitches p for which every one of
p+1 ... p+kmax exists in the same (instrument, velocity) group are used. Without
this, which instruments contribute changes with k -- instruments with narrow
ranges drop out of large k -- and the resulting composition shift is
indistinguishable from a real effect. This is why nsynth-train is the split:
901 instruments survive the constraint, against 23 in nsynth-valid.

**Bootstrap resamples instruments, not pairs.** Thousands of pairs come from a
few hundred instruments, and pairs within an instrument are not independent;
resampling pairs would give intervals that are far too narrow.

S(0) is not available. NSynth holds exactly one recording per (instrument,
pitch, velocity) -- checked, the maximum is 1 -- so there is no way to build a
same-pitch different-recording ceiling, and self-similarity is trivially 1 and
useless. The curve therefore has no upper reference and only relative structure
across k is interpreted.

**No DSP pitch shifting.** NSynth records the same instrument at the same
velocity across a wide pitch range -- nsynth-valid has a median of 48 pitches
per (instrument, velocity) group -- so x_p and x_{p+k} are both real recordings.
A phase vocoder would have introduced exactly the spectral artifacts the
measurement is trying to read, and comparing within an instrument also holds
timbre fixed for free. Register-dependent timbral variation remains a potential confound: real
instruments change timbre across their range, and register transitions and body
resonances need not be monotonic. What can be said is weaker than "it cannot
mimic an octave effect": unlike octave equivalence, it is not expected a priori
to exhibit a consistent 12-semitone periodicity across instruments.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import SourceFileLoader
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NSYNTH = Path("/lustre/dataset/musicdataset/marble/nsynth")
WIN = (0.2, 2.5)
SR_IN, MERT_SR, CQT_SR, MUQ_SR = 16000, 24000, 16000, 24000
MATPAC_SR, PUPU_SR = 16000, 24000
m9 = SourceFileLoader('m9', str(Path(__file__).with_name('09_clip_key_geometry.py'))).load_module()
PQ_SR, HCQT_SR = m9.PQ_SR, m9.HCQT_SR
MATPAC_CKPT = "/scratch2/solbon1212/ckpt/matpac_plus_music.pt"



# The two hand-built front ends are pure DSP and cost more per note than any
# learned encoder here, so they are spread over processes. Defined at module
# level because a process pool has to pickle the callable.
def _hcqt_one(w, sr):
    import librosa, numpy as np
    fmin = librosa.note_to_hz("C1")
    return np.concatenate([
        librosa.amplitude_to_db(np.abs(librosa.cqt(
            w, sr=sr, hop_length=512, fmin=h * fmin,
            n_bins=m9.HCQT_NB, bins_per_octave=m9.HCQT_BPO)),
            ref=np.max).mean(1) for h in m9.HCQT_H]).astype(np.float32)


def _pq_one(w, sr):
    import numpy as np, torch
    torch.set_num_threads(1)
    f_k = m9.PQ_FLOW * (2.0 ** (np.arange(m9.PQ_K) * m9.PQ_CENTS / 1200.0))
    yt, ch = torch.from_numpy(np.ascontiguousarray(w)), []
    for n_fft in m9.PQ_NFFTS:
        mag = torch.stft(yt, n_fft=n_fft, hop_length=m9.PQ_HOP, win_length=n_fft,
                         window=torch.hann_window(n_fft), center=True,
                         return_complex=True, pad_mode="reflect").abs()
        idx = torch.from_numpy(
            np.floor(f_k / (sr / n_fft) + 0.5).astype(np.int64)
        ).clamp(0, mag.shape[0] - 1)
        c = torch.log1p(mag[idx, :])
        ch.append(((c - c.mean()) / (c.std() + 1e-6)).numpy().astype(np.float32))
    return np.concatenate(ch, 0).mean(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="chroma | cqt | mert_L<n> | encodec_32k | ...")
    ap.add_argument("--split", default="nsynth-train")
    ap.add_argument("--kmax", type=int, default=25)
    ap.add_argument("--min-anchors", type=int, default=4)
    ap.add_argument("--anchors-per-group", type=int, default=6)
    ap.add_argument("--max-instruments", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the hcqt / pq_stft front ends")
    ap.add_argument("--save-emb", action="store_true",
                    help="also cache the note embeddings and the anchor "
                         "structure, so the curve can be recomputed inside a "
                         "learned subspace without re-running the encoder")
    a = ap.parse_args()

    import librosa
    KMAX = a.kmax
    meta = json.loads((NSYNTH / a.split / "examples.json").read_text())
    groups: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for k, v in meta.items():
        groups[(v["instrument_str"], v["velocity"])][v["pitch"]] = k

    rng = np.random.default_rng(a.seed)
    # one (instrument, velocity) group per instrument, and only anchors that
    # support every k -- so the anchor set is the same at all k
    per_inst: dict[str, tuple] = {}
    for (inst, vel), pm in groups.items():
        anc = sorted(p for p in pm if all(p + k in pm for k in range(1, KMAX + 1)))
        if len(anc) >= a.min_anchors and inst not in per_inst:
            per_inst[inst] = ((inst, vel), anc)
    insts = sorted(per_inst)
    if len(insts) > a.max_instruments:
        insts = [insts[i] for i in rng.choice(len(insts), a.max_instruments, replace=False)]

    sel = {}
    for inst in insts:
        g, anc = per_inst[inst]
        if len(anc) > a.anchors_per_group:
            anc = [anc[i] for i in sorted(rng.choice(len(anc), a.anchors_per_group,
                                                     replace=False))]
        sel[inst] = (g, anc)
    need = sorted({groups[g][p + k] for g, anc in sel.values()
                   for p in anc for k in range(0, KMAX + 1)})
    print(f"[{a.arm}] {len(insts)} instruments, {len(need)} notes, kmax={KMAX}",
          flush=True)

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
    elif a.arm == "pq_stft":
        native_sr = PQ_SR
    elif a.arm == "hcqt":
        native_sr = HCQT_SR
    elif is_matpac:
        from matpac.model import get_matpac
        layer = int(a.arm.split("_L")[1])
        model = get_matpac(checkpoint_path=MATPAC_CKPT,
                           pull_time_dimension=False).to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        native_sr = MATPAC_SR
    elif is_pupu:
        from topo import pupujepa_feats as pj
        model, pj_cfg = pj.load(device=a.device)
        native_sr = PUPU_SR
    elif is_muq:
        from muq import MuQ
        layer = int(a.arm.split("_L")[1])
        model = m9.patch_muq_hidden_states(
            MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(a.device).eval())
        for p_ in model.parameters():
            p_.requires_grad_(False)
        native_sr = MUQ_SR
    else:
        native_sr = CQT_SR

    lo, hi = int(WIN[0] * SR_IN), int(WIN[1] * SR_IN)

    POOL = (ProcessPoolExecutor(a.workers)
            if a.arm in ("hcqt", "pq_stft") else None)

    def embed(names):
        ws = []
        for n in names:
            x, sr = sf.read(NSYNTH / a.split / "audio" / f"{n}.wav", dtype="float32")
            w = x[lo:hi]
            ws.append(librosa.resample(w, orig_sr=sr, target_sr=native_sr)
                      if sr != native_sr else w)
        m = min(len(w) for w in ws)
        ws = [w[:m] for w in ws]
        if a.arm == "chroma":
            return np.stack([librosa.feature.chroma_cqt(
                y=w, sr=native_sr, hop_length=512).mean(1) for w in ws])
        if a.arm == "cqt":
            return np.stack([librosa.amplitude_to_db(np.abs(librosa.cqt(
                w, sr=native_sr, hop_length=512, fmin=librosa.note_to_hz("C1"),
                n_bins=84, bins_per_octave=12)), ref=np.max).mean(1) for w in ws])
        if a.arm in ("hcqt", "pq_stft"):
            fn = _hcqt_one if a.arm == "hcqt" else _pq_one
            return np.stack(list(POOL.map(fn, ws, [native_sr] * len(ws))))
        if is_codec:
            codes = codec.encode(torch.from_numpy(np.stack(ws)).unsqueeze(1))
            lat = sum(CB[l][codes[:, l]] for l in range(CB.shape[0]))
            return lat.mean(1).numpy()
        if is_matpac:
            with torch.no_grad():
                _, lay = model(torch.from_numpy(np.stack(ws)).to(a.device))
            return lay[:, layer - 1].float().mean(1).cpu().numpy()
        if is_pupu:
            return pj.embed(model, pj_cfg, np.stack(ws), device=a.device)
        x = torch.from_numpy(np.stack(ws)).to(a.device)
        with torch.no_grad():
            if is_muq:
                h = model(x, output_hidden_states=True).hidden_states[layer]
            else:
                taps.clear()
                model(x)
                h = taps[layer]
        return h.float().mean(1).cpu().numpy()

    t0 = time.time()
    Z: dict[str, np.ndarray] = {}
    for b0 in range(0, len(need), a.batch):
        nb = need[b0:b0 + a.batch]
        for n, e in zip(nb, embed(nb)):
            Z[n] = e.astype(np.float64)
        if b0 % (a.batch * 40) == 0:
            print(f"{b0+len(nb)}/{len(need)}  {time.time()-t0:.0f}s", flush=True)

    if a.save_emb:
        # Everything script 13 needs to redo this curve under a projection:
        # the embeddings, their pitch/instrument labels, and the exact anchor
        # structure -- so a projected curve is comparable to the ambient one
        # note for note rather than merely similar in construction.
        names = sorted(Z)
        idx = {n: i for i, n in enumerate(names)}
        # flat triples, because instruments do not all carry the same number of
        # anchors (min_anchors <= n <= anchors_per_group) and a padded matrix
        # would quietly invent notes
        rec = [(ii, ai, k, idx[groups[sel[inst][0]][p + k]])
               for ii, inst in enumerate(insts)
               for ai, p in enumerate(sel[inst][1])
               for k in range(0, KMAX + 1)]
        anchors = np.array(rec, dtype=np.int64)       # [M, 4] inst/anchor/k/row
        ep = Path(f"runs/nsynth_emb_{a.arm}.npz")
        np.savez_compressed(
            ep, Z=np.stack([Z[n] for n in names]).astype(np.float32),
            pitch=np.array([meta[n]["pitch"] for n in names]),
            velocity=np.array([meta[n]["velocity"] for n in names]),
            inst=np.array([meta[n]["instrument_str"] for n in names]),
            family=np.array([meta[n]["instrument_family_str"] for n in names]),
            anchors=anchors, kmax=KMAX, insts=np.array(insts))
        print("cached", ep, flush=True)

    def cossim(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    # per-instrument mean similarity at each k, over the *same* anchors for all k
    per_inst_S = np.full((len(insts), KMAX + 1), np.nan)
    for ii, inst in enumerate(insts):
        g, anc = sel[inst]
        pm = groups[g]
        for k in range(1, KMAX + 1):
            per_inst_S[ii, k] = float(np.mean(
                [cossim(Z[pm[p]], Z[pm[p + k]]) for p in anc]))

    ks = np.arange(1, KMAX + 1)
    S = np.nanmean(per_inst_S[:, 1:], axis=0)

    def _fit(sv, quadratic: bool):
        cols = [np.ones_like(ks, float), ks.astype(float)]
        if quadratic:
            cols.append(ks.astype(float) ** 2)
        cols.append(np.cos(2 * np.pi * ks / 12))
        X = np.column_stack(cols)
        ok = ~np.isnan(sv)
        coef, *_ = np.linalg.lstsq(X[ok], sv[ok], rcond=None)
        return coef

    def at(sv, k):
        return sv[k - 1]                                 # ks starts at 1

    def stats(sv):
        """Primary statistics are model-free local recurrences; the cosine
        amplitude is a secondary summary, and the quadratic-trend version is a
        sensitivity check -- a linear background trend that is actually curved
        would otherwise leak into the cosine term."""
        d12 = at(sv, 12) - 0.5 * (at(sv, 11) + at(sv, 13))
        d24 = at(sv, 24) - 0.5 * (at(sv, 23) + at(sv, 25))
        d_str = at(sv, 12) - np.nanmax(sv[7:11])         # S(12) - max S(8..11)
        b0, b1, b2 = _fit(sv, False)
        q = _fit(sv, True)                               # b0, b1, b2quad, a
        return np.array([d12, d24, d_str, b1, b2, q[-1]])

    obs = stats(S)
    NAMES = ["delta12", "delta24", "delta_strict", "b1_pitch_height",
             "b2_chroma", "b2_chroma_quadratic_trend"]
    # bootstrap over instruments: pairs within an instrument are not independent
    boot = np.array([stats(np.nanmean(
        per_inst_S[rng.choice(len(insts), len(insts)), 1:], 0)) for _ in range(a.n_boot)])
    ci = lambda j: [float(np.percentile(boot[:, j], 2.5)),
                    float(np.percentile(boot[:, j], 97.5))]

    out = dict(arm=a.arm, split=a.split, kmax=KMAX, n_instruments=len(insts),
               n_notes=len(need), anchors_per_group=a.anchors_per_group,
               k=ks.tolist(), S=S.tolist(), S0_available=False,
               stats={n: dict(value=float(obs[j]), ci=ci(j))
                      for j, n in enumerate(NAMES)},
               S_at=dict(S11=float(at(S, 11)), S12=float(at(S, 12)),
                         S13=float(at(S, 13)), S23=float(at(S, 23)),
                         S24=float(at(S, 24)), S25=float(at(S, 25))))
    print(f"[{a.arm}]  PRIMARY (model-free)")
    for j, n in enumerate(NAMES[:3]):
        print(f"    {n:<14} {obs[j]:+.4f}  CI{[round(x, 4) for x in ci(j)]}")
    print(f"  SECONDARY (fitted)")
    for j, n in enumerate(NAMES[3:], start=3):
        print(f"    {n:<26} {obs[j]:+.5f}  CI{[round(x, 5) for x in ci(j)]}")
    print(f"  S(11)={at(S,11):.4f} S(12)={at(S,12):.4f} S(13)={at(S,13):.4f} | "
          f"S(23)={at(S,23):.4f} S(24)={at(S,24):.4f} S(25)={at(S,25):.4f}")

    p = Path(a.out or f"runs/transpose_{a.arm}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
