"""If the mechanism is averaging, octave equivalence must appear on NSynth too.

The examples-per-centroid sweep showed that clip-level key geometry is a
sample-averaging effect: with one clip per key the geometry is absent
(rho_fifths +0.035 for MERT L12) and it reaches +0.468 only after ~293 clips
are averaged per centroid, while stretching the window from 13 ms to 20 s moves
it by a factor of 1.7. That reframes the note/clip contradiction as 1 sample vs
293 samples rather than 0.2 s vs 20 s.

It also makes a prediction that has to hold or the account is wrong. The
note-level transposition test compares *individual* notes, which is the n = 1
condition. If low between-item SNR is what hides octave equivalence, then
building pitch centroids by averaging n notes of the same pitch across
instruments should make it appear -- in the same representation, on the same
notes, with the same statistic. This runs that sweep.

Nothing here is fitted, so there is no subspace and no supervision: only the
number of notes averaged into each pitch centroid changes.
"""
import json
from pathlib import Path

import numpy as np

ARMS = ["chroma", "cqt", "mert_L4", "mert_L12", "mert_L16", "mert_L24"]
NPP, R, KMAX = [1, 2, 4, 8, 16, 32, 64, 0], 10, 25      # 0 = every note
KS = np.arange(1, KMAX + 1)


def fit(sv, quad):
    cols = [np.ones_like(KS, float), KS.astype(float)]
    if quad:
        cols.append(KS.astype(float) ** 2)
    cols.append(np.cos(2 * np.pi * KS / 12))
    X = np.column_stack(cols)
    ok = ~np.isnan(sv)
    return np.linalg.lstsq(X[ok], sv[ok], rcond=None)[0]


def stats(sv):
    at = lambda k: sv[k - 1]
    return dict(d12=float(at(12) - 0.5 * (at(11) + at(13))),
                d24=float(at(24) - 0.5 * (at(23) + at(25))),
                d_strict=float(at(12) - np.nanmax(sv[7:11])),
                b1_height=float(fit(sv, False)[1]),
                b2_chroma=float(fit(sv, True)[-1]))


out = {}
for arm in ARMS:
    z = np.load(f"runs/nsynth_emb_{arm}.npz", allow_pickle=True)
    Z, pitch = z["Z"].astype(np.float64), z["pitch"]
    by = {p: np.where(pitch == p)[0] for p in sorted(set(pitch.tolist()))}
    by = {p: i for p, i in by.items() if len(i) >= 2}
    ps = np.array(sorted(by))
    rng = np.random.default_rng(0)
    rec = []
    for n in NPP:
        vals = []
        for r in range(R):
            C = {}
            for p, i in by.items():
                sel = i if n == 0 or n >= len(i) else rng.choice(i, n, replace=False)
                c = Z[sel].mean(0)
                C[p] = c / (np.linalg.norm(c) + 1e-12)
            sv = np.array([np.nanmean([C[p] @ C[p + k] for p in ps
                                       if p + k in C]) for k in KS])
            vals.append(stats(sv))
            if n == 0:
                break
        agg = {k: (float(np.mean([v[k] for v in vals])),
                   float(np.std([v[k] for v in vals]))) for k in vals[0]}
        rec.append(dict(n_per_pitch=n, n_draws=len(vals),
                        **{k: dict(mean=m, std=s) for k, (m, s) in agg.items()}))
        print(f"{arm:<10} n/pitch={('all' if n == 0 else n):>4} "
              f"d12={agg['d12'][0]:+.4f}+-{agg['d12'][1]:.4f}  "
              f"d_strict={agg['d_strict'][0]:+.4f}+-{agg['d_strict'][1]:.4f}  "
              f"b2={agg['b2_chroma'][0]:+.4f}", flush=True)
    out[arm] = dict(n_notes=int(len(Z)), n_pitches=int(len(ps)), curve=rec)
Path("runs/nsynth_class_means.json").write_text(json.dumps(out, indent=2))
print("wrote runs/nsynth_class_means.json")
