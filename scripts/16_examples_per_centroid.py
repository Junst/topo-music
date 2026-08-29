"""Which averaging exposes the geometry: over time, or over examples?

The accumulation curve came out flat where it was expected to rise. MERT L12
already shows rho_fifths = +0.276 from a 27 ms window -- 59% of its 20 s value
-- so pooling more audio per clip is not what creates the clip-level geometry.

But the clip-level measurement averages twice, and only one of those averages
was being varied. Each key centroid is the mean of ~293 clips, while the
note-level transposition test averages nothing at all: it compares individual
notes pairwise. So the candidate explanation is not temporal scale but
estimator variance -- a low-energy tonal signal buried under high-variance
nuisance directions emerges from *any* average over enough independent
examples.

This varies the other axis with the window length held at the full clip: n
clips per key centroid, everything else identical. If rho rises steeply with n
while it was flat in window length, the mechanism is averaging over examples,
not aggregation over time, and the note/clip gap is a sample-size effect rather
than a scale effect.
"""
import json
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader

s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()
ARMS = ["chroma", "cqt", "mert_L4", "mert_L12", "mert_L16", "mert_L24"]
NPK = [1, 2, 4, 8, 16, 32, 64, 128, 0]          # 0 = every clip of that key
R = 10

out = {}
for arm in ARMS:
    gz = np.load(f"runs/clipkey_{arm}.npz")
    Z = gz["Z"].astype(np.float64)
    Z = Z[:, :Z.shape[1] // 2]
    Y = gz["Y"]
    present = np.array(sorted(set(Y.tolist())))
    idx = {k: np.where(Y == k)[0] for k in present}
    rng = np.random.default_rng(0)
    rec = []
    for n in NPK:
        vals = []
        for r in range(R):
            C = np.stack([Z[i if n == 0 or n >= len(i)
                            else rng.choice(i, n, replace=False)].mean(0)
                          for i in idx.values()])
            g = s13.key_geometry(C, present, np.random.default_rng(r),
                                 n_perm=500)
            vals.append([g["fifths"]["rho"], g["kk"]["rho"]])
            if n == 0:
                break                      # deterministic, one draw is enough
        A = np.array(vals)
        rec.append(dict(n_per_key=n, n_draws=len(A),
                        fifths_mean=float(A[:, 0].mean()),
                        fifths_std=float(A[:, 0].std()),
                        kk_mean=float(A[:, 1].mean())))
        print(f"{arm:<10} n/key={('all' if n == 0 else n):>4} "
              f"fifths={A[:,0].mean():+.3f}+-{A[:,0].std():.3f} "
              f"kk={A[:,1].mean():+.3f}", flush=True)
    out[arm] = dict(median_clips_per_key=int(np.median(
        [len(i) for i in idx.values()])), curve=rec)
Path("runs/examples_per_centroid.json").write_text(json.dumps(out, indent=2))
print("wrote runs/examples_per_centroid.json")
