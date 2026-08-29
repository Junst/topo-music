"""Split-matched random-subspace control for the GS side of the key_lda arm.

Script 13's random control evaluates clip geometry on all 7035 clips, but the
key_lda subspace is evaluated on the 2406 held-out test clips, where centroids
are much noisier (ambient rho_fifths +0.127 vs +0.468). Comparing a test-split
projection against an all-splits random baseline would compare two different
things. This recomputes the random baseline on exactly the clips key_lda is
scored on.
"""
import json, sys
import numpy as np
from importlib.machinery import SourceFileLoader
from pathlib import Path

s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()
DIMS, R = [1, 2, 3, 4, 6, 8, 11], 20

lab, gsplit = s13.gs_key_labels()
out = {}
for arm in sys.argv[1:]:
    gz = np.load(f"runs/clipkey_{arm}.npz")
    Z = gz["Z"].astype(np.float64); Y = gz["Y"]
    Zg = Z[:, :Z.shape[1] // 2]
    assert (lab == Y).all()
    te = gsplit == 2
    rng = np.random.default_rng(0)
    rec = {}
    for d in DIMS:
        if d > Zg.shape[1]:
            continue
        v = []
        for _ in range(R):
            W = np.linalg.qr(rng.normal(size=(Zg.shape[1], d)))[0]
            g = s13.key_geometry(Zg[te] @ W, Y[te], np.random.default_rng(0),
                                 n_perm=300)
            v.append([g["fifths"]["rho"], g["kk"]["rho"]])
        A = np.array(v)
        rec[str(d)] = dict(fifths_mean=float(A[:, 0].mean()),
                           fifths_std=float(A[:, 0].std()),
                           fifths_hi=float(np.percentile(A[:, 0], 95)),
                           kk_mean=float(A[:, 1].mean()))
        print(f"{arm:<13} d={d:<3} random(test split) fifths="
              f"{A[:,0].mean():+.3f}+-{A[:,0].std():.3f} "
              f"p95={np.percentile(A[:,0],95):+.3f}", flush=True)
    out[arm] = rec
Path("runs/random_gs_test_control.json").write_text(json.dumps(out, indent=2))
print("wrote runs/random_gs_test_control.json")
