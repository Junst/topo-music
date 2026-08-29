"""Probe-weight geometry for the clip-key arms, recomputed from cached embeddings.

Script 09 gained probe-weight geometry after the seven real arms had already
run, and re-extracting embeddings costs GPU hours for nothing: the clip
embeddings are already saved in runs/clipkey_<arm>.npz. This refits the same
probe on the cached Z and measures the geometry of its 24 class weight vectors,
then merges the result back into the arm's JSON.

The spectral partial is not available here -- the per-clip log-mel summary is
not cached -- so rho_given_spec is omitted rather than faked. The centroid
geometry in each JSON, which is the primary object, already carries it.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib.machinery import SourceFileLoader
m = SourceFileLoader("s09", str(Path(__file__).with_name("09_clip_key_geometry.py"))).load_module()


def reference_distances(present):
    """The same four reference structures script 09 builds, on the same pairs."""
    iu = np.triu_indices(len(present), 1)
    P = m.kk_profiles()[present]
    kk = 1.0 - np.corrcoef(P)[iu]
    ton = present // 2
    dchr = m.circ12(np.abs(ton[:, None] - ton[None, :]))[iu]
    f = (ton * 7) % 12
    dfif = m.circ12(np.abs(f[:, None] - f[None, :]))[iu]
    dmode = np.abs((present % 2)[:, None] - (present % 2)[None, :])[iu].astype(float)
    return kk, dfif, dchr, dmode, iu

N_PERM = 2000


def main(arm):
    p = Path(f"runs/clipkey_{arm}.json")
    d = np.load(f"runs/clipkey_{arm}.npz")
    Z, Y, present = d["Z"].astype(np.float64), d["Y"], d["keys"]
    out = json.loads(p.read_text())
    kk, dfif, dchr, dmode, iu = reference_distances(present)
    rng = np.random.default_rng(0)
    D = Z.shape[1] // 2

    for pool_name, sl in [("mean", slice(0, D)), ("mean+std", slice(0, 2 * D))]:
        ZP = Z[:, sl]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0,
                                               multi_class="multinomial"))
        t0 = time.time()
        clf.fit(ZP, Y)
        W = clf[-1].coef_
        assert list(clf[-1].classes_) == list(present)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        dw = (1.0 - Wn @ Wn.T)[iu]
        res = {}
        for nm, dd in [("kk_key_distance", kk), ("fifths", dfif),
                       ("chromatic", dchr), ("mode_mismatch", dmode)]:
            obs = float(spearmanr(dd, dw).statistic)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                pi = rng.permutation(len(present))
                null[i] = spearmanr(dd, (1.0 - Wn[pi] @ Wn[pi].T)[iu]).statistic
            res[nm] = dict(rho=obs, null_mean=float(null.mean()),
                           null_std=float(null.std()),
                           z=float((obs - null.mean()) / (null.std() + 1e-12)),
                           p=float((np.abs(null) >= abs(obs)).mean()))
            print(f"[{arm}/{pool_name}] {nm:<18} rho={obs:+.3f} "
                  f"z={res[nm]['z']:+5.1f} p={res[nm]['p']:.3f}", flush=True)
        out["poolings"][pool_name]["probe_weight_geometry"] = res
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    p.write_text(json.dumps(out, indent=2))
    print("updated", p)


if __name__ == "__main__":
    main(sys.argv[1])
