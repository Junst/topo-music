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


def synth(kind, Y, present, rng, D=256):
    """Two controls for the probe-weight result.

    Every arm's classifier weights show circle-of-fifths geometry, including
    CQT, whose centroids show none. Before reading that as a property of the
    representations, it has to be shown that it is not automatic. Keys a fifth
    apart share six of seven scale degrees, so they are confusable, and a
    sceptic can argue any competent 24-way key classifier ends up with
    fifths-structured weights whatever it was fitted on.

      random_feat : no information at all. Accuracy must fall to chance, and
                    the weights must lose the geometry. Establishes that label
                    structure alone does not produce it.
      orthocode   : perfect information, no tonal content. Each key gets its
                    own random orthogonal code, so classes are equidistant by
                    construction and carry no pitch-class relation. Accuracy
                    stays high, matching the real arms, while the fifths
                    structure is absent from the data. If the weights still
                    show fifths geometry here, the effect is an artefact of the
                    classifier and the label set, not of the representation.
    """
    if kind == "random_feat":
        return rng.normal(size=(len(Y), D))
    codes = np.linalg.qr(rng.normal(size=(D, len(present))))[0].T
    pos = {k: i for i, k in enumerate(present)}
    return codes[[pos[y] for y in Y]] + rng.normal(0, 0.35, (len(Y), D))


def main(arm):
    p = Path(f"runs/clipkey_{arm}.json")
    base = arm.split("@")[-1] if "@" in arm else "cqt"
    d = np.load(f"runs/clipkey_{base}.npz")
    Z, Y, present = d["Z"].astype(np.float64), d["Y"], d["keys"]
    if arm.startswith(("random_feat", "orthocode")):
        Z = synth(arm.split("@")[0], Y, present, np.random.default_rng(0))
        p = Path(f"runs/clipkey_{arm.split('@')[0]}.json")
        p.write_text(json.dumps({"arm": arm, "n_clips": int(len(Y)),
                                 "dim": Z.shape[1] // 2, "poolings":
                                 {"mean": {}, "mean+std": {}}}, indent=2))
    out = json.loads(p.read_text())
    kk, dfif, dchr, dmode, iu = reference_distances(present)
    rng = np.random.default_rng(0)
    D = Z.shape[1] // 2

    pools = [("mean", slice(0, D)), ("mean+std", slice(0, 2 * D))]
    if arm.startswith(("random_feat", "orthocode")):
        pools = [("mean", slice(0, Z.shape[1]))]
    for pool_name, sl in pools:
        ZP = Z[:, sl]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
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
        from sklearn.model_selection import cross_val_score
        acc = cross_val_score(clf, ZP, Y, cv=5, scoring="accuracy")
        out["poolings"].setdefault(pool_name, {})
        out["poolings"][pool_name].setdefault(
            "probe", {})["key24_acc_refit"] = float(acc.mean())
        out["poolings"][pool_name]["probe_weight_geometry"] = res
        print(f"[{arm}/{pool_name}] key24 acc {acc.mean():.3f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    p.write_text(json.dumps(out, indent=2))
    print("updated", p)


if __name__ == "__main__":
    main(sys.argv[1])
