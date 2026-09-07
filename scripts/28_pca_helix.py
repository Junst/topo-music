"""Is the octave structure visible to an unsupervised projection too?

Yagi et al. (ISMIR 2026) take principal components of foundation model
representations of isolated notes and report a helical arrangement reflecting
octave periodicity. That is a projection, so it is not in conflict with a null
under the native metric, but it does raise a question this paper should answer
rather than assume: does an *unsupervised* rank-4 projection expose the same
octave contrast that the key-supervised one does?

Two measurements, on the cached NSynth embeddings, with the same
instrument-disjoint discipline script 13 uses.

1. The strict octave contrast under a rank-4 PCA subspace, fitted on one half
   of the instruments and evaluated on the other. Directly comparable to the
   d_strict of the key-fitted subspace, since it is the same statistic, the
   same anchors and the same bootstrap.

2. Where the twelve-semitone periodicity actually sits. For each of the leading
   components, the pitch-centroid score is regressed on a constant, pitch
   height, and a cosine and sine of period twelve. A helix means one component
   is carried by the linear term while two more are carried by the periodic
   pair. Reporting the variance each component holds says whether the helix is
   a leading feature of the space or a small one.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader

m13 = SourceFileLoader("m13", "scripts/13_tonal_subspace.py").load_module()

ARMS = ["mert_L12", "muq_L6", "matpac_L6", "pupujepa", "encodec_32k",
        "cqt", "pq_stft", "chroma"]
DIMS = [2, 3, 4, 8]


def pca_directions(X, d):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d].T, (S ** 2) / (S ** 2).sum()


def helix_terms(Zp, pitch, keep, n_pc):
    """Split each component's pitch profile into height and chroma parts."""
    ps = np.array(sorted(set(pitch[keep].tolist())))
    C = np.stack([Zp[keep][pitch[keep] == p].mean(0) for p in ps])
    X = np.column_stack([np.ones(len(ps)), ps.astype(float),
                         np.cos(2 * np.pi * ps / 12),
                         np.sin(2 * np.pi * ps / 12)])
    rows = []
    for j in range(min(n_pc, C.shape[1])):
        y = C[:, j]
        y = (y - y.mean()) / (y.std() + 1e-12)
        full = np.linalg.lstsq(X, y, rcond=None)[0]
        r_full = 1 - ((y - X @ full) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lin = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
        r_lin = 1 - ((y - X[:, :2] @ lin) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(dict(pc=j + 1, r2_full=float(r_full),
                         r2_height=float(max(r_lin, 0.0)),
                         r2_chroma=float(max(r_full - r_lin, 0.0)),
                         amp_chroma=float(np.hypot(full[2], full[3])),
                         phase=float(np.arctan2(full[3], full[2]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n-pc", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = {}
    for arm in a.arms:
        p = Path(f"runs/nsynth_emb_{arm}.npz")
        if not p.exists():
            print(f"[{arm}] no cached embeddings, skipped", flush=True)
            continue
        t0 = time.time()
        nz = np.load(p, allow_pickle=True)
        Zn = nz["Z"].astype(np.float64)
        pitch, inst = nz["pitch"], nz["inst"]
        anchors, kmax = nz["anchors"], int(nz["kmax"])

        # the same instrument-disjoint split script 13 draws, same seed, so the
        # PCA subspace and the key-fitted subspace are read on the same notes
        rng = np.random.default_rng(a.seed)
        uinst = np.array(sorted(set(inst.tolist())))
        fit_inst = set(rng.choice(uinst, len(uinst) // 2, replace=False).tolist())
        fit_n = np.array([s in fit_inst for s in inst])
        curve_names = nz["insts"]
        eval_inst = np.array([i for i in sorted(set(anchors[:, 0].tolist()))
                              if curve_names[i] not in fit_inst])

        rec = {"D": int(Zn.shape[1]), "n_notes": int(len(Zn)),
               "n_eval_inst": int(len(eval_inst)), "pca": {}}
        for d in DIMS:
            W, evr = pca_directions(Zn[fit_n], d)
            cs, _ = curve_stats_wrapper(Zn @ W, anchors, kmax, eval_inst,
                                        a.seed, a.n_boot)
            rec["pca"][str(d)] = {
                "var_frac_nsynth": m13.var_fraction(W, Zn),
                "evr_top": [float(v) for v in evr[:d]],
                "nsynth_curve": cs}
            print(f"[{arm}] PCA d={d} d_strict={cs['d_strict']['v']:+.3f} "
                  f"[{cs['d_strict']['lo']:+.3f},{cs['d_strict']['hi']:+.3f}] "
                  f"var={rec['pca'][str(d)]['var_frac_nsynth']:.3f}", flush=True)

        W, evr = pca_directions(Zn[fit_n], a.n_pc)
        keep = np.array([curve_names[i] for i in range(len(curve_names))])
        keep_names = set(keep[eval_inst].tolist())
        sel = np.array([s in keep_names for s in inst])
        rec["helix"] = helix_terms(Zn @ W, pitch, sel, a.n_pc)
        rec["evr"] = [float(v) for v in evr[:a.n_pc]]
        for r in rec["helix"]:
            print(f"  PC{r['pc']} evr={evr[r['pc']-1]:.3f} "
                  f"R2height={r['r2_height']:.3f} R2chroma={r['r2_chroma']:.3f}",
                  flush=True)
        out[arm] = rec
        print(f"[{arm}] {time.time()-t0:.0f}s", flush=True)

    Path("runs/pca_helix.json").write_text(json.dumps(out, indent=2))
    print("wrote runs/pca_helix.json")


def curve_stats_wrapper(Z, anchors, kmax, eval_inst, seed, n_boot):
    return m13.curve_stats(Z, anchors, kmax, eval_inst,
                           np.random.default_rng(seed), n_boot=n_boot)


if __name__ == "__main__":
    main()
