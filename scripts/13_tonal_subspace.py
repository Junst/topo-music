"""Is tonal structure absent from the representation, or only from its metric?

Two results are in tension. At the note level MERT shows no octave equivalence:
transposing a real recording by an octave leaves it no more similar than
transposing by 11 or 13 semitones (Delta_strict <= 0). At the clip level the
same model's 24 key centroids lie on a circle of fifths as clearly as an
octave-folded chromagram does (rho_fifths ~ .45, and it survives a spectral
partial). Both cannot be statements about the same geometry.

The hypothesis this script tests is that they are not. Write

    z = z_chroma + z_rest,

where z_chroma is a low-dimensional linear subspace carrying pitch-class
identity. If Var(z_chroma) << Var(z_rest), then cosine distance in the ambient
space is dominated by z_rest and octave equivalence is invisible there, while
averaging thousands of frames within a key cancels much of z_rest and lets
z_chroma show through in the centroid geometry. That predicts three things,
each measured below:

  (i)  a projection onto a few directions should *recover* octave equivalence
       on notes the projection was never fitted on;
  (ii) that subspace should hold a small fraction of total variance;
  (iii) projecting it *out* should destroy the clip-level fifths geometry.

Circularity is the obvious threat, so the estimators are kept honest in two
ways. Instruments are disjoint between fitting and evaluation for anything
measured on NSynth. And the strongest cell is deliberately cross-dataset and
cross-task: a subspace fitted only to predict the tonic of GiantSteps *tracks*
is applied, unchanged, to the NSynth *note* transposition curve. Nothing about
single notes or octaves enters that fit.

Every learned subspace is compared against random subspaces of equal dimension,
because a d-dimensional projection of a 1024-dimensional space changes the
metric no matter what d is, and some of the effect is that alone.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from importlib.machinery import SourceFileLoader
m9 = SourceFileLoader("s09", str(Path(__file__).with_name(
    "09_clip_key_geometry.py"))).load_module()

DIMS = [int(x) for x in __import__("os").environ.get("TOPO_DIMS", "1,2,3,4,6,8,11").split(",")]
N_PERM = 2000
N_BOOT = 2000
GS = m9.GS


# ---------------------------------------------------------------- references
def reference_distances(present):
    iu = np.triu_indices(len(present), 1)
    P = m9.kk_profiles()[present]
    kk = 1.0 - np.corrcoef(P)[iu]
    ton = present // 2
    f = (ton * 7) % 12
    return (kk, m9.circ12(np.abs(f[:, None] - f[None, :]))[iu],
            m9.circ12(np.abs(ton[:, None] - ton[None, :]))[iu], iu)


def gs_key_labels():
    """Rebuild script 09's row order so GS split membership can be recovered."""
    lab, split = [], []
    for si, sp in enumerate(["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"]):
        for line in (GS / sp).read_text().splitlines():
            r = json.loads(line)
            km = m9.parse_key(r["label"])
            p = GS / Path(r["audio_path"]).relative_to("data/GS")
            if km is not None and p.exists():
                lab.append(km[0] * 2 + km[1]); split.append(si)
    return np.array(lab), np.array(split)


# ------------------------------------------------------------------ geometry
def key_geometry(Zg, Yg, rng, n_perm=N_PERM):
    present = np.array(sorted(set(Yg.tolist())))
    kk, dfif, dchr, iu = reference_distances(present)
    C = np.stack([Zg[Yg == k].mean(0) for k in present])
    dz = squareform(pdist(C))[iu]
    out = {}
    for nm, dd in [("kk", kk), ("fifths", dfif), ("chromatic", dchr)]:
        obs = float(spearmanr(dd, dz).statistic)
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = spearmanr(dd, squareform(
                pdist(C[rng.permutation(len(present))]))[iu]).statistic
        out[nm] = dict(rho=obs, z=float((obs - null.mean()) / (null.std() + 1e-12)),
                       p=float((np.abs(null) >= abs(obs)).mean()))
    return out


# ------------------------------------------------- transposition in a subspace
def curve_stats(Zn, anchors, kmax, keep_inst, rng, n_boot=N_BOOT):
    """Per-instrument transposition curve, then the three octave statistics.

    Same construction as script 10 -- the same anchors at every k, cosine
    similarity, instrument-level bootstrap -- so a projected curve is directly
    comparable to the ambient one rather than merely analogous.
    """
    Zn = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12)
    ii, ai, kk_, row = anchors.T
    ks = np.arange(1, kmax + 1)
    insts = np.array(sorted(set(ii[np.isin(ii, keep_inst)].tolist())))
    base = {}                                   # (inst, anchor) -> row at k=0
    for i, a_, k, r in anchors[kk_ == 0]:
        base[(i, a_)] = r
    per = np.full((len(insts), kmax + 1), np.nan)
    pos = {v: j for j, v in enumerate(insts)}
    acc = {}
    for i, a_, k, r in anchors:
        if k == 0 or i not in pos:
            continue
        acc.setdefault((pos[i], k), []).append(
            float(Zn[base[(i, a_)]] @ Zn[r]))
    for (j, k), v in acc.items():
        per[j, k] = float(np.mean(v))

    def _fit(sv, quad):
        cols = [np.ones_like(ks, float), ks.astype(float)]
        if quad:
            cols.append(ks.astype(float) ** 2)
        cols.append(np.cos(2 * np.pi * ks / 12))
        X = np.column_stack(cols)
        ok = ~np.isnan(sv)
        return np.linalg.lstsq(X[ok], sv[ok], rcond=None)[0]

    def stats(sv):
        at = lambda k: sv[k - 1]
        d12 = at(12) - 0.5 * (at(11) + at(13))
        d24 = (at(24) - 0.5 * (at(23) + at(25))) if kmax >= 25 else np.nan
        d_str = at(12) - np.nanmax(sv[7:11])   # same window as script 10
        b = _fit(sv, False)
        return np.array([d12, d24, d_str, b[1], _fit(sv, True)[-1]])

    S = np.nanmean(per[:, 1:], axis=0)
    obs = stats(S)
    boot = np.array([stats(np.nanmean(
        per[rng.integers(0, len(insts), len(insts)), 1:], axis=0))
        for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    names = ["d12", "d24", "d_strict", "b1_height", "b2_chroma"]
    return {n: dict(v=float(o), lo=float(l), hi=float(h),
                    sig=bool(l > 0 or h < 0))
            for n, o, l, h in zip(names, obs, lo, hi)}, S.tolist()


# ------------------------------------------------------------------ subspaces
def lda_directions(X, y, d):
    """Top-d discriminant directions, shrunk -- D >> n_class needs shrinkage."""
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X, y)
    W = lda.scalings_[:, :d]
    assert W.shape[1] == d, f"only {W.shape[1]} discriminants for d={d}"
    return np.linalg.qr(W)[0]                      # orthonormal basis


def var_fraction(W, X):
    Xc = X - X.mean(0)
    return float((np.linalg.norm(Xc @ W) ** 2) / (np.linalg.norm(Xc) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mert_L12")
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    nz = np.load(f"runs/nsynth_emb_{a.arm}.npz", allow_pickle=True)
    Zn = nz["Z"].astype(np.float64)
    pitch, inst = nz["pitch"], nz["inst"]
    anchors, kmax = nz["anchors"], int(nz["kmax"])
    gz = np.load(f"runs/clipkey_{a.arm}.npz")
    Zg_full = gz["Z"].astype(np.float64)
    Zg = Zg_full[:, :Zg_full.shape[1] // 2]        # the mean half only
    Yg = gz["Y"]
    lab, gsplit = gs_key_labels()
    assert len(lab) == len(Yg) and (lab == Yg).all(), "GS row order drifted"
    D = Zn.shape[1]
    assert Zg.shape[1] == D

    # instrument-disjoint fit/eval split for anything measured on NSynth
    uinst = np.array(sorted(set(inst.tolist())))
    fit_inst = set(rng.choice(uinst, len(uinst) // 2, replace=False).tolist())
    fit_n = np.array([s in fit_inst for s in inst])
    inst_id = anchors[:, 0]
    insts_in_curve = np.array(sorted(set(inst_id.tolist())))
    curve_names = nz["insts"]
    eval_inst = np.array([i for i in insts_in_curve
                          if curve_names[i] not in fit_inst])
    print(f"[{a.arm}] D={D} notes={len(Zn)} clips={len(Zg)} "
          f"fit-inst={len(fit_inst)} eval-inst={len(eval_inst)}", flush=True)

    out = {"arm": a.arm, "D": D, "dims": DIMS, "n_notes": int(len(Zn)),
           "n_clips": int(len(Zg)), "n_eval_inst": int(len(eval_inst)),
           "ambient": {}, "subspaces": {}}

    # ambient reference
    out["ambient"]["nsynth_curve"], _ = curve_stats(
        Zn, anchors, kmax, eval_inst, np.random.default_rng(a.seed))
    out["ambient"]["gs_geometry"] = key_geometry(Zg, Yg, np.random.default_rng(a.seed))
    print("  ambient done", f"{time.time()-t0:.0f}s", flush=True)

    # ---- estimators ------------------------------------------------------
    tr = gsplit < 2
    fits = {
        # octave-invariant by objective: octaves of one pitch class are pooled
        "pc_lda":    (Zn[fit_n], (pitch[fit_n] % 12)),
        # pitch height, not class -- the control that should NOT fold octaves
        "pitch_lda": (Zn[fit_n], pitch[fit_n]),
        # cross-dataset, cross-task: fitted on track tonic, never on notes
        "key_lda":   (Zg[tr], (Yg[tr] // 2)),
    }
    for nm, (X, y) in fits.items():
        out["subspaces"][nm] = {}
        for d in DIMS:
            if d > len(set(y.tolist())) - 1:
                continue
            W = lda_directions(X, y, d)
            r = {"var_frac_nsynth": var_fraction(W, Zn),
                 "var_frac_gs": var_fraction(W, Zg)}
            r["nsynth_curve"], _ = curve_stats(
                Zn @ W, anchors, kmax, eval_inst, np.random.default_rng(a.seed))
            gsel = np.ones(len(Yg), bool) if nm != "key_lda" else (gsplit == 2)
            r["gs_geometry"] = key_geometry(Zg[gsel] @ W, Yg[gsel],
                                            np.random.default_rng(a.seed))
            # (iii) remove the subspace and ask whether the geometry goes with it
            r["gs_geometry_complement"] = key_geometry(
                Zg[gsel] - (Zg[gsel] @ W) @ W.T, Yg[gsel],
                np.random.default_rng(a.seed))
            out["subspaces"][nm][str(d)] = r
            print(f"  {nm:<10} d={d:<3} var={r['var_frac_gs']*100:5.2f}%gs "
                  f"d12={r['nsynth_curve']['d12']['v']:+.3f} "
                  f"dstr={r['nsynth_curve']['d_strict']['v']:+.3f} "
                  f"rho5={r['gs_geometry']['fifths']['rho']:+.3f} "
                  f"rho5_comp={r['gs_geometry_complement']['fifths']['rho']:+.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- random-subspace control ----------------------------------------
    out["subspaces"]["random"] = {}
    for d in DIMS:
        rs = []
        for _ in range(a.n_random):
            W = np.linalg.qr(rng.normal(size=(D, d)))[0]
            cs, _ = curve_stats(Zn @ W, anchors, kmax, eval_inst,
                                np.random.default_rng(a.seed), n_boot=200)
            g = key_geometry(Zg @ W, Yg, np.random.default_rng(a.seed), n_perm=200)
            rs.append([cs["d12"]["v"], cs["d_strict"]["v"],
                       g["fifths"]["rho"], g["kk"]["rho"], var_fraction(W, Zg)])
        A = np.array(rs)
        out["subspaces"]["random"][str(d)] = {
            k: dict(mean=float(A[:, i].mean()), std=float(A[:, i].std()),
                    lo=float(np.percentile(A[:, i], 5)),
                    hi=float(np.percentile(A[:, i], 95)))
            for i, k in enumerate(["d12", "d_strict", "fifths_rho", "kk_rho",
                                   "var_frac_gs"])}
        print(f"  random     d={d:<3} d12={A[:,0].mean():+.3f} "
              f"dstr={A[:,1].mean():+.3f} rho5={A[:,2].mean():+.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    p = Path(f"runs/subspace_{a.arm}.json")
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
