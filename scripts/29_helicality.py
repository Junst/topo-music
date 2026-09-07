"""Helicality and octave equivalence, measured on the same notes.

Yagi et al. (ISMIR 2026) fit a parametric helix to principal components of
note representations and score it by the inverse mean squared error. A helix
places octaves at the same angle but at different heights, so a high score is
not a statement about octave-equivalent distances: the model contains an
explicit height axis that separates C3 from C4. This script measures both
quantities on the same embeddings and the same instruments, so the relation
between them is read off rather than argued.

Their helix, for the semitone-ordered pitch index p:

    y(p) = h(p) c + r(p) (cos th(p) u + sin th(p) v)
    h(p) = h_pitch p + h_0,  r(p) = r_slope p + r_0,  th(p) = w (p - p_0)

with {c,u,v} an orthonormal basis of R^3. Yagi search all nine parameters with
Optuna. The fit is cheaper than that: writing R for the orthogonal matrix with
columns (u,v,c), the model is R g(p) with g(p) the helix in canonical
coordinates, so for a fixed w the optimal R is an orthogonal Procrustes
solution and the four scalars then follow in closed form. Alternating the two
converges in a few steps. The phase p_0 needs no search of its own: shifting it
rotates the (u,v) plane, which R already spans. Only w is gridded, over the
range Yagi state, |w| in [pi/6, pi/2].

Helicality as defined is an inverse MSE and therefore depends on the scale of
the embedding, which is fine within one model but not across representations of
different width and norm. We report it for faithfulness and alongside it the
scale-free reading of the same fit, the fraction of pitch-centroid variance the
helix explains.
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader
from scipy.stats import spearmanr

m13 = SourceFileLoader("m13", "scripts/13_tonal_subspace.py").load_module()

ARMS = ["mert_L12", "muq_L6", "matpac_L6", "pupujepa", "encodec_32k",
        "cqt", "pq_stft", "chroma"]
N_KEY = 36                     # 3 octaves, the pitch span Yagi et al. use
N_PC = 5                       # top five components, ten triples
W_GRID = np.linspace(np.pi / 6, np.pi / 2, 41)


def fit_helix(X, n_iter=40):
    """Best helix fit to X, an [N,3] pitch-ordered point set. Returns MSE."""
    p = np.arange(1, len(X) + 1, dtype=float)
    Xc = X - X.mean(0)
    A_h = np.column_stack([p, np.ones_like(p)])
    best = np.inf
    for w in W_GRID:
        th = w * p
        cs, sn = np.cos(th), np.sin(th)
        # start from the identity frame and alternate
        R = np.eye(3)
        for _ in range(n_iter):
            Y = Xc @ R                                  # canonical coordinates
            rad = Y[:, 0] * cs + Y[:, 1] * sn           # radial component
            r_coef = np.linalg.lstsq(A_h, rad, rcond=None)[0]
            h_coef = np.linalg.lstsq(A_h, Y[:, 2], rcond=None)[0]
            r = A_h @ r_coef
            G = np.column_stack([r * cs, r * sn, A_h @ h_coef])
            U, _, Vt = np.linalg.svd(G.T @ Xc)
            R_new = (U @ Vt).T                          # Procrustes, X ~ G R^T
            if np.allclose(R_new, R, atol=1e-9):
                R = R_new
                break
            R = R_new
        mse = float(((Xc - G @ R.T) ** 2).sum(1).mean())
        best = min(best, mse)
    return best


def helicality(Z, pitch, rng=None):
    """Yagi's layer-max Helicality on one instrument, plus the scale-free fit.

    PCA over the pitch-ordered vectors, then the best of the ten three-component
    projections. `rng` shuffles the pitch order, which keeps the point cloud and
    destroys only its correspondence with pitch: the reference this is read
    against.
    """
    order = np.argsort(pitch)
    Zp = Z[order]
    if rng is not None:
        Zp = Zp[rng.permutation(len(Zp))]
    Zc = Zp - Zp.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:N_PC].T                                # [N, 5]
    best_mse, best_r2 = np.inf, -np.inf
    for tri in combinations(range(N_PC), 3):
        X = P[:, list(tri)]
        mse = fit_helix(X)
        var = float(((X - X.mean(0)) ** 2).sum(1).mean())
        best_mse = min(best_mse, mse)
        best_r2 = max(best_r2, 1.0 - mse / (var + 1e-12))
    return 1.0 / (best_mse + 1e-12), best_r2


def per_instrument_dstrict(Zn, anchors, kmax, keep_inst):
    """The strict octave contrast of each instrument's own curve.

    Same construction as script 13's curve_stats, kept per instrument instead
    of averaged, so it can be paired with that instrument's Helicality.
    """
    Zn = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12)
    ii, ai, kk_, row = anchors.T
    insts = np.array(sorted(set(ii[np.isin(ii, keep_inst)].tolist())))
    pos = {v: j for j, v in enumerate(insts)}
    base = {(i, a_): r for i, a_, k, r in anchors[kk_ == 0]}
    acc = {}
    for i, a_, k, r in anchors:
        if k == 0 or i not in pos:
            continue
        acc.setdefault((pos[i], k), []).append(float(Zn[base[(i, a_)]] @ Zn[r]))
    per = np.full((len(insts), kmax + 1), np.nan)
    for (j, k), v in acc.items():
        per[j, k] = float(np.mean(v))
    sv = per[:, 1:]
    return insts, sv[:, 11] - np.nanmax(sv[:, 7:11], axis=1)


def selftest():
    """The fitter has to recover a helix it is given and reject one it is not."""
    rng = np.random.default_rng(0)
    p = np.arange(1, N_KEY + 1, dtype=float)
    for w_true in (np.pi / 6, np.pi / 4, np.pi / 3):
        th = w_true * p
        G = np.column_stack([(0.2 * p + 3) * np.cos(th),
                             (0.2 * p + 3) * np.sin(th), 0.5 * p])
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]     # arbitrary frame
        X = G @ Q.T + rng.normal(size=(3,))
        var = float(((X - X.mean(0)) ** 2).sum(1).mean())
        r2 = 1.0 - fit_helix(X) / var
        print(f"  exact helix w={w_true:.3f}: R2={r2:.4f}", flush=True)
        assert r2 > 0.99, r2
    # a helix plus noise degrades smoothly
    th = (np.pi / 6) * p
    G = np.column_stack([3 * np.cos(th), 3 * np.sin(th), 0.5 * p])
    for sd in (0.1, 0.5, 1.0):
        X = G + rng.normal(0, sd, G.shape)
        var = float(((X - X.mean(0)) ** 2).sum(1).mean())
        print(f"  helix + noise {sd}: R2={1.0 - fit_helix(X) / var:.4f}", flush=True)
    # isotropic noise must not look helical
    r2s = []
    for _ in range(20):
        X = rng.normal(size=(N_KEY, 3))
        var = float(((X - X.mean(0)) ** 2).sum(1).mean())
        r2s.append(1.0 - fit_helix(X) / var)
    print(f"  random points: R2 mean {np.mean(r2s):.4f} max {np.max(r2s):.4f}",
          flush=True)
    # a circle with no height axis: octave-equivalent, and still fittable, which
    # is the point -- the model allows h_pitch = 0
    X = np.column_stack([3 * np.cos(th), 3 * np.sin(th), np.zeros_like(p)])
    var = float(((X - X.mean(0)) ** 2).sum(1).mean())
    print(f"  flat circle (no height): R2={1.0 - fit_helix(X) / var:.4f}",
          flush=True)
    print("selftest passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    out, dstrict_by_arm = {}, {}
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
        names = nz["insts"]

        # the instrument split scripts 13 and 28 draw, same seed
        rng = np.random.default_rng(a.seed)
        uinst = np.array(sorted(set(inst.tolist())))
        fit_inst = set(rng.choice(uinst, len(uinst) // 2, replace=False).tolist())
        eval_inst = np.array([i for i in sorted(set(anchors[:, 0].tolist()))
                              if names[i] not in fit_inst])

        insts, dstr = per_instrument_dstrict(Zn, anchors, kmax, eval_inst)
        dmap = {names[i]: float(v) for i, v in zip(insts, dstr)}

        rows, nrng = [], np.random.default_rng(a.seed + 1)
        for i in insts:
            s = names[i]
            sel = inst == s
            ps = np.array(sorted(set(pitch[sel].tolist())))
            runs, cur = [], [ps[0]]
            for x, y in zip(ps[:-1], ps[1:]):
                if y == x + 1:
                    cur.append(y)
                else:
                    runs.append(cur); cur = [y]
            runs.append(cur)
            run = max(runs, key=len)
            if len(run) < N_KEY:
                continue
            o = (len(run) - N_KEY) // 2                 # centred 3-octave window
            keep = set(run[o:o + N_KEY])
            m = sel & np.isin(pitch, list(keep))
            # one vector per pitch, so the fit sees a pitch-ordered point set
            pv = np.array(sorted(keep))
            Zi = np.stack([Zn[m & (pitch == q)].mean(0) for q in pv])
            hel, r2 = helicality(Zi, pv)
            hel0, r20 = helicality(Zi, pv, rng=nrng)
            rows.append(dict(inst=s, helicality=hel, r2=r2,
                             helicality_shuffled=hel0, r2_shuffled=r20,
                             d_strict=dmap.get(s)))

        H = np.array([r["helicality"] for r in rows])
        R2 = np.array([r["r2"] for r in rows])
        R20 = np.array([r["r2_shuffled"] for r in rows])
        Dv = np.array([r["d_strict"] for r in rows], dtype=float)
        ok = ~np.isnan(Dv)
        rho = float(spearmanr(H[ok], Dv[ok]).statistic) if ok.sum() > 3 else float("nan")
        rho2 = float(spearmanr(R2[ok], Dv[ok]).statistic) if ok.sum() > 3 else float("nan")
        out[arm] = dict(n_inst=len(rows),
                        helicality_mean=float(H.mean()),
                        helicality_median=float(np.median(H)),
                        r2_mean=float(R2.mean()),
                        r2_shuffled_mean=float(R20.mean()),
                        d_strict_mean=float(np.nanmean(Dv)),
                        frac_inst_positive=float((Dv[ok] > 0).mean()),
                        rho_helicality_dstrict=rho,
                        rho_r2_dstrict=rho2,
                        per_instrument=rows)
        dstrict_by_arm[arm] = dmap
        print(f"[{arm}] n={len(rows)}  helicality {H.mean():.3f} "
              f"(median {np.median(H):.3f})  R2 {R2.mean():.3f} "
              f"(shuffled {R20.mean():.3f})  d_strict {np.nanmean(Dv):+.3f}  "
              f"pos {100*(Dv[ok] > 0).mean():.0f}%  "
              f"rho(hel,d)={rho:+.3f}  rho(R2,d)={rho2:+.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)

    # do the same instruments carry octave structure in different arms?
    keys = list(dstrict_by_arm)
    cross = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a1, a2 = keys[i], keys[j]
            shared = sorted(set(dstrict_by_arm[a1]) & set(dstrict_by_arm[a2]))
            if len(shared) > 3:
                v1 = [dstrict_by_arm[a1][s] for s in shared]
                v2 = [dstrict_by_arm[a2][s] for s in shared]
                cross[f"{a1}|{a2}"] = float(spearmanr(v1, v2).statistic)
    print("\nper-instrument d_strict agreement between arms:")
    for k, v in sorted(cross.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<28} rho={v:+.3f}")

    Path("runs/helicality.json").write_text(json.dumps(
        {"arms": out, "cross_arm_dstrict_rho": cross}, indent=2))
    print("\nwrote runs/helicality.json")


if __name__ == "__main__":
    main()
