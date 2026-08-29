"""L1'-c / L1'-d: is the *musical* content of a codebook geometrically organised?

L1'-a showed (or not) that code indices carry no geometry. That is the weak
claim. The sharp claim is semantic:

  L1'-c  Each code has a preferred pitch and a preferred instrument family.
         Is that preference field smooth over the codebook's own geometry, and
         over the best 2D layout of it? Smoothness is Moran's I on a kNN graph,
         against a label-permutation null.

         Three outcomes, decided in PREREG.md before running:
           already smooth  -> the codebook is topographic; project dies.
           informative but not smooth -> the information exists, the geometry
                                         does not. This is the case the project
                                         needs.
           not informative -> codes do not carry pitch at all; the probe, not
                              the codebook, is the thing to fix.

  L1'-d  Does RVQ depth already factorise? Normalised MI(code; pitch) and
         MI(code; family) per level. The hypothesis is coarse levels carry
         pitch/harmony and deep levels carry timbre -- which, if true, gives a
         factorisation axis for free, with no augmentation and no labels.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topo.codecs import load

MIN_COUNT = 20          # codes seen fewer times than this are dropped


def norm_mi(counts: np.ndarray) -> float:
    """Normalised MI between code identity and label, from a [K, C] count table.
    Normalisation is by H(label), so 1.0 = code determines the label."""
    P = counts / max(counts.sum(), 1)
    pk, pc = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = P * (np.log(P) - np.log(pk) - np.log(pc))
    mi = np.nansum(np.where(P > 0, t, 0.0))
    hc = -np.nansum(np.where(pc > 0, pc * np.log(pc), 0.0))
    return float(mi / hc) if hc > 0 else 0.0


def selectivity(counts: np.ndarray) -> np.ndarray:
    """Per-code 1 - H(label|code)/H(label). 1 = code fires for one label only."""
    P = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    tot = counts.sum(0)
    q = tot / max(tot.sum(), 1)
    h0 = -np.nansum(np.where(q > 0, q * np.log(q), 0.0))
    return 1.0 - h / h0 if h0 > 0 else np.zeros(len(counts))


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    z = x - x.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return 0.0
    return float(len(x) / W.sum() * (z @ W @ z) / denom)


def knn_W(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, ind = nn.kneighbors(X)
    W = np.zeros((len(X), len(X)))
    for i, row in enumerate(ind):
        W[i, row[1:]] = 1.0
    return np.maximum(W, W.T)


def moran_with_null(x: np.ndarray, W: np.ndarray, n_perm: int, rng) -> dict:
    obs = morans_i(x, W)
    null = np.array([morans_i(rng.permutation(x), W) for _ in range(n_perm)])
    return dict(I=obs, null_mean=float(null.mean()), null_std=float(null.std()),
                z=float((obs - null.mean()) / (null.std() + 1e-12)),
                p=float((null >= obs).mean()))


def best_2d(X, seed):
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                         min_dist=0.0).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.probe, allow_pickle=True)
    codec = str(d["codec"])
    pc, fc = d["pitch_counts"], d["fam_counts"]
    pitches = d["pitches"]
    E_all = load(codec).codebooks
    rng = np.random.default_rng(a.seed)

    out = {"codec": codec, "n_clips": int(d["n_clips"]), "levels": []}
    for l in range(pc.shape[0]):
        E = E_all[l].astype(np.float64)
        used = pc[l].sum(1) >= MIN_COUNT
        rec = dict(level=l, n_codes=int(len(E)), n_used=int(used.sum()),
                   usage_frac=float(used.mean()),
                   nmi_pitch=norm_mi(pc[l][used]), nmi_family=norm_mi(fc[l][used]))

        if used.sum() >= 50:
            Eu = E[used]
            sel_p = selectivity(pc[l][used])
            pref_p = pitches[pc[l][used].argmax(1)].astype(float)
            pref_f = fc[l][used].argmax(1).astype(float)
            rec["median_pitch_selectivity"] = float(np.median(sel_p))

            W_code = knn_W(Eu, a.k)                       # codebook's own geometry
            Z = best_2d(Eu, a.seed)
            W_2d = knn_W(Z, a.k)                          # best 2D layout
            idx = np.arange(used.sum()).reshape(-1, 1).astype(float)
            W_idx = knn_W(idx, 2)                         # index order

            rec["moran_pitch_codespace"] = moran_with_null(pref_p, W_code, a.n_perm, rng)
            rec["moran_pitch_2d"] = moran_with_null(pref_p, W_2d, a.n_perm, rng)
            rec["moran_pitch_index"] = moran_with_null(pref_p, W_idx, a.n_perm, rng)
            rec["moran_family_codespace"] = moran_with_null(pref_f, W_code, a.n_perm, rng)

            # octave equivalence: does chroma distance explain code distance
            # beyond absolute pitch distance?
            strong = sel_p >= np.quantile(sel_p, 0.75)
            if strong.sum() >= 50:
                Es, ps = Eu[strong], pref_p[strong]
                iu = np.triu_indices(len(Es), 1)
                dc = squareform(pdist(Es))[iu]
                dp = np.abs(ps[:, None] - ps[None, :])[iu]
                dchr = np.minimum(dp % 12, 12 - (dp % 12))
                rec["octave"] = dict(
                    n=int(strong.sum()),
                    rho_abs_pitch=float(spearmanr(dp, dc).statistic),
                    rho_chroma=float(spearmanr(dchr, dc).statistic),
                    rho_chroma_partial=float(
                        spearmanr(dchr - np.polyval(np.polyfit(dp, dchr, 1), dp), dc).statistic),
                )
        out["levels"].append(rec)

        m = rec.get("moran_pitch_codespace", {})
        print(f"[{codec}] L{l}  used={rec['n_used']}/{rec['n_codes']}  "
              f"nMI(pitch)={rec['nmi_pitch']:.3f}  nMI(fam)={rec['nmi_family']:.3f}  "
              f"MoranI(pitch|codespace)={m.get('I', float('nan')):.3f} "
              f"z={m.get('z', float('nan')):.1f}", flush=True)

    p = Path(a.out or f"runs/topo_{codec}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
