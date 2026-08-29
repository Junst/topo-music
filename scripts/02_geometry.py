"""L1'-a / L1'-b: is an existing RVQ codebook already topographic, geometrically?

L1'-a  index-order test.  The project's premise is that code *indices* are an
       arbitrary permutation. That premise is measured here, not assumed.
       Statistic: Spearman rho between |i-j| and ||e_i - e_j||, plus the ratio
       of mean adjacent-index distance to mean pairwise distance (1.0 = index
       carries no geometry).

L1'-b  2D-embeddability.  How much of the codebook's neighbourhood structure
       survives being forced onto 2 dimensions? This upper-bounds what any
       topographic objective could achieve *without changing the encoder*:
       a free 2D embedding is strictly easier than a discrete grid.
       Statistics: TwoNN intrinsic dimension, PCA participation ratio,
       trustworthiness/continuity of the best 2D embedding at k=10.

Reference baseline: the same statistics on a random Gaussian codebook of
identical shape. That is what "no structure" looks like at this K and d.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topo.codecs import load, CODECS


def twonn_id(X: np.ndarray, discard: float = 0.1) -> float:
    """Facco et al. (2017) two-NN intrinsic dimension estimator."""
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    d = np.sort(D, axis=1)[:, :2]
    ok = d[:, 0] > 0
    mu = d[ok, 1] / d[ok, 0]
    mu = np.sort(mu)
    n = len(mu)
    keep = int(n * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - (np.arange(1, keep + 1) / n))
    return float((x @ y) / (x @ x))          # slope through the origin


def participation_ratio(X: np.ndarray) -> float:
    ev = PCA().fit(X).explained_variance_
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def continuity(X: np.ndarray, E: np.ndarray, k: int) -> float:
    """Continuity = trustworthiness with the two spaces swapped."""
    return float(trustworthiness(E, X, n_neighbors=k))


def best_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Best-case 2D layout. UMAP if available (nonlinear, neighbourhood-driven),
    else PCA. Reported as an upper bound on grid quality."""
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                         min_dist=0.0).fit_transform(X)
    except Exception:
        return PCA(n_components=2, random_state=seed).fit_transform(X)


def analyse(E: np.ndarray, k: int, seed: int) -> dict:
    K = len(E)
    idx = np.arange(K)
    Dg = squareform(pdist(E))
    Di = np.abs(idx[:, None] - idx[None, :]).astype(float)
    iu = np.triu_indices(K, 1)
    rho = float(spearmanr(Di[iu], Dg[iu]).statistic)

    adj = np.linalg.norm(E[1:] - E[:-1], axis=1).mean()
    allp = Dg[iu].mean()

    Z = best_2d(E, seed)
    return dict(
        n_codes=K, dim=int(E.shape[1]),
        index_spearman=rho,
        adjacent_over_mean=float(adj / allp),
        twonn_id=twonn_id(E),
        participation_ratio=participation_ratio(E),
        trustworthiness_2d=float(trustworthiness(E, Z, n_neighbors=k)),
        continuity_2d=continuity(E, Z, k),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", nargs="+", default=list(CODECS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/l1_geometry.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    results = {}
    for name in a.codecs:
        c = load(name)
        per_level = []
        for l, E in enumerate(c.codebooks):
            E = E.astype(np.float64)
            r = analyse(E, a.k, a.seed)
            R = rng.standard_normal(E.shape)          # matched-shape null
            r["null"] = analyse(R, a.k, a.seed)
            r["level"] = l
            per_level.append(r)
            print(f"[{name}] L{l}  K={r['n_codes']} d={r['dim']}  "
                  f"idx_rho={r['index_spearman']:+.4f}  "
                  f"adj/mean={r['adjacent_over_mean']:.3f}  "
                  f"ID={r['twonn_id']:.2f} (null {r['null']['twonn_id']:.2f})  "
                  f"PR={r['participation_ratio']:.2f}  "
                  f"trust2d={r['trustworthiness_2d']:.3f} "
                  f"(null {r['null']['trustworthiness_2d']:.3f})", flush=True)
        results[name] = per_level

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
