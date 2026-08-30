"""Octave equivalence under a nonlinear metric: shortest paths on a kNN graph.

The paper so far compares two metrics on the same NSynth recordings, the native
cosine one and the one induced by a rank-4 projection fitted on GiantSteps
tonics. Both are global and linear. That leaves a question open: when a relation
is absent from the ambient metric, is it absent from the representation, or only
from its *global* geometry?

This measures the same three registered octave statistics under a third metric
that involves no supervision and no linear map. A symmetric k-nearest-neighbour
graph is built on the embeddings with cosine distance on the edges, and the
distance between two notes is the shortest path along it. Nearby notes keep
their native distance; distant ones are reached only through the manifold.

Reading the three together:

  ambient ~ 0, geodesic ~ 0, projected >> 0
      the relation is neither globally nor locally expressed by the native
      metric and is only recoverable by supervised reweighting

  ambient ~ 0, geodesic >> 0, projected >> 0
      global distances obscure a relation that is already organised along the
      manifold, which makes the masking specifically global

  ambient ~ 0, geodesic >> 0, projected ~ 0
      the relation lives in nonlinear structure the tested linear map misses

Two things make this comparable to the existing numbers rather than merely
analogous. The anchors, the transposition set and the three statistics are the
ones script 10 and script 13 use, loaded from the same cached embeddings. And
because S(k) is built from a *similarity*, the geodesic distance enters
negated, so a shorter path at twelve semitones moves the contrast positive in
the same direction a higher cosine does.

Disconnected pairs are reported rather than imputed: a k too small leaves the
graph in pieces, and filling the gaps with a large constant would manufacture
the effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors

s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()

KS = (5, 10, 20, 50)


def knn_graph(Z, k):
    """Symmetric kNN graph with cosine distance on the edges."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Zn)
    d, idx = nn.kneighbors(Zn)
    n = len(Zn)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    vals = np.maximum(d[:, 1:].ravel(), 1e-9)
    g = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return g.maximum(g.T)                      # symmetric: an edge either way


def geodesic_similarity(Z, anchors, k):
    """Negated shortest-path distance for every anchor pair, plus coverage.

    Paths are computed from the base notes only, which is a few hundred
    sources rather than every node.
    """
    g = knn_graph(Z, k)
    ncomp, lab = connected_components(g, directed=False)
    base_rows = sorted({int(r) for i, a, kk, r in anchors if kk == 0})
    src = np.array(base_rows)
    D = shortest_path(g, method="D", directed=False, indices=src)
    pos = {r: j for j, r in enumerate(base_rows)}
    base = {(int(i), int(a)): int(r) for i, a, kk, r in anchors if kk == 0}
    sim, seen, miss = {}, 0, 0
    for i, a, kk, r in anchors:
        if kk == 0:
            continue
        d = D[pos[base[(int(i), int(a))]], int(r)]
        seen += 1
        if not np.isfinite(d):
            miss += 1
            continue
        sim[(int(i), int(a), int(kk))] = -float(d)
    return sim, dict(n_components=int(ncomp),
                     largest_component=int(np.bincount(lab).max()),
                     unreachable_pairs=miss, pairs=seen)


def curve_from_sim(sim, anchors, kmax, keep_inst):
    """Script 13's three statistics, on a supplied similarity rather than cosine."""
    ii = anchors[:, 0]
    insts = np.array(sorted(set(ii[np.isin(ii, keep_inst)].tolist())))
    pos = {v: j for j, v in enumerate(insts)}
    per = np.full((len(insts), kmax + 1), np.nan)
    acc = {}
    for (i, a, k), v in sim.items():
        if i not in pos:
            continue
        acc.setdefault((pos[i], k), []).append(v)
    for (j, k), v in acc.items():
        per[j, k] = float(np.mean(v))
    S = np.nanmean(per[:, 1:], axis=0)
    ks = np.arange(1, kmax + 1)

    def fit(sv, quad):
        cols = [np.ones_like(ks, float), ks.astype(float)]
        if quad:
            cols.append(ks.astype(float) ** 2)
        cols.append(np.cos(2 * np.pi * ks / 12))
        X = np.column_stack(cols)
        ok = ~np.isnan(sv)
        return np.linalg.lstsq(X[ok], sv[ok], rcond=None)[0]

    def stats(sv):
        at = lambda k: sv[k - 1]
        return dict(d12=float(at(12) - 0.5 * (at(11) + at(13))),
                    d_strict=float(at(12) - np.nanmax(sv[7:11])),
                    b2=float(fit(sv, False)[-1]))

    rng = np.random.default_rng(0)
    obs = stats(S)
    boot = [stats(np.nanmean(per[rng.integers(0, len(insts), len(insts)), 1:], 0))
            for _ in range(1000)]
    out = {}
    for key in obs:
        v = np.array([b[key] for b in boot])
        out[key] = dict(v=obs[key], lo=float(np.percentile(v, 2.5)),
                        hi=float(np.percentile(v, 97.5)))
    out["S"] = [None if np.isnan(x) else float(x) for x in S]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    a = ap.parse_args()

    z = np.load(f"runs/nsynth_emb_{a.arm}.npz", allow_pickle=True)
    Z = z["Z"].astype(np.float64)
    anchors, kmax = z["anchors"], int(z["kmax"])
    keep = np.array(sorted(set(anchors[:, 0].tolist())))
    print(f"[{a.arm}] {Z.shape[0]} notes, {len(keep)} instruments, kmax={kmax}",
          flush=True)

    out = {"arm": a.arm, "n_notes": int(Z.shape[0]), "geodesic": {}}
    amb, ambS = s13.curve_stats(Z, anchors, kmax, keep,
                                np.random.default_rng(0), n_boot=1000)
    out["ambient"] = {k: amb[k] for k in ("d12", "d_strict", "b2_chroma")}
    out["ambient"]["S"] = ambS
    print(f"  cosine      d_strict={amb['d_strict']['v']:+.4f} "
          f"[{amb['d_strict']['lo']:+.4f},{amb['d_strict']['hi']:+.4f}] "
          f"d12={amb['d12']['v']:+.4f}", flush=True)

    for k in a.ks:
        sim, info = geodesic_similarity(Z, anchors, k)
        st = curve_from_sim(sim, anchors, kmax, keep)
        out["geodesic"][str(k)] = dict(**{q: st[q] for q in ("d12", "d_strict", "b2")},
                                       S=st["S"], graph=info)
        print(f"  geodesic k={k:<3} d_strict={st['d_strict']['v']:+.4f} "
              f"[{st['d_strict']['lo']:+.4f},{st['d_strict']['hi']:+.4f}] "
              f"d12={st['d12']['v']:+.4f} b2={st['b2']['v']:+.4f} "
              f"| {info['n_components']} components, "
              f"{info['unreachable_pairs']}/{info['pairs']} pairs unreachable",
              flush=True)

    p = Path(f"runs/geodesic_{a.arm}.json")
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
