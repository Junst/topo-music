"""Look at the spaces, not just the statistics.

Three views, all from the cached embeddings.

  fig3  NSynth notes in 2-D, ambient vs the key-fitted projection, coloured
        twice: once by pitch class and once by pitch height. The claim under
        test is an axis swap -- ambient geometry should be organised by height
        and not by class, the projection the other way round. PCA is used for
        the main panels because it is linear and has no free parameters, so the
        comparison cannot be tuned; t-SNE is shown beneath as a check that the
        same structure survives a nonlinear view.

  fig4  The 24 GiantSteps key centroids laid out by metric MDS, with edges
        drawn between keys a perfect fifth apart. If the circle of fifths is in
        the metric, those edges connect neighbours; if it is not, they cross.
        Repeated over the number of clips per centroid, which is what
        mechanism 1 says should make the ring condense out of noise.

  fig5  The distributions the octave statistic summarises: cosine similarity
        for note pairs 12 semitones apart against 11 and 13, ambient and
        projected. Delta_strict is a difference of means; this shows whether
        the underlying distributions actually separate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE

s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()
ARM = "mert_L12"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# ---- the projection, refitted exactly as script 13 fits it -----------------
gz = np.load(f"runs/clipkey_{ARM}.npz")
Zg = gz["Z"].astype(np.float64); Zg = Zg[:, :Zg.shape[1] // 2]
Yg = gz["Y"]
lab, gsplit = s13.gs_key_labels()
assert (lab == Yg).all()
W = s13.lda_directions(Zg[gsplit < 2], Yg[gsplit < 2] // 2, 4)

nz = np.load(f"runs/nsynth_emb_{ARM}.npz", allow_pickle=True)
Zn = nz["Z"].astype(np.float64); pitch = nz["pitch"]
pc = pitch % 12
np.save("runs/figs/_W_key_lda_d4.npy", W)


def scatter(ax, X, c, cmap, title, cyclic):
    ax.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, s=2.5, linewidths=0, alpha=.75,
               rasterized=True)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_color(GRID)
    return ax


P_amb = PCA(2).fit_transform(Zn - Zn.mean(0))
P_prj = PCA(2).fit_transform(Zn @ W - (Zn @ W).mean(0))
fig, axs = plt.subplots(2, 4, figsize=(9.0, 4.6))
for j, (X, nm) in enumerate([(P_amb, "ambient (1024-d)"), (P_prj, "projected (d=4)")]):
    scatter(axs[0][2 * j], X, pc, "twilight", f"PCA, {nm}\ncolour = pitch class", True)
    scatter(axs[0][2 * j + 1], X, pitch, "viridis", f"PCA, {nm}\ncolour = pitch height", False)
print("PCA done, running t-SNE", flush=True)
T_amb = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform(Zn)
T_prj = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform(Zn @ W)
for j, (X, nm) in enumerate([(T_amb, "ambient"), (T_prj, "projected")]):
    scatter(axs[1][2 * j], X, pc, "twilight", f"t-SNE, {nm}\ncolour = pitch class", True)
    scatter(axs[1][2 * j + 1], X, pitch, "viridis", f"t-SNE, {nm}\ncolour = pitch height", False)
fig.tight_layout()
fig.savefig(FIG / "fig3_axis_swap.png"); fig.savefig(FIG / "fig3_axis_swap.pdf")
print("fig3 written", flush=True)
np.savez("runs/figs/_fig3_coords.npz", P_amb=P_amb, P_prj=P_prj,
         T_amb=T_amb, T_prj=T_prj, pitch=pitch)


# ============================ fig 4: the ring, one mode at a time ==========
# A 24-point MDS mixes major and minor into one cloud and the fifths edges
# tangle even where the correlation is high, so each mode is laid out on its
# own: 12 points, and a ring either appears or it does not.
print("fig4: key centroids", flush=True)
present = np.array(sorted(set(Yg.tolist())))
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
idx = {k: np.where(Yg == k)[0] for k in present}
rng = np.random.default_rng(0)


def layout(Z, keys, n=0):
    C = np.stack([Z[i if n == 0 or n >= len(i)
                    else rng.choice(i, n, replace=False)].mean(0)
                  for i in (idx[k] for k in keys)])
    return MDS(2, dissimilarity="precomputed", random_state=0,
               normalized_stress="auto").fit_transform(squareform(pdist(C)))


def ring(ax, X, keys, title, col):
    ton = keys // 2
    order = {t: i for i, t in enumerate(ton)}
    for t in ton:                       # connect each key to the one a fifth up
        j = order.get((t + 7) % 12)
        if j is not None:
            ax.plot(*zip(X[order[t]], X[j]), color=col, lw=1.1, alpha=.6, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=18, c=col, zorder=3, edgecolors="white",
               linewidths=.7)
    for i, t in enumerate(ton):
        ax.annotate(NAMES[t], X[i], fontsize=6, xytext=(0, 5),
                    textcoords="offset points", ha="center", color=INK2, zorder=4)
    ax.set_title(title, loc="left", color=INK, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color(GRID)


cq = np.load("runs/clipkey_cqt.npz"); Zc = cq["Z"].astype(np.float64)
Zc = Zc[:, :Zc.shape[1] // 2]
ch = np.load("runs/clipkey_chroma.npz"); Zh = ch["Z"].astype(np.float64)
Zh = Zh[:, :Zh.shape[1] // 2]
ARMS4 = [(Zh, "chromagram"), (Zc, "log-CQT"), (Zg, "MERT L12")]
maj = present[present % 2 == 0]
mino = present[present % 2 == 1]
fig, axs = plt.subplots(2, 4, figsize=(8.6, 4.6))
for c, (Z2, nm) in enumerate(ARMS4):
    ring(axs[0][c], layout(Z2, maj), maj, f"{nm} · major", BLUE)
    ring(axs[1][c], layout(Z2, mino), mino, f"{nm} · minor", ORANGE)
for r, (keys, col, nm) in enumerate([(maj, BLUE, "major"), (mino, ORANGE, "minor")]):
    ring(axs[r][3], layout(Zg, keys, 4), keys, f"MERT L12 · {nm}, 4 clips/key", col)
fig.tight_layout()
fig.savefig(FIG / "fig4_key_rings.png"); fig.savefig(FIG / "fig4_key_rings.pdf")
print("fig4 written", flush=True)


# ============ fig 5: the distributions the octave statistic reduces =========
# An earlier version of this figure also plotted mean centroid distance against
# fifths distance and against semitone distance. It was dropped: a pair of keys
# t semitones apart is always circ12(7t) steps apart on the circle of fifths,
# so the two panels held the same seven groups in two orders and the difference
# between the arms lives in the ordering rather than in the magnitudes, which a
# plot of group means shows poorly. What follows plots the note-level
# similarities directly, where the effect is a visible separation rather than a
# rank.
print("fig5: distributions", flush=True)
Zn_n = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12)
Zp = Zn @ W
Zp_n = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
anch = nz["anchors"]
kmax = int(nz["kmax"])
base = {(i, a_): r for i, a_, k, r in anch if k == 0}
byk = {}
for i, a_, k, r in anch:
    byk.setdefault(k, []).append((base[(i, a_)], r))

fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.5))
for ax, (Zx, nm) in zip(axs[:2], [(Zn_n, "(a) ambient, 1024-d"),
                                  (Zp_n, "(b) projected, d = 4")]):
    for k, col, ls in [(11, AQUA, "--"), (12, BLUE, "-"), (13, ORANGE, ":")]:
        v = np.array([Zx[a] @ Zx[b] for a, b in byk[k]])
        h, e_ = np.histogram(v, bins=55, range=(-.3, 1.0), density=True)
        ax.plot((e_[:-1] + e_[1:]) / 2, h, color=col, ls=ls, lw=1.6,
                label=f"{k} semitones apart")
    ax.set_xlabel("cosine similarity between a note and its transposition")
    ax.grid(True, color=GRID, lw=.5); ax.set_axisbelow(True)
    ax.set_title(nm, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("density")
axs[1].legend(frameon=False, loc="upper left", handlelength=2.0)

ks = np.arange(1, kmax + 1)
for Zx, col, ls, nm in [(Zn_n, AQUA, "--", "ambient"), (Zp_n, BLUE, "-", "projected")]:
    S = np.array([np.mean([Zx[a] @ Zx[b] for a, b in byk[k]]) for k in ks])
    S = (S - S.mean()) / S.std()
    axs[2].plot(ks, S, color=col, ls=ls, lw=1.6, marker="o", ms=2.6, label=nm)
for k in (12, 24):
    axs[2].axvline(k, color=INK2, lw=.7, ls=":", zorder=1)
axs[2].set_xlabel("transposition (semitones)")
axs[2].set_ylabel("mean similarity (z)")
axs[2].set_title("(c) the octave peak appears", loc="left", color=INK)
axs[2].grid(True, color=GRID, lw=.5); axs[2].set_axisbelow(True)
axs[2].legend(frameon=False, loc="lower left", handlelength=2.0)
for sp in ("top", "right"):
    axs[2].spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig5_distributions.png"); fig.savefig(FIG / "fig5_distributions.pdf")
print("fig5 written", flush=True)
