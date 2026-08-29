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
