"""What the rho_fifths curve looks like as geometry.

Fig 1(b) says the fifths correlation moves from +0.035 at one clip per key
centroid to +0.468 at all of them. That is a number about a 24x24 matrix, and
the matrix itself is more legible than the number: order the keys by position
on the circle of fifths and the ideal reference is a banded, wrapping pattern,
so the question is simply how much of that band is present at each sample size.

Every panel is rank-transformed over its own off-diagonal entries. That is
exactly what a Spearman correlation sees, so the panels are comparable to each
other and to the reference without any choice of scale on my part.

At n = 1, 8 and 32 the centroids depend on which clips were drawn, so ten draws
are taken and the one whose rho is the median is shown. Averaging the matrices
across draws would denoise them and destroy the very thing the figure is about.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

m9 = SourceFileLoader("m9", "scripts/09_clip_key_geometry.py").load_module()
ARM = "mert_L12"
NS = [1, 8, 32, 0]                       # 0 = every clip of that key
R = 10
INK, INK2 = "#1a1a19", "#55554e"
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

PC = ["C", "C$\\sharp$", "D", "D$\\sharp$", "E", "F",
      "F$\\sharp$", "G", "G$\\sharp$", "A", "A$\\sharp$", "B"]

gz = np.load(f"runs/clipkey_{ARM}.npz")
Z = gz["Z"].astype(np.float64); Z = Z[:, :Z.shape[1] // 2]
Y = gz["Y"]
present = np.array(sorted(set(Y.tolist())))
idx = {k: np.where(Y == k)[0] for k in present}

# keys ordered round the circle of fifths, majors then minors. Interleaving the
# two modes puts a 2x2 checkerboard on top of the band and hides it; blocking
# them keeps the fifths band readable inside each quadrant.
tonic = present // 2
fpos = (tonic * 7) % 12
order = np.lexsort((fpos, present % 2))
REF = m9.circ12(np.abs(fpos[order][:, None] - fpos[order][None, :]))
iu = np.triu_indices(len(present), 1)


def ranked(M):
    """Rank-transform the off-diagonal entries into [0, 1], keep it symmetric.

    Ties take the average rank, which matters for the reference: its distances
    take seven values, and breaking those ties arbitrarily would dither the
    bands into stripes that are an artefact of the sort order.
    """
    r = rankdata(M[iu])
    r = (r - r.min()) / (r.max() - r.min())
    out = np.zeros_like(M)
    out[iu] = r
    return out + out.T


def centroids(n, rng):
    return np.stack([Z[i if n == 0 or n >= len(i)
                       else rng.choice(i, n, replace=False)].mean(0)
                     for i in idx.values()])[order]


panels = []
for n in NS:
    draws = []
    for r in range(1 if n == 0 else R):
        D = squareform(pdist(centroids(n, np.random.default_rng(100 + r))))
        draws.append((float(spearmanr(REF[iu], D[iu]).statistic), D))
    draws.sort(key=lambda x: x[0])
    rho, D = draws[len(draws) // 2]          # median draw, not the best one
    panels.append((("all" if n == 0 else str(n)), rho, ranked(D)))
    print(f"n={('all' if n==0 else n):>3}  rho={rho:+.3f}"
          f"  (over {len(draws)} draws: {draws[0][0]:+.3f} to {draws[-1][0]:+.3f})",
          flush=True)
panels.append(("reference", 1.0, ranked(REF.astype(float))))

fig, axs = plt.subplots(1, len(panels), figsize=(6.9, 1.62))
for ax, (nm, rho, M) in zip(axs, panels):
    ax.imshow(M, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ttl = ("ideal circle of fifths" if nm == "reference"
           else f"{nm} clip{'' if nm == '1' else 's'} per key")
    ax.set_title(ttl, loc="left", fontsize=7, color=INK, pad=2.5)
    if nm != "reference":
        ax.text(0.5, -0.20, f"$\\rho={rho:+.3f}$", transform=ax.transAxes,
                ha="center", fontsize=7, color=INK2)
    tk = np.arange(0, 24, 4)
    ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xticklabels([PC[t] for t in tonic[order][tk]])
    ax.set_yticklabels([PC[t] for t in tonic[order][tk]] if ax is axs[0] else [])
    ax.tick_params(length=2, pad=1.5)
# the three panels Fig 1(c) uses, cached so the figure script stays cheap
pick = {p[0]: p for p in panels}
np.savez(FIG / "_key_matrices.npz",
         one=pick["1"][2], rho_one=pick["1"][1],
         eight=pick["8"][2], rho_eight=pick["8"][1],
         alle=pick["all"][2], rho_all=pick["all"][1],
         ref=panels[-1][2], tonic_order=tonic[order])

fig.savefig(FIG / "fig6_key_matrices.png"); fig.savefig(FIG / "fig6_key_matrices.pdf")
print("fig6 written", flush=True)
