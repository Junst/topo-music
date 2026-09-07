"""Figure 2: what is measured on what, for every representation.

Three things this has to make visible, none of which prose carries well.

Which corpus feeds which measurement. Key is measured on music clips, pitch on
isolated notes, and the same nine representations serve both.

That the rank-4 subspace crosses between them. It is fitted on clip-level key
labels and applied, unchanged, to notes it never saw. That arrow is the only
link between the two halves.

That helical organization is not a fifth metric. The four metrics recompute one
statistic, the octave contrast, under different notions of distance. The helix
fit asks a different question of the same points, so it sits beside them rather
than after them.

Model names are left to the method text. Boxes are sized from the rendered text
rather than by hand, since at this size a character-count estimate puts labels
outside their frames.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2 = "#1a1a19", "#55554e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
PANEL, GROUP = "#f2f2ee", "#fafaf7"
FS = 5.0
PAD_X, PAD_Y = 2.4, 2.0
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"text.color": INK, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.015})

fig, ax = plt.subplots(figsize=(3.42, 1.88))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
fig.canvas.draw()
inv = ax.transData.inverted()
B = {}


def node(name, cx, cy, text, ec=INK2, fc="#ffffff", fs=FS, tc=None, ls="-"):
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                zorder=4, color=tc or INK, linespacing=1.35)
    fig.canvas.draw()
    bb = t.get_window_extent()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    x0, y0, x1, y1 = x0 - PAD_X, y0 - PAD_Y, x1 + PAD_X, y1 + PAD_Y
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                linewidth=0.6, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    B[name] = (x0, y0, x1, y1)
    return B[name]


def panel(names, pad=2.6, top=5.4, fc=GROUP, ec="#d6d6d0"):
    """Group frame with a strip at the top for its name."""
    xs = [B[n] for n in names]
    x0 = min(b[0] for b in xs) - pad; x1 = max(b[2] for b in xs) + pad
    y0 = min(b[1] for b in xs) - pad; y1 = max(b[3] for b in xs) + top
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=2",
                                linewidth=0.6, edgecolor=ec, facecolor=fc,
                                zorder=1))
    return x0, y0, x1, y1


def side(name, w):
    x0, y0, x1, y1 = B[name]
    return {"l": (x0, (y0 + y1) / 2), "r": (x1, (y0 + y1) / 2),
            "t": ((x0 + x1) / 2, y1), "b": ((x0 + x1) / 2, y0)}[w]


def arrow(p0, p1, color=INK2, lw=0.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5,
                                 linewidth=lw, color=color, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.0, shrinkB=1.0))


# ---- corpora ---------------------------------------------------------------
node("gs", 14, 44, "GiantSteps, FMAK\nclips")
node("ns", 14, 11, "NSynth\nisolated notes")

# ---- one representation, three families ------------------------------------
node("rep", 41, 26, "a representation\n\nspectral\ncodec\nlearned encoder",
     fc=PANEL)

# ---- key, from clips -------------------------------------------------------
node("k1", 78, 49, "macro F1", ec=BLUE)
node("k2", 78, 41, r"$\rho^{\mathrm{cent}}_{5}$  centroids", ec=BLUE)
node("k3", 78, 33, r"$\rho^{W}_{5}$  readout", ec=BLUE)
pk = panel(["k1", "k2", "k3"])
ax.text(pk[0] + 1.5, pk[3] - 1.0, "key", fontsize=FS, color=INK2,
        fontweight="bold", ha="left", va="top")

# ---- pitch, from notes -----------------------------------------------------
# side by side rather than stacked: the helix fit is a sibling of the octave
# contrast, not a further step in the same ladder, and a vertical stack would
# read as a sequence
node("p1", 70, 11, "octave contrast\nnative, geodesic\nPCA 4, key 4")
node("p2", 92, 11, "helix\nfit", ls=(0, (2.4, 1.6)))
pp = panel(["p1", "p2"])
ax.text(pp[0] + 1.5, pp[3] - 1.0, "pitch", fontsize=FS, color=INK2,
        fontweight="bold", ha="left", va="top")

# both corpora go through the representation; nothing reaches a measurement
# without passing through it
arrow(side("gs", "r"), (B["rep"][0], 34))
arrow(side("ns", "r"), (B["rep"][0], 19))
arrow((B["rep"][2], 34), (pk[0], 41))
arrow((B["rep"][2], 19), (pp[0], 11))
# the only link between the two halves
arrow(side("k3", "b"), side("p1", "t"), color=ORANGE, lw=1.0, rad=0.0)
ax.text(83, 25, "rank-4 LDA on\nclip key labels", fontsize=FS - 0.3,
        color=ORANGE, ha="left", va="center", linespacing=1.35, zorder=6)

lo = min(b[0] for b in B.values()), min(b[1] for b in B.values())
hi = max(b[2] for b in B.values()), max(b[3] for b in B.values())
print(f"box extent x [{lo[0]:.1f}, {hi[0]:.1f}]  y [{lo[1]:.1f}, {hi[1]:.1f}]")

out = FIG / "fig3_design.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=340)
print("wrote", out)
