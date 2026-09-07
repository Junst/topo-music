"""Helical organization does not imply octave equivalence.

The other measurements in the paper vary the metric and recompute one
statistic. This one does not: the helix fit asks a different question of the
same points, namely whether pitch traces a spiral, and a spiral has a height
axis that holds octaves apart. So the two need not agree, and the scatter shows
that on these representations they disagree in the strongest way available,
by ordering the arms in opposite directions.

The inset is the reference the fit is read against. Shuffling the pitch order
leaves the point cloud alone and destroys only its correspondence with pitch,
so the gap between the two distributions is what the helix score is detecting.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, BROWN = "#8b5fd6", "#9c6b1f"
CRIMSON, PINK, OLIVE = "#c0392b", "#d64a9c", "#8a8f1d"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
SERIES = {
    "mert_L12":    ("MERT L12",  BLUE,    "o"),
    "muq_L6":      ("MuQ L6",    PURPLE,  "D"),
    "matpac_L6":   ("MATPAC L6", CRIMSON, "P"),
    "pupujepa":    ("PupuJEPA",  PINK,    "X"),
    "chroma":      ("chromagram", ORANGE, "s"),
    "cqt":         ("log-CQT",   AQUA,    "^"),
    "pq_stft":     ("PQ-STFT",   OLIVE,   "*"),
    "encodec_32k": ("EnCodec",   BROWN,   "<"),
}
# Placed across both columns: the three best helix fits differ by 0.01, and at
# one column's width their markers and labels sat on top of each other. Where a
# label still cannot sit next to its point it gets a leader.
LABEL = {"chroma":      (10,  0, "left",   False),
         "pupujepa":    (0,  11, "center", False),
         "encodec_32k": (0,  11, "center", False),
         "pq_stft":     (0,  11, "center", False),
         "muq_L6":      (0,  11, "center", False),
         "cqt":         (-12, 22, "center", True),
         "matpac_L6":   (2,  35, "center", True),
         "mert_L12":    (16, 48, "center", True)}

FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})

H = json.loads(Path("runs/helicality.json").read_text())["arms"]


def native(arm):
    return json.loads(Path(f"runs/transpose_{arm}.json").read_text()) \
        ["stats"]["delta_strict"]["value"]


# Two panels rather than one with an inset: the reference distribution sat
# inside the plotting area and read as part of the data. Single column, so the
# scatter keeps the full width it needs to separate the three best fits.
fig, (ax, ins) = plt.subplots(
    2, 1, figsize=(3.35, 2.52),
    gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.62})

for arm, (label, col, mk) in SERIES.items():
    if arm not in H:
        continue
    x, y = H[arm]["r2_mean"], native(arm)
    ax.scatter([x], [y], s=30, color=col, marker=mk, zorder=4,
               edgecolors="none")
    dx, dy, ha, leader = LABEL[arm]
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=6, color=INK2, ha=ha, va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2,
                                shrinkA=1, shrinkB=3) if leader else None)

ax.axhline(0, color=INK2, lw=0.6, zorder=1)
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylabel(r"octave contrast $\Delta_{\mathrm{strict}}$")
ax.set_xlim(0.562, 0.836)
ax.set_ylim(-0.10, 0.80)
# the axis descriptions live in the panel titles rather than under the axes
ax.set_title("(a) helix fit, pitch-centroid variance explained", loc="center",
             fontsize=7, color=INK, pad=3)

real = np.concatenate([[r["r2"] for r in H[a]["per_instrument"]]
                       for a in SERIES if a in H])
shuf = np.concatenate([[r["r2_shuffled"] for r in H[a]["per_instrument"]]
                       for a in SERIES if a in H])
bins = np.linspace(0, 1, 40)
ins.hist(shuf, bins=bins, color=INK2, alpha=0.55, label="pitch shuffled")
ins.hist(real, bins=bins, color=BLUE, alpha=0.8, label="real")
ins.set_yticks([])
ins.set_xlim(0, 1)
ins.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ins.set_xticklabels(["0", "", ".5", "", "1"])
ins.tick_params(length=2, pad=1.5)
for sp in ("top", "right", "left"):
    ins.spines[sp].set_visible(False)
ins.set_title("(b) the same fit, per instrument", loc="center", fontsize=7,
              color=INK, pad=3)
# inside the panel and centred: the two humps leave the middle empty, and above
# the axes the legend would run into the panel label
ins.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=2,
           handlelength=0.9, handletextpad=0.4, borderpad=0.0,
           columnspacing=0.9, bbox_to_anchor=(0.52, 1.02))

out = FIG / "fig4_helix.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=340)
print("wrote", out)
print("pitch order real  R2: mean %.3f  n=%d" % (real.mean(), len(real)))
print("pitch order shuffled: mean %.3f" % shuf.mean())
