"""What each rank-4 subspace recovers, on radial axes.

One spoke per representation, one polygon per fitting target, and the shaded
ring is the null range spanned by a random subspace and by a fit on permuted
key labels. Every point is the strict octave contrast at rank 4 on the same
held-out NSynth instruments.

Two things a radar plot usually gets wrong are handled explicitly. The radial
origin sits below the smallest value rather than at zero, so the negative
contrasts of the leading components do not wrap through the centre, and the
zero circle is drawn darker so that inside it still reads as negative. The
polygons are left unfilled, since a filled area grows with the square of the
value and would overstate the differences the plot is about.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUL = json.loads(Path("runs/label_permutation_null.json").read_text())
PCA = json.loads(Path("runs/pca_helix.json").read_text())

BLUE, AQUA, OLIVE, CRIMSON = "#2a78d6", "#1baf7a", "#8a8f1d", "#c0392b"
INK, INK2, GRID, BAND = "#1a1a19", "#55554e", "#dededa", "#d8d8d1"
# learned encoders first, then the codec and the spectral arm
# layer numbers are in the caption, not on the spokes: they make the labels
# long enough to shrink the circle, which is the thing that has to be read
import sys
ARMS = [("mert_L12", "MERT"), ("muq_L6", "MuQ"),
        ("matpac_L6", "MATPAC"), ("pupujepa", "PupuJEPA"),
        ("encodec_32k", "EnCodec"), ("cqt", "log-CQT"),
        ("hcqt", "HCQT"), ("pq_stft", "PQ-STFT")]
# the chromagram folds octaves by construction, so every one of its subspaces
# scores near 0.9 and the radial axis has to stretch to hold it
if "--chroma" in sys.argv:
    ARMS = ARMS + [("chroma", "chromagram")]
ARMS = [a for a in ARMS if Path(f"runs/subspace_{a[0]}.json").exists()
        and a[0] in NUL and a[0] in PCA]
RMIN = -0.15
RMAX = 0.95 if "--chroma" in sys.argv else 0.66
SUFFIX = "_chroma" if "--chroma" in sys.argv else ""
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 6.6, "legend.fontsize": 6.0, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def sub(arm, kind, d="4"):
    s = json.loads(Path(f"runs/subspace_{arm}.json").read_text())["subspaces"]
    if kind == "random":
        return s["random"][d]["d_strict"]["mean"]
    return s[kind][d]["nsynth_curve"]["d_strict"]["v"]


SERIES = [("leading components", OLIVE, "^", (0, (2.2, 1.4))),
          ("pitch height", CRIMSON, "v", (0, (1.2, 1.2))),
          ("pitch class", AQUA, "D", (0, (4, 1.4))),
          ("key labels", BLUE, "o", "-")]
GET = {"leading components": lambda a: PCA[a]["pca"]["4"]["nsynth_curve"]["d_strict"]["v"],
       "pitch height": lambda a: sub(a, "pitch_lda"),
       "pitch class": lambda a: sub(a, "pc_lda"),
       "key labels": lambda a: sub(a, "key_lda")}

n = len(ARMS)
th = np.linspace(0, 2 * np.pi, n, endpoint=False)
close = lambda v: np.concatenate([v, v[:1]])
thc = close(th)

fig = plt.figure(figsize=(3.35, 3.05))
ax = fig.add_subplot(projection="polar")
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

lo = close(np.array([min(sub(a, "random"), NUL[a]["null_lo"]) for a, _ in ARMS]))
hi = close(np.array([max(sub(a, "random"), NUL[a]["null_hi"]) for a, _ in ARMS]))
ax.fill_between(thc, lo, hi, color=BAND, zorder=1, linewidth=0,
                label="null range")
ax.plot(thc, np.zeros_like(thc), color=INK2, lw=0.8, zorder=2)

for label, col, mk, ls in SERIES:
    v = close(np.array([GET[label](a) for a, _ in ARMS]))
    ax.plot(thc, v, color=col, linestyle=ls, lw=1.0, marker=mk,
            markersize=3.0, markeredgewidth=0.8,
            markerfacecolor=col if mk == "o" else "none",
            markeredgecolor=col, label=label, zorder=4)

ax.set_ylim(RMIN, RMAX)
# radial labels in the gap between two spokes rather than on one of them
ax.set_rgrids([0.0, 0.2, 0.4, 0.6] + ([0.8] if "--chroma" in sys.argv else []),
              labels=["0", ".2", ".4", ".6"] + ([".8"] if "--chroma" in sys.argv else []),
              angle=25.7, fontsize=5.8, color=INK2)
ax.set_xticks(th)
ax.set_xticklabels([nm for _, nm in ARMS], fontsize=6.8)
ax.tick_params(axis="x", pad=0.5)
ax.grid(color=GRID, lw=0.5)
ax.spines["polar"].set_edgecolor(GRID)
ax.spines["polar"].set_linewidth(0.6)
# the legend sits in the corners a circle leaves empty, so it costs the figure
# no height at all. Beside the plot it took a third of the width away from the
# circle, and below it added the same again in height.
h, lb = ax.get_legend_handles_labels()
fig.legend(h, lb, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0),
           ncol=1, fontsize=5.8, handlelength=1.2, handletextpad=0.3,
           columnspacing=0.8, labelspacing=0.28, borderpad=0.0)

out = FIG / f"fig5_radar{SUFFIX}.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=340)
print("wrote", out)
