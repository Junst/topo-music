"""What each rank-4 subspace recovers, as a grouped dot plot.

Same content as the table this replaces: one row per representation, one
position per subspace, and the only thing that varies along a row is what the
subspace was fitted on. Everything is the strict octave contrast at rank 4 on
the same held-out NSynth instruments.

The two nulls are drawn as a grey band rather than as two more dots. They are
the level a fit has to clear, not a value anyone reads off, and at this size
six markers on a row collide where the table's six columns did not.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OLIVE, CRIMSON = "#8a8f1d", "#c0392b"
INK, INK2, GRID, BAND = "#1a1a19", "#55554e", "#dededa", "#dcdcd5"
ARMS = [("mert_L12", "MERT L12"), ("muq_L6", "MuQ L6"),
        ("matpac_L6", "MATPAC L6"), ("encodec_32k", "EnCodec"),
        ("cqt", "log-CQT"), ("pupujepa", "PupuJEPA")]
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.edgecolor": INK2,
    "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

NUL = json.loads(Path("runs/label_permutation_null.json").read_text())
PCA = json.loads(Path("runs/pca_helix.json").read_text())


def sub(arm, kind, d="4"):
    s = json.loads(Path(f"runs/subspace_{arm}.json").read_text())["subspaces"]
    if kind == "random":
        return s["random"][d]["d_strict"]["mean"]
    return s[kind][d]["nsynth_curve"]["d_strict"]["v"]


SERIES = [                       # label, colour, marker, filled
    ("leading components", OLIVE,   "^", False),
    ("pitch height",       CRIMSON, "v", False),
    ("pitch class",        AQUA,    "D", False),
    ("key labels",         BLUE,    "o", True),
]
GET = {"leading components": lambda a: PCA[a]["pca"]["4"]["nsynth_curve"]["d_strict"]["v"],
       "pitch height": lambda a: sub(a, "pitch_lda"),
       "pitch class": lambda a: sub(a, "pc_lda"),
       "key labels": lambda a: sub(a, "key_lda")}
OFF = [0.21, 0.07, -0.07, -0.21]

fig, ax = plt.subplots(figsize=(3.35, 1.78))
for i, (arm, name) in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    lo = min(sub(arm, "random"), NUL[arm]["null_lo"])
    hi = max(sub(arm, "random"), NUL[arm]["null_hi"])
    ax.barh(y, hi - lo, left=lo, height=0.72, color=BAND, edgecolor="none",
            zorder=1, label="null range" if i == 0 else None)
    for (lab, col, mk, fill), dy in zip(SERIES, OFF):
        ax.scatter([GET[lab](arm)], [y + dy], s=14, marker=mk, zorder=3,
                   color=col if fill else "none",
                   edgecolors=col, linewidths=0.9,
                   label=lab if i == 0 else None)

ax.axvline(0, color=INK2, lw=0.6, zorder=2)
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([n for _, n in ARMS][::-1])
ax.set_ylim(-0.55, len(ARMS) + 0.15)
ax.set_xlim(-0.13, 0.53)
ax.set_xlabel(r"octave contrast $\Delta_{\mathrm{strict}}$ at rank 4")
ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
# inside the panel: the upper right is empty for the three encoders, and a
# legend below the axis costs a third of an inch the page does not have
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.8,
          handlelength=0.9, handletextpad=0.25, columnspacing=0.8,
          labelspacing=0.25, borderpad=0.1, bbox_to_anchor=(1.005, 1.03))

out = FIG / "fig5_subspaces.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=340)
print("wrote", out)
