"""The transposition curve on its own.

Panel (b) of the two-panel mechanisms figure compared four metrics as bars.
The subspace figure now carries that comparison with more of it visible, so
this keeps only the curve, which is the part a bar chart cannot show: where
the maxima sit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
FIG = Path("runs/figs")
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})

oc = np.load(FIG / "_octave_curve.npz")
fig, ax = plt.subplots(figsize=(3.35, 1.22))
ax.plot(oc["ks"], oc["ambient"], color=AQUA, ls=":", lw=1.2, label="native")
ax.plot(oc["ks"], oc["projected"], color=BLUE, lw=1.2, label="rank-4 key")
for k in (12, 24):
    ax.axvline(k, color=INK2, lw=0.6, ls=":", zorder=1)
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xticks([1, 5, 7, 12, 17, 19, 24])
ax.set_yticks([])
ax.set_xlabel("transposition (semitones)")
ax.set_ylabel("mean similarity (z)")
ax.legend(frameon=False, loc="upper right", fontsize=6.4, handlelength=1.4,
          handletextpad=0.4, borderpad=0.1)
out = FIG / "fig2_curve.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=340)
print("wrote", out)
