"""Figure 2 for the paper: the octave peak the strict contrast summarises.

Delta_strict is a difference of means, so it says that transposition by an
octave is more similarity-preserving than transposition by eleven or thirteen
semitones, but it says nothing about the shape of the curve it is drawn from.
This plots the curve. In the ambient metric mean similarity falls smoothly with
transposition distance and nothing marks the octave. Under the rank-4
projection fitted only on GiantSteps track tonics the same notes show sharp
maxima at exactly twelve and twenty-four semitones, and the subsidiary maxima
at five and seven semitones are the fourth and the fifth, which a key-supervised
projection has every reason to carry.

Both curves are z-scored per arm, because the projection changes the scale of
the similarities and the claim is about where the maxima sit rather than how
large they are.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader

s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()
ARM = "mert_L12"
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

gz = np.load(f"runs/clipkey_{ARM}.npz")
Zg = gz["Z"].astype(np.float64); Zg = Zg[:, :Zg.shape[1] // 2]
Yg = gz["Y"]
lab, gsplit = s13.gs_key_labels()
assert (lab == Yg).all()
W = s13.lda_directions(Zg[gsplit < 2], Yg[gsplit < 2] // 2, 4)

nz = np.load(f"runs/nsynth_emb_{ARM}.npz", allow_pickle=True)
Zn = nz["Z"].astype(np.float64)
anch = nz["anchors"]; kmax = int(nz["kmax"])
base = {(i, a_): r for i, a_, k, r in anch if k == 0}
byk = {}
for i, a_, k, r in anch:
    byk.setdefault(k, []).append((base[(i, a_)], r))

unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
ks = np.arange(1, kmax + 1)

fig, ax = plt.subplots(figsize=(3.35, 2.05))
for Zx, col, ls, nm in [(unit(Zn), AQUA, "--", "ambient"),
                        (unit(Zn @ W), BLUE, "-", "rank-4 projection")]:
    S = np.array([np.mean([Zx[a] @ Zx[b] for a, b in byk[k]]) for k in ks])
    S = (S - S.mean()) / S.std()
    ax.plot(ks, S, color=col, ls=ls, lw=1.5, marker="o", ms=2.4, label=nm)
    print(nm, np.round(S, 3), flush=True)
for k in (12, 24):
    ax.axvline(k, color=INK2, lw=.7, ls=":", zorder=1)
ax.set_xlabel("transposition (semitones)")
ax.set_ylabel("mean similarity (z)")
ax.set_xticks([1, 5, 7, 12, 17, 19, 24])
ax.grid(True, color=GRID, lw=.5); ax.set_axisbelow(True)
ax.set_ylim(top=4.05)
ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.2,
          columnspacing=1.6, borderpad=0.1)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig2_octave_peak.png"); fig.savefig(FIG / "fig2_octave_peak.pdf")
print("fig2_octave_peak written", flush=True)
