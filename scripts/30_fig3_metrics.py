"""Figure 3: the same recordings, measured four ways, and what a helix is not.

Panel (a) is the claim of Section 4.3 in one picture. Each line is one
representation and each position on the x axis is a different metric applied to
the identical NSynth notes and the identical anchors, so the vertical spread
within a line is entirely the metric. Nothing is near the octave under the
native metric except the chromagram and the pitch-quantized STFT; the geodesic
metric moves things a little; a rank-4 subspace of the leading principal
components moves several arms below zero; and a rank-4 subspace fitted on
GiantSteps key labels, which never saw a note, lifts every arm but PupuJEPA.

Panel (b) is the dissociation that panel (a) cannot show. Fitting Yagi et al.'s
parametric pitch helix to the leading components of each instrument's notes
scores how helical the pitch geometry is. A helix has a height axis, so it
separates octaves rather than identifying them, and the two scores need not
agree. They do not: the ordering runs the wrong way, with MERT at the top left
and the chromagram, which folds octaves and therefore traces a closed cycle
rather than a helix, at the bottom right.

Helicality is an inverse mean squared error and so depends on the scale of the
embedding, which is not comparable across representations of different width.
The panel plots the scale-free reading of the same fit, the fraction of
pitch-centroid variance the helix explains.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader

s18 = SourceFileLoader("s18", "scripts/18_figures.py")
# script 18 draws on import, so lift only what is needed rather than run it
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, BROWN = "#8b5fd6", "#9c6b1f"
CRIMSON, PINK, TEAL, OLIVE = "#c0392b", "#d64a9c", "#0e7c7b", "#8a8f1d"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
SERIES = {
    "mert_L12":    ("MERT L12",  BLUE,    "-",  "o"),
    "muq_L6":      ("MuQ L6",    PURPLE,  "-",  "D"),
    "matpac_L6":   ("MATPAC L6", CRIMSON, "-",  "P"),
    "pupujepa":    ("PupuJEPA",  PINK,    "-",  "X"),
    "chroma":      ("chroma",    ORANGE,  "--", "s"),
    "cqt":         ("log-CQT",   AQUA,    ":",  "^"),
    "pq_stft":     ("PQ-STFT",   OLIVE,   ":",  "*"),
    "encodec_32k": ("EnCodec",   BROWN,   "-.", "<"),
}
K_GEO = "10"
FIG = Path("runs/figs"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def load(path, *keys):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def native(arm):
    return load(f"runs/transpose_{arm}.json", "stats", "delta_strict", "value")


def geodesic(arm):
    return load(f"runs/geodesic_{arm}.json", "geodesic", K_GEO, "d_strict", "v")


def pca4(arm):
    return load("runs/pca_helix.json", arm, "pca", "4", "nsynth_curve",
                "d_strict", "v")


def key4(arm):
    return load(f"runs/subspace_{arm}.json", "subspaces", "key_lda", "4",
                "nsynth_curve", "d_strict", "v")


def helix_r2(arm):
    return load("runs/helicality.json", "arms", arm, "r2_mean")


METRICS = [("native", native), ("geodesic", geodesic),
           ("PCA 4", pca4), ("key 4", key4)]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.42, 1.72))
xs = np.arange(len(METRICS))
missing = []
for arm, (label, col, ls, mk) in SERIES.items():
    ys = [f(arm) for _, f in METRICS]
    if any(v is None for v in ys):
        missing += [f"{arm}/{METRICS[i][0]}"
                    for i, v in enumerate(ys) if v is None]
    y = np.array([np.nan if v is None else v for v in ys], dtype=float)
    axa.plot(xs, y, ls, color=col, marker=mk, markersize=3, linewidth=1.0,
             label=label, clip_on=False)
axa.axhline(0, color=INK2, linewidth=0.5, zorder=0)
axa.set_xticks(xs)
axa.set_xticklabels([m for m, _ in METRICS], rotation=20, ha="right")
axa.set_ylabel(r"$\Delta_{\mathrm{strict}}$")
axa.set_xlim(-0.25, len(METRICS) - 0.75)
axa.set_title("(a) four metrics, same notes", pad=3)

for arm, (label, col, ls, mk) in SERIES.items():
    x, y = helix_r2(arm), native(arm)
    if x is None or y is None:
        continue
    # no point labels: panel (a) already carries a legend over the same arms
    # with the same colour and marker, and five of the eight sit close enough
    # to zero that any label would overlap its neighbours
    axb.scatter([x], [y], s=20, color=col, marker=mk, zorder=3,
                edgecolors="none")
axb.axhline(0, color=INK2, linewidth=0.5, zorder=0)
axb.set_xlabel("helix fit  ($R^2$)")
axb.set_ylabel(r"$\Delta_{\mathrm{strict}}$, native")
axb.set_xlim(0.52, 0.90)
axb.set_title("(b) helix fit against contrast", pad=3)

for ax in (axa, axb):
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axa.legend(loc="upper left", frameon=False, handlelength=1.6,
           labelspacing=0.25, borderpad=0.1, ncol=1)
fig.tight_layout(pad=0.25, w_pad=1.0)
out = FIG / "fig3_metrics.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=300)
print("wrote", out)
if missing:
    print("missing series points:", ", ".join(missing))
