"""ICASSP figures 1 and 2, built from the run JSONs.

Figure 1 contrasts the two averaging axes on the same y scale: stretching the
window vs adding examples to each key centroid. Figure 2 contrasts the two
mechanisms: averaging cannot expose octave equivalence on NSynth, a projection
can. Both panels of figure 2 share a y axis on purpose -- the left panel is
flat at zero at the scale where the right panel's effect is obvious, and
rescaling it would invent structure that is not there.

Palette: slots 1-3 of the reference categorical theme (blue / orange / aqua),
the documented all-pairs-safe subset. Line style and marker carry identity as
well as hue, so the figures survive grayscale printing and CVD.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, BROWN = "#8b5fd6", "#9c6b1f"
CRIMSON, PINK, TEAL, OLIVE = "#c0392b", "#d64a9c", "#0e7c7b", "#8a8f1d"
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
# Line style groups the families and hue separates within them: solid for the
# four self-supervised encoders, dashed for the chroma control, dotted for the
# three log-frequency front ends, dash-dot for the codec. MuQ L6 and MATPAC L6
# sit at the same fraction of depth as MERT L12.
SERIES = {                       # arm -> (label, colour, linestyle, marker)
    "mert_L12":    ("MERT L12",  BLUE,    "-",  "o"),
    "muq_L6":      ("MuQ L6",    PURPLE,  "-",  "D"),
    "matpac_L6":   ("MATPAC L6", CRIMSON, "-",  "P"),
    "pupujepa":    ("PupuJEPA",  PINK,    "-",  "X"),
    "chroma":      ("chroma",    ORANGE,  "--", "s"),
    "cqt":         ("log-CQT",   AQUA,    ":",  "^"),
    "hcqt":        ("HCQT",      TEAL,    ":",  "v"),
    "pq_stft":     ("PQ-STFT",   OLIVE,   ":",  "*"),
    "encodec_32k": ("EnCodec",   BROWN,   "-.", "<"),
}
FIGDIR = Path("runs/figs"); FIGDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- figure 1
# two panels only: the centroid distance matrices that used to sit in (c) show
# the same rise as (b) and the section reads without them
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.7, 1.88), sharey=True,
                               gridspec_kw={"wspace": 0.10})
for arm, (lab, col, ls, mk) in SERIES.items():
    d = json.load(open(f"runs/accum_{arm}.json"))
    full = d["curve"][-1]["fifths"]["mean"]
    xs = [c["seconds"] if c["seconds"] else 20.0 for c in d["curve"]]
    ys = [c["fifths"]["mean"] for c in d["curve"]]
    es = [c["fifths"]["std"] for c in d["curve"]]
    axa.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker=mk, ms=3.2, lw=1.6,
                 capsize=1.5, elinewidth=0.7, label=lab, zorder=3)

    e = json.load(open("runs/examples_per_centroid.json"))[arm]
    npk = e["median_clips_per_key"]
    xs2 = [c["n_per_key"] if c["n_per_key"] else npk for c in e["curve"]]
    ys2 = [c["fifths_mean"] for c in e["curve"]]
    es2 = [c["fifths_std"] for c in e["curve"]]
    axb.errorbar(xs2, ys2, yerr=es2, color=col, ls=ls, marker=mk, ms=3.2,
                 lw=1.6, capsize=1.5, elinewidth=0.7, zorder=3)

for ax, xl, ti in [(axa, "pooling window (s)", "(a) averaging over time"),
                   (axb, "clips per key centroid",
                    "(b) averaging over examples")]:
    style(ax); ax.set_xscale("log"); ax.set_xlabel(xl)
    ax.set_title(ti, loc="center", color=INK)
    ax.axhline(0, color=INK2, lw=0.6, zorder=2)
axa.set_ylabel(r"Key centroids ($\rho_{\mathrm{fifths}}$)")
axa.set_xticks([0.02, 0.1, 0.5, 2, 8, 20])
axa.set_xticklabels(["0.02", "0.1", "0.5", "2", "8", "20"])
axb.set_xticks([1, 4, 16, 64, 293]); axb.set_xticklabels(["1", "4", "16", "64", "293"])
handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.33), handlelength=2.0, fontsize=6.8,
           columnspacing=1.3)
# the 1.7x and 13.4x labels are in the text; with five series the panels are
# busy enough without them
axb.set_xlim(right=380)
plt.setp(axb.get_yticklabels(), visible=False)

fig.savefig(FIGDIR / "fig1_averaging.pdf"); fig.savefig(FIGDIR / "fig1_averaging.png")
print("fig1 written")

# ---------------------------------------------------------------- figure 2
# Two panels, both contrasting the ambient metric with the same rank-4
# projection, so one legend serves both. The averaging null that used to be a
# third panel is a flat line at zero and the text states its numbers exactly,
# so it costs a panel and shows nothing a sentence does not.
ARMS2 = ["cqt", "mert_L4", "mert_L12", "mert_L16", "mert_L24"]
NICE = {"cqt": "log-CQT", "mert_L4": "L4", "mert_L12": "L12",
        "mert_L16": "L16", "mert_L24": "L24"}

# stacked rather than side by side: each panel then gets the full column
# width, which the 25-point transposition curve in (a) needs
fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.35, 2.85),
                               gridspec_kw={"hspace": 0.78})

# (a) the curve the strict contrast is a summary of
oc = np.load("runs/figs/_octave_curve.npz")
axa.plot(oc["ks"], oc["ambient"], color=AQUA, ls=":", lw=1.2)
axa.plot(oc["ks"], oc["projected"], color=BLUE, lw=1.2)
for k in (12, 24):
    axa.axvline(k, color=INK2, lw=0.6, ls=":", zorder=1)
style(axa)
axa.set_xticks([1, 5, 7, 12, 17, 19, 24]); axa.set_yticks([])
axa.set_xlabel("transposition (semitones)", fontsize=7)
axa.set_ylabel("mean similarity (z)", fontsize=7.5)
axa.set_title("(a) MERT L12 notes", loc="center", color=INK, fontsize=7.5)

# (b) the same statistic under three metrics: the native cosine one, shortest
# paths on a kNN graph over the same embeddings, and the rank-4 projection
# fitted on GiantSteps tonics. The matched random subspace is a tick rather
# than a fourth bar, since it only matters where it sits relative to ambient.
# the four patterns the section describes, one group each where possible:
# MERT L4/L12 native-null but geodesic-positive, MERT L24 null under both,
# MuQ a rising ladder, PupuJEPA null under all three
BARS = [("mert_L4", "MERT L4"), ("mert_L12", "L12"), ("mert_L24", "L24"),
        ("muq_L2", "MuQ L2"), ("muq_L12", "L12"), ("pupujepa", "PupuJEPA")]
amb, geo, prj, perr, rnd = [], [], [], [[], []], []
for a, _ in BARS:
    d = json.load(open(f"runs/subspace_{a}.json"))
    amb.append(d["ambient"]["nsynth_curve"]["d_strict"]["v"])
    rnd.append(d["subspaces"]["random"]["4"]["d_strict"]["mean"])
    v = d["subspaces"]["key_lda"]["4"]["nsynth_curve"]["d_strict"]
    prj.append(v["v"]); perr[0].append(v["v"] - v["lo"]); perr[1].append(v["hi"] - v["v"])
    geo.append(json.load(open(f"runs/geodesic_{a}.json"))["geodesic"]["10"]["d_strict"]["v"])
x = np.arange(len(BARS)); w = 0.27
axb.bar(x - w, amb, w, color=AQUA, label="native", zorder=3)
axb.bar(x, geo, w, color=PURPLE, label="geodesic", zorder=3)
axb.bar(x + w, prj, w, color=BLUE, yerr=np.array(perr), capsize=1.8,
        error_kw=dict(elinewidth=0.7, ecolor=INK2), label="GS-key $d=4$", zorder=3)
axb.scatter(x + w, rnd, marker="_", s=42, linewidths=1.1, color=INK2,
            label="random $d=4$", zorder=5)
style(axb); axb.axhline(0, color=INK2, lw=0.6, zorder=2)
# symlog: the native and geodesic values sit two orders of magnitude below the
# projected ones, and on a linear axis the contrast this panel is about would
# be invisible
axb.set_yscale("symlog", linthresh=0.01, linscale=0.45)
axb.set_yticks([0, 0.01, 0.1, 0.4])
axb.set_yticklabels(["0", ".01", ".1", ".4"])
axb.set_xticks(x); axb.set_xticklabels([n for _, n in BARS], fontsize=7)
axb.set_ylabel(r"$\Delta_{\mathrm{strict}}$", fontsize=7.5)
axb.set_title("(b) octave equivalence under three metrics", loc="center",
              color=INK, fontsize=7.5)

h, l = axb.get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.10), handlelength=1.5, fontsize=7,
           columnspacing=1.2, handletextpad=0.5)
fig.savefig(FIGDIR / "fig2_mechanisms.pdf"); fig.savefig(FIGDIR / "fig2_mechanisms.png")
print("fig2 written")
