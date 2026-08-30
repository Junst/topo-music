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
INK, INK2, GRID = "#1a1a19", "#55554e", "#dededa"
SERIES = {                       # arm -> (label, colour, linestyle, marker)
    "mert_L12": ("MERT L12", BLUE, "-", "o"),
    "chroma":   ("chroma",   ORANGE, "--", "s"),
    "cqt":      ("log-CQT",  AQUA, ":", "^"),
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
# (c) is a 2x2 block rather than a row of three, which leaves (a) and (b)
# wider in the same total width. Its first sub-row is an empty header whose
# top edge coincides with (a) and (b), so the three panel titles sit on one
# line even though the matrices below start lower.
fig = plt.figure(figsize=(7.0, 2.40))
gs = fig.add_gridspec(1, 3, width_ratios=[1.46, 1.46, 0.94], wspace=0.24)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharey=axa)
gsc = gs[2].subgridspec(3, 2, height_ratios=[0.001, 1, 1],
                        wspace=0.10, hspace=0.46)
axch = fig.add_subplot(gsc[0, :]); axch.axis("off")
axc = [fig.add_subplot(gsc[i, j]) for i in (1, 2) for j in (0, 1)]
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
    # no in-panel label here: the shared legend names all three arms, and a
    # label hanging off the right edge collided with panel (c)

for ax, xl, ti in [(axa, "pooling window (s)", "(a) more audio per clip"),
                   (axb, "clips per key centroid",
                    "(b) more clips per centroid")]:
    style(ax); ax.set_xscale("log"); ax.set_xlabel(xl)
    ax.set_title(ti, loc="center", color=INK)
    ax.axhline(0, color=INK2, lw=0.6, zorder=2)
axa.set_ylabel(r"Key centroids ($\rho_{\mathrm{fifths}}$)")
axa.set_xticks([0.02, 0.1, 0.5, 2, 8, 20])
axa.set_xticklabels(["0.02", "0.1", "0.5", "2", "8", "20"])
axb.set_xticks([1, 4, 16, 64, 293]); axb.set_xticklabels(["1", "4", "16", "64", "293"])
handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.155), handlelength=2.4)
axa.annotate(r"$1.7\times$", (20, 0.468), xytext=(-4, -18),
             textcoords="offset points", ha="right", color=BLUE, fontsize=7.5)
axb.annotate(r"$13.4\times$", (170, 0.16), color=BLUE, fontsize=7.5,
             ha="center")
axb.set_xlim(right=380)
plt.setp(axb.get_yticklabels(), visible=False)

# (c) the same rise as geometry. Ordered round the circle of fifths, majors
# then minors, and rank-transformed within each panel, which is what a Spearman
# correlation sees, so the three are comparable without a scale choice.
km = np.load(FIGDIR / "_key_matrices.npz")
PCN = ["C", "C$\\sharp$", "D", "D$\\sharp$", "E", "F",
       "F$\\sharp$", "G", "G$\\sharp$", "A", "A$\\sharp$", "B"]
ton = km["tonic_order"]
axch.set_title("(c) key-centroid distances", loc="center", color=INK)
for ax, key, ttl, sub in [
        (axc[0], "one", "1 audio clip", rf"$\rho={float(km['rho_one']):+.2f}$"),
        (axc[1], "eight", "8 audio clips", rf"$\rho={float(km['rho_eight']):+.2f}$"),
        (axc[2], "alle", "all audio clips", rf"$\rho={float(km['rho_all']):+.2f}$"),
        (axc[3], "ref", "reference", "circle of fifths")]:
    ax.imshow(km[key], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(ttl, loc="center", fontsize=6.2, color=INK, pad=2.2)
    ax.set_xlabel(sub, fontsize=6.2, color=INK2, labelpad=1.5)
    for v in (11.5,):                    # majors above and left, minors below
        ax.axhline(v, color="white", lw=0.6, alpha=0.55)
        ax.axvline(v, color="white", lw=0.6, alpha=0.55)
    tk = np.arange(0, 24, 6)
    ax.set_xticks([])
    ax.set_yticks(tk)
    ax.set_yticklabels([PCN[i] for i in ton[tk]] if ax in (axc[0], axc[2])
                       else [], fontsize=5.5)
    ax.tick_params(length=1.5, pad=1.0)

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
axa.plot(oc["ks"], oc["ambient"], color=AQUA, ls=":", lw=1.2, label="ambient")
axa.plot(oc["ks"], oc["projected"], color=BLUE, lw=1.2,
         label="GS-key projection (d=4)")
for k in (12, 24):
    axa.axvline(k, color=INK2, lw=0.6, ls=":", zorder=1)
style(axa)
axa.set_xticks([1, 5, 7, 12, 17, 19, 24]); axa.set_yticks([])
axa.set_xlabel("transposition (semitones)", fontsize=7)
axa.set_ylabel("mean similarity (z)", fontsize=7.5)
axa.set_title("(a) MERT L12 notes", loc="center", color=INK, fontsize=7.5)

# (b) the same contrast as one number per representation
prj = [json.load(open(f"runs/subspace_{a}.json"))["subspaces"]["key_lda"]["4"]["nsynth_curve"] for a in ARMS2]
amb = [json.load(open(f"runs/subspace_{a}.json"))["ambient"]["nsynth_curve"]["d_strict"]["v"] for a in ARMS2]
pv = [p["d_strict"]["v"] for p in prj]
perr = np.array([[p["d_strict"]["v"] - p["d_strict"]["lo"] for p in prj],
                 [p["d_strict"]["hi"] - p["d_strict"]["v"] for p in prj]])
x = np.arange(len(ARMS2)); w = 0.36
axb.bar(x - w / 2, amb, w, color=AQUA, zorder=3)
axb.bar(x + w / 2, pv, w, color=BLUE, yerr=perr, capsize=2,
        error_kw=dict(elinewidth=0.7, ecolor=INK2), zorder=3)
style(axb); axb.axhline(0, color=INK2, lw=0.6, zorder=2)
axb.set_xticks(x); axb.set_xticklabels([NICE[a] for a in ARMS2], fontsize=7)
axb.set_xlabel("representation", fontsize=7)
axb.set_ylabel(r"$\Delta_{\mathrm{strict}}$", fontsize=7.5)
axb.set_title("(b) octave equivalence", loc="center", color=INK, fontsize=7.5)

h, l = axa.get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.11), handlelength=2.0, fontsize=7)
fig.savefig(FIGDIR / "fig2_mechanisms.pdf"); fig.savefig(FIGDIR / "fig2_mechanisms.png")
print("fig2 written")
