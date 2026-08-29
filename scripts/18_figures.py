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
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
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
    # staggered, because MERT L12 and chroma land within 0.007 of each other
    dy = {"mert_L12": -6.5, "chroma": 6.5, "cqt": 0.0}[arm]
    axb.annotate(lab, (xs2[-1], ys2[-1]), xytext=(5, dy),
                 textcoords="offset points", color=col, fontsize=7,
                 va="center", fontweight="bold")

for ax, xl, ti in [(axa, "pooling window (s, log)", "(a) more audio per clip"),
                   (axb, "clips per key centroid (log)",
                    "(b) more clips per centroid")]:
    style(ax); ax.set_xscale("log"); ax.set_xlabel(xl)
    ax.set_title(ti, loc="left", color=INK)
    ax.axhline(0, color=INK2, lw=0.6, zorder=2)
axa.set_ylabel(r"$\rho_{\mathrm{fifths}}$ (key centroids)")
axa.set_xticks([0.02, 0.1, 0.5, 2, 8, 20])
axa.set_xticklabels(["0.02", "0.1", "0.5", "2", "8", "20"])
axb.set_xticks([1, 4, 16, 64, 293]); axb.set_xticklabels(["1", "4", "16", "64", "293"])
handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.20), handlelength=2.4)
axa.annotate(r"$1.7\times$", (20, 0.468), xytext=(-4, -18),
             textcoords="offset points", ha="right", color=BLUE, fontsize=7.5)
axb.annotate(r"$13.4\times$", (170, 0.16), color=BLUE, fontsize=7.5,
             ha="center")
axb.set_xlim(right=430)
fig.savefig(FIGDIR / "fig1_averaging.pdf"); fig.savefig(FIGDIR / "fig1_averaging.png")
print("fig1 written")

# ---------------------------------------------------------------- figure 2
ARMS2 = ["cqt", "mert_L4", "mert_L12", "mert_L16", "mert_L24"]
NICE = {"cqt": "log-CQT", "mert_L4": "L4", "mert_L12": "L12",
        "mert_L16": "L16", "mert_L24": "L24"}
cm = json.load(open("runs/nsynth_class_means.json"))
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1.15]})
for arm in ARMS2:
    c = cm[arm]["curve"]
    npp = [r["n_per_pitch"] if r["n_per_pitch"] else 128 for r in c]
    ys = [r["d_strict"]["mean"] for r in c]
    col = BLUE if arm.startswith("mert") else AQUA
    axa.plot(npp, ys, color=col, lw=1.3, marker="o", ms=2.6,
             alpha=0.55 if arm != "mert_L12" else 1.0,
             ls="-" if arm.startswith("mert") else ":", zorder=3)
axa.annotate("MERT L4-L24, log-CQT\n(all arms flat at zero;\nmax $+0.008$)",
             (2.5, 0.06), color=INK2, fontsize=7)
axa.annotate("chroma control: $+0.57 \\rightarrow +0.62$, off scale",
             (1.05, 0.44), color=ORANGE, fontsize=7)
style(axa); axa.set_xscale("log"); axa.axhline(0, color=INK2, lw=0.6, zorder=2)
axa.set_xticks([1, 4, 16, 64, 128]); axa.set_xticklabels(["1", "4", "16", "64", "all"])
axa.set_xlabel("notes per pitch centroid (log)")
axa.set_ylabel(r"$\Delta_{\mathrm{strict}}$ (octave equivalence)")
axa.set_title("(a) averaging does not help", loc="left", color=INK)

sub = json.load(open("runs/subspace_mert_L12.json"))
x = np.arange(len(ARMS2)); w = 0.36
amb = [json.load(open(f"runs/subspace_{a}.json"))["ambient"]["nsynth_curve"]["d_strict"]["v"] for a in ARMS2]
prj = [json.load(open(f"runs/subspace_{a}.json"))["subspaces"]["key_lda"]["4"]["nsynth_curve"] for a in ARMS2]
pv = [p["d_strict"]["v"] for p in prj]
perr = np.array([[p["d_strict"]["v"] - p["d_strict"]["lo"] for p in prj],
                 [p["d_strict"]["hi"] - p["d_strict"]["v"] for p in prj]])
axb.bar(x - w / 2, amb, w, color=AQUA, label="ambient", zorder=3)
axb.bar(x + w / 2, pv, w, color=BLUE, yerr=perr, capsize=2,
        error_kw=dict(elinewidth=0.7, ecolor=INK2), label="GS-key projection (d=4)",
        zorder=3)
style(axb); axb.axhline(0, color=INK2, lw=0.6, zorder=2)
axb.set_xticks(x); axb.set_xticklabels([NICE[a] for a in ARMS2])
axb.set_xlabel("representation")
axb.set_title("(b) a projection does", loc="left", color=INK)
axb.legend(frameon=False, loc="upper right", handlelength=1.4)
fig.savefig(FIGDIR / "fig2_mechanisms.pdf"); fig.savefig(FIGDIR / "fig2_mechanisms.png")
print("fig2 written")
