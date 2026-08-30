"""Table 1: decodability, ambient geometry, and readout geometry in one place.

Three quantities per representation, all on the same 7035 GiantSteps clips and
the same 24 keys:

  acc        24-way key accuracy from the pooled clip embedding (chance .042) --
             is the information there at all;
  rho_cent   circle-of-fifths correlation among the 24 key centroids -- is it in
             the ambient metric;
  rho_W      the same correlation among the fitted probe's 24 class weight
             vectors -- is it linearly accessible.

The three come apart, which is the point of the table, so the control rows are
what make it readable and they are grouped first. The synthetic controls have no
centroid geometry stored (script 11 only fits a probe on them), so it is
computed here rather than left blank.
"""
import json
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader

s11 = SourceFileLoader("s11", "scripts/11_probe_weight_from_cache.py").load_module()
s13 = SourceFileLoader("s13", "scripts/13_tonal_subspace.py").load_module()

# --- centroid geometry for the two synthetic probe-weight controls ----------
gz = np.load("runs/clipkey_cqt.npz")
Y, present = gz["Y"], gz["keys"]
for kind in ("random_feat", "orthocode"):
    p = Path(f"runs/clipkey_{kind}.json")
    d = json.loads(p.read_text())
    if "centroid_geometry" in d["poolings"]["mean"]:
        continue
    Z = s11.synth(kind, Y, present, np.random.default_rng(0))
    g = s13.key_geometry(Z, Y, np.random.default_rng(0), n_perm=2000)
    d["poolings"]["mean"]["centroid_geometry"] = {
        "kk_key_distance": {"rho": g["kk"]["rho"], "z": g["kk"]["z"]},
        "fifths": {"rho": g["fifths"]["rho"], "z": g["fifths"]["z"],
                   "p": g["fifths"]["p"], "rho_given_spec": float("nan")},
        "chromatic": {"rho": g["chromatic"]["rho"]},
        "mode_mismatch": {"rho": float("nan")}}
    p.write_text(json.dumps(d, indent=2))
    print(f"{kind}: centroid fifths {g['fifths']['rho']:+.3f} "
          f"kk {g['kk']['rho']:+.3f}", flush=True)

# --- assemble ---------------------------------------------------------------
BLOCKS = [
    ("Controls", [
        ("fifths_analytic", "analytic fifths"),
        ("random_feat",     "random features"),
        ("orthocode",       "orthogonal key codes"),
    ]),
    # the fold x norm 2x2 (cqt_norm, cqt_fold) is registered in PREREG_SCALES.md
    # and printed in runs/TABLE1.md; it is left out here because no sentence in
    # the four pages leans on it
    ("Spectral front ends", [
        ("cqt",      "log-CQT, 12 bin/oct"),
        ("hcqt",     "HCQT, 60 bin/oct"),
        ("pq_stft",  "PQ-STFT, 60 bin/oct"),
        ("chroma",   "chromagram"),
    ]),
    # DAC is in runs/TABLE1.md; at a macro F1 of .096 its geometry columns are
    # not interpretable, and EnCodec already carries the codec case
    ("Neural codec", [("encodec_32k", "EnCodec 32\\,kHz")]),
    # the layers the text cites, at matched fractions of depth; the full sweeps
    # are in runs/TABLE1.md
    ("MERT-v1-330M", [(f"mert_L{l}", f"layer {l}") for l in (4, 12, 24)]),
    ("MuQ-large", [(f"muq_L{l}", f"layer {l}") for l in (2, 12)]),
    ("MATPAC++ music", [(f"matpac_L{l}", f"layer {l}") for l in (6, 12)]),
    ("PupuJEPA", [("pupujepa", "teacher encoder")]),
]


def cell(v, w=6, d=3, star=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--".rjust(w)
    s = f"{v:+.{d}f}"
    return (s + ("*" if star else " ")).rjust(w)


# Highest and lowest value in each column, among the real representations only:
# the control block is excluded because an analytic circle of fifths would win
# every geometry column and orthogonal codes would win accuracy by construction.
# rho_chr is left unmarked, since it is a discriminant where a high value is a
# warning rather than a result.
MARKED_COLS = ("f1", "r5", "rw")

# Accuracy and macro F1 come from script 23, which runs the same probe under
# StratifiedGroupKFold on track identity. The clips are chunks of 1763 tracks,
# and a shuffled fold reads the track back rather than the key: it adds 0.28 to
# MERT and 0.01 to the chromagram, which reverses their order.
_PF1 = json.loads(Path("runs/probe_f1.json").read_text()) \
    if Path("runs/probe_f1.json").exists() else {}

rows = []
for block, arms in BLOCKS:
    rows.append((block, None))
    for arm, label in arms:
        f = Path(f"runs/clipkey_{arm}.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text())["poolings"]["mean"]
        c = d.get("centroid_geometry", {})
        w = d.get("probe_weight_geometry", {})
        pr = d.get("probe", {})
        g = _PF1.get(arm, {}).get("grouped", {})
        rows.append((label, dict(
            arm=arm,
            acc=g.get("acc", pr.get("key24_acc")),
            f1=g.get("macro_f1"),
            r5=c.get("fifths", {}).get("rho"),
            r5s=c.get("fifths", {}).get("rho_given_spec"),
            rw=w.get("fifths", {}).get("rho"),
            zw=w.get("fifths", {}).get("z"),
            rc=c.get("chromatic", {}).get("rho"),
            rm=c.get("mode_mismatch", {}).get("rho"))))

real = [(l, r) for l, r in rows
        if r and not l.startswith(("analytic", "random f", "orthogonal"))]
marks = {}
for c in MARKED_COLS:
    vals = [(r[c], r["arm"]) for l, r in real if r.get(c) is not None]
    if not vals:
        continue
    marks[(max(vals)[1], c)] = "hi"
    marks[(min(vals)[1], c)] = "lo"

hdr = f"{'representation':<38}{'acc':>7}{'r5_cent':>9}{'r5|spec':>9}{'r5_W':>8}{'z_W':>7}{'r_chrom':>9}{'r_mode':>8}"
print("\n" + hdr); print("-" * len(hdr))
for label, r in rows:
    if r is None:
        print(f"[{label}]"); continue
    acc = f"{r['acc']:.3f}" if r["acc"] else "--"
    zw = f"{r['zw']:+.1f}" if r["zw"] is not None else "--"
    print(f"{label:<38}{acc:>7}{cell(r['r5'],9)}{cell(r['r5s'],9)}"
          f"{cell(r['rw'],8)}{zw:>7}{cell(r['rc'],9)}{cell(r['rm'],8)}")

Path("runs/table1_rows.json").write_text(json.dumps(
    [{"label": l, **(r or {})} for l, r in rows], indent=2))


# the four cells that carry the mirror-image pair discussed in Sec. 4.1:
# a front end with readout geometry and no ambient geometry, and a learned
# representation with ambient geometry and no readout geometry


def tex(v, d=3, mark=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"${v:+.{d}f}$"
    return rf"\textcolor{{{mark}}}{{$\mathbf{{{s[1:-1]}}}$}}" if mark else s


# A single-column table. As a full-width table* every row costs two column
# lines, and with four blocks that was about a third of the four pages. The
# columns dropped here (the spectral partial, z_W and the mode discriminant)
# are in runs/TABLE1.md and summarised in the caption.
L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{3pt}",
     r"\caption{Key information, ambient geometry and readout geometry come "
     r"apart, on the same 7035 GiantSteps clips and 24 keys. F1 is "
     r"track-grouped macro F1 over the 24 imbalanced keys; accuracy is "
     r"reported in the repository. $\rho^{\text{cent}}_{5}$ is the "
     r"circle-of-fifths correlation among the 24 key centroids and "
     r"$\rho^{W}_{5}$ the same among the probe's class weight vectors, while "
     r"$\rho_{\text{chr}}$ measures correlation with semitone distance and "
     r"controls for proximity in chromatic pitch space. Orthogonal key codes "
     r"reach a macro F1 of $.613$ with classes equidistant by construction "
     r"and still show nothing, showing that fifths structure in "
     r"$\rho^{W}_{5}$ is not a consequence of classifier accuracy alone. In "
     r"each of the first three columns blue marks the highest value among "
     r"the real representations and red the lowest.}",
     r"\label{tab:decodable-vs-geometric}",
     r"\begin{tabular}{l r r r r}", r"\toprule",
     r"representation & F1 & $\rho^{\text{cent}}_{5}$ & $\rho^{W}_{5}$ & "
     r"$\rho_{\text{chr}}$ \\", r"\midrule"]
for label, r in rows:
    if r is None:
        L.append(r"\multicolumn{5}{l}{\textit{" + label + r"}} \\")
        continue
    mk = lambda k: marks.get((r["arm"], k))
    f1 = "--" if r.get("f1") is None else f"${r['f1']:.3f}$"
    if r.get("f1") is not None and mk("f1"):
        f1 = rf"\textcolor{{{mk('f1')}}}{{$\mathbf{{{r['f1']:.3f}}}$}}"
    L.append(f"\\quad {label} & {f1} & {tex(r['r5'], mark=mk('r5'))} & "
             f"{tex(r['rw'], mark=mk('rw'))} & {tex(r['rc'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
Path("runs/table1.tex").write_text("\n".join(L) + "\n")
print("wrote runs/table1_rows.json and runs/table1.tex")
