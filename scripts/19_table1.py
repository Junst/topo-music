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
        ("fifths_analytic", "analytic fifths (metric ctrl)"),
        ("random_feat",     "random features (no info)"),
        ("orthocode",       "orthogonal key codes (no tonal rel.)"),
    ]),
    (r"Spectral front end (fold $\times$ norm)", [
        ("cqt",      "log-CQT, 84 bin, dB"),
        ("cqt_norm", "log-CQT, 84 bin, per-frame norm"),
        ("cqt_fold", "octave-summed, 12 bin, dB"),
        ("chroma",   "chromagram (fold + norm)"),
    ]),
    ("Neural codec", [("encodec_32k", "EnCodec 32 kHz")]),
    ("MERT-v1-330M", [(f"mert_L{l}", f"layer {l}") for l in (4, 12, 16, 24)]),
]


def cell(v, w=6, d=3, star=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--".rjust(w)
    s = f"{v:+.{d}f}"
    return (s + ("*" if star else " ")).rjust(w)


rows = []
for block, arms in BLOCKS:
    rows.append((block, None))
    for arm, label in arms:
        f = Path(f"runs/clipkey_{arm}.json")
        if not f.exists():
            rows.append((label, None)); continue
        d = json.loads(f.read_text())["poolings"]["mean"]
        c = d.get("centroid_geometry", {})
        w = d.get("probe_weight_geometry", {})
        pr = d.get("probe", {})
        rows.append((label, dict(
            acc=pr.get("key24_acc", pr.get("key24_acc_refit")),
            r5=c.get("fifths", {}).get("rho"),
            r5s=c.get("fifths", {}).get("rho_given_spec"),
            rw=w.get("fifths", {}).get("rho"),
            zw=w.get("fifths", {}).get("z"),
            rc=c.get("chromatic", {}).get("rho"),
            rm=c.get("mode_mismatch", {}).get("rho"))))

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


def tex(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"${v:+.{d}f}$"


L = [r"\begin{table*}[t]", r"\centering", r"\small",
     r"\caption{Key information, ambient geometry and readout geometry come "
     r"apart. All columns are measured on the same 7035 GiantSteps clips and "
     r"the same 24 keys. \textit{acc}: 24-way key accuracy from the pooled "
     r"clip embedding (chance $.042$). $\rho^{\text{cent}}_{5}$: "
     r"circle-of-fifths correlation among the 24 key centroids, i.e.\ the "
     r"ambient metric; $\rho_{5}\!\mid\!\text{spec}$ partials out "
     r"mel-spectral centroid distance. $\rho^{W}_{5}$: the same correlation "
     r"among the fitted probe's 24 class weight vectors, i.e.\ what a linear "
     r"readout can reach. $\rho_{\text{chr}}$ (semitone adjacency) and "
     r"$\rho_{\text{mode}}$ (major/minor) are reported as discriminants. "
     r"The control block fixes the scale: an analytic circle of fifths scores "
     r"$\approx\!1$ everywhere; random features carry no information and "
     r"show nothing; orthogonal per-key codes decode key better than any real "
     r"representation ($.651$) with equidistant classes by construction, and "
     r"show nothing either -- so $\rho^{W}_{5}$ is not an artefact of an "
     r"accurate classifier on this label set.}",
     r"\label{tab:decodable-vs-geometric}",
     r"\begin{tabular}{l r r r r r r r}", r"\toprule",
     r"representation & acc & $\rho^{\text{cent}}_{5}$ & "
     r"$\rho_{5}\!\mid\!\text{spec}$ & $\rho^{W}_{5}$ & $z_W$ & "
     r"$\rho_{\text{chr}}$ & $\rho_{\text{mode}}$ \\", r"\midrule"]
for label, r in rows:
    if r is None:
        L.append(r"\multicolumn{8}{l}{\textit{" + label + r"}} \\")
        continue
    acc = f"${r['acc']:.3f}$" if r["acc"] else "--"
    zw = f"${r['zw']:+.1f}$" if r["zw"] is not None else "--"
    L.append(f"\\quad {label} & {acc} & {tex(r['r5'])} & {tex(r['r5s'])} & "
             f"{tex(r['rw'])} & {zw} & {tex(r['rc'])} & {tex(r['rm'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
Path("runs/table1.tex").write_text("\n".join(L) + "\n")
print("wrote runs/table1_rows.json and runs/table1.tex")
