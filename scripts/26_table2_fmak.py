"""Table 2: the same measurements on FMAK, and the projection refitted there.

GiantSteps is entirely electronic dance music, where key correlates with
subgenre, which is the confound the paper otherwise only concedes. FMAK is
5476 usable tracks with expert key and mode annotation spread over 17 genres,
one excerpt per track, and fma_large stores the same 30 s at 44.1 kHz that the
GiantSteps clips use, so nothing about the measurement changes except the
corpus.

Two things are worth reading off this table rather than the first one.

The chromatic discriminant is negative for every arm here and was positive for
the three highest-scoring spectral arms on GiantSteps. An EDM corpus was
contributing semitone proximity that a genre-diverse one does not.

The projection column is refitted on FMAK keys and applied unchanged to the
same NSynth recordings, so the cross-dataset transfer no longer starts from a
single-genre corpus.
"""
import json
from pathlib import Path

ARMS = [
    # short labels: the block headers already carry the family, and Table 1
    # gives the resolutions
    # the citation rides with the row: this is the only table that names every
    # representation on its own line, so it is where a reader looks them up.
    # PQ-STFT is ours and has nothing to point at.
    ("Spectral representations", [("cqt", r"log-CQT~\cite{brown1991}"),
                             ("hcqt", r"HCQT~\cite{hcqt}"),
                             ("pq_stft", "PQ-STFT"),
                             ("chroma", r"chromagram~\cite{librosa}")]),
    ("Neural codec", [("encodec_32k", r"EnCodec~\cite{encodec}")]),
    ("Learned encoders", [("mert_L12", r"MERT L12~\cite{mert}"),
                          ("muq_L6", r"MuQ L6~\cite{muq}"),
                          ("matpac_L6", r"MATPAC L6~\cite{matpac}"),
                          ("pupujepa", r"PupuJEPA~\cite{pupujepa}")]),
]

f1 = json.loads(Path("runs/probe_f1_fmak.json").read_text()) \
    if Path("runs/probe_f1_fmak.json").exists() else {}


def geom(arm):
    p = Path(f"runs/clipkey_{arm}_fmak.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())["poolings"]["mean"]
    c, w = d["centroid_geometry"], d.get("probe_weight_geometry", {})
    g = lambda b, k: (b.get(k, {}) or {}).get("rho")
    return dict(f1=f1.get(arm, {}).get("grouped", {}).get("macro_f1"),
                r5=g(c, "fifths"), rw=g(w, "fifths"), rc=g(c, "chromatic"))


def native(arm):
    """Octave contrast on NSynth before any projection.

    This one does not depend on the corpus -- it is measured on NSynth in the
    representation's own cosine metric -- so it is the shared baseline the two
    projection columns are read against, and it is the same number in both
    tables.
    """
    p = Path(f"runs/transpose_{arm}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())["stats"]["delta_strict"]["value"]


def proj(arm, tag):
    p = Path(f"runs/subspace_{arm}{tag}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())["subspaces"]["key_lda"]["4"] \
        ["nsynth_curve"]["d_strict"]["v"]


# Same rule as Table 1: highest in a column blue, lowest red, both bold. The
# chromagram is excluded as the positive control, since it encodes octave
# equivalence by construction and would win several columns trivially. The
# discriminant and the two projection columns are left unmarked.
MARKED_COLS = ("r5", "rw", "rc", "nat", "pg", "pf")
CONTROL = {"chroma"}


def num(v, d=3, sign=False, mark=None):
    if v is None:
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if mark:
        return rf"\textcolor{{{mark}}}{{$\mathbf{{{s}}}$}}"
    return f"${s}$"

L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{1.5pt}",
     r"\caption{Centroid geometry, readout geometry and the chromatic "
     r"control, measured on FMAK rather than GiantSteps: 5476 tracks over "
     r"17 genres, one excerpt per track. The three $\Delta$ columns are the NSynth "
     r"octave contrast under the native metric and after a rank-4 projection "
     r"fitted on GiantSteps and on FMAK keys. $\Delta^{\mathrm{nat}}$ is "
     r"measured on NSynth alone and so does not depend on the corpus.}",
     r"\label{tab:fmak}",
     r"\resizebox{\columnwidth}{!}{%",
     r"\begin{tabular}{l r r r r r r}", r"\toprule",
     r"representation & $\rho^{\text{cent}}_{5}$ & $\rho^{W}_{5}$ & "
     r"$\rho_{\text{chr}}$ & $\Delta^{\mathrm{nat}}$ & "
     r"$\Delta^{\mathrm{GS}}$ & $\Delta^{\mathrm{FM}}$ \\",
     r"\midrule"]
# collect first so the column extremes are known before any row is written
data = {}
for _, arms in ARMS:
    for arm, _ in arms:
        g = geom(arm)
        if g is not None:
            g.update(nat=native(arm), pg=proj(arm, ""), pf=proj(arm, "_fmak"))
            data[arm] = g
marks = {}
for c in MARKED_COLS:
    vals = [(g[c], a) for a, g in data.items()
            if a not in CONTROL and g.get(c) is not None]
    if not vals:
        continue
    marks[(max(vals)[1], c)] = "hi"
    marks[(min(vals)[1], c)] = "lo"

rows = []
for block, arms in ARMS:
    L.append(r"\multicolumn{7}{l}{\textit{" + block + r"}} \\")
    for arm, label in arms:
        g = data.get(arm)
        if g is None:
            continue
        na, pg, pf = g["nat"], g["pg"], g["pf"]
        rows.append((arm, g, na, pg, pf))
        mk = lambda k: marks.get((arm, k))
        L.append(f"\\hspace{{3pt}}{label} & "
                 f"{num(g['r5'], sign=True, mark=mk('r5'))} & "
                 f"{num(g['rw'], sign=True, mark=mk('rw'))} & "
                 f"{num(g['rc'], sign=True, mark=mk('rc'))} & "
                 f"{num(na, sign=True, mark=mk('nat'))} & "
                 f"{num(pg, sign=True, mark=mk('pg'))} & "
                 f"{num(pf, sign=True, mark=mk('pf'))} \\\\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
Path("runs/table2_fmak.tex").write_text("\n".join(L) + "\n")

hdr = f"{'arm':<14}{'F1':>7}{'cent':>8}{'W':>8}{'chr':>8}{'native':>9}{'GSproj':>9}{'FMproj':>9}"
print(hdr); print("-" * len(hdr))
for arm, g, na, pg, pf in rows:
    f = lambda v, d=3: "  --" if v is None else f"{v:+.{d}f}"
    print(f"{arm:<14}{f(g['f1']):>7}{f(g['r5']):>8}{f(g['rw']):>8}"
          f"{f(g['rc']):>8}{f(na):>9}{f(pg):>9}{f(pf):>9}")
print("\nwrote runs/table2_fmak.tex")
