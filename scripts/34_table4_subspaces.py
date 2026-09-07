"""Table 4: what each rank-4 subspace recovers, on the same NSynth notes.

Section 4.3 had grown into a paragraph that named six alternative subspaces and
their numbers in sequence, which is a table written as prose. The rows are what
the subspace was fitted on, the columns are representations, and every cell is
the same strict octave contrast at the same rank on the same held-out
instruments, so the only thing varying down a column is the fitting target.

The grouping carries the argument. Nothing is fitted at all in the first block.
The second block is fitted on the notes themselves, which is the natural thing
to try and is where an unsupervised projection sits. Only the third block is
fitted somewhere else entirely, on clip-level key labels from a music corpus,
and the permuted row underneath it is that same fit with the labels detached
from the audio.
"""
import json
from pathlib import Path

ARMS = [("mert_L12", "MERT L12"), ("muq_L6", "MuQ L6"),
        ("matpac_L6", "MATPAC L6"), ("encodec_32k", "EnCodec"),
        ("cqt", "log-CQT"), ("pupujepa", "PupuJEPA")]
NUL = json.loads(Path("runs/label_permutation_null.json").read_text())
PCA = json.loads(Path("runs/pca_helix.json").read_text())


def sub(arm, kind, d="4"):
    p = Path(f"runs/subspace_{arm}.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text())["subspaces"]
    if kind == "random":
        return s["random"][d]["d_strict"]["mean"]
    return s[kind][d]["nsynth_curve"]["d_strict"]["v"]


def native(arm):
    return json.loads(Path(f"runs/transpose_{arm}.json").read_text()) \
        ["stats"]["delta_strict"]["value"]


ROWS = [
    ("no projection", [
        ("random subspace", lambda a: sub(a, "random")),
    ]),
    ("fitted on the notes", [
        ("leading components",
         lambda a: PCA[a]["pca"]["4"]["nsynth_curve"]["d_strict"]["v"]
         if a in PCA else None),
        ("pitch height", lambda a: sub(a, "pitch_lda")),
        ("pitch class", lambda a: sub(a, "pc_lda")),
    ]),
    ("fitted on clip key labels", [
        ("permuted labels",
         lambda a: NUL[a]["null_mean"] if a in NUL else None),
        ("true labels", lambda a: sub(a, "key_lda")),
    ]),
]

L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{2pt}",
     r"\caption{The strict octave contrast on the same NSynth notes under "
     r"rank-4 subspaces that differ only in what they were fitted on: nothing, "
     r"the notes themselves, and GiantSteps clip key labels.}",
     r"\label{tab:subspaces}",
     r"\resizebox{\columnwidth}{!}{%",
     r"\begin{tabular}{l" + " r" * len(ARMS) + "}", r"\toprule",
     "subspace & " + " & ".join(n for _, n in ARMS) + r" \\", r"\midrule"]
# the three groups are separated by a rule rather than by a header row, which
# costs three lines the page does not have; the caption names them
for bi, (block, items) in enumerate(ROWS):
    if bi:
        L.append(r"\addlinespace[1.5pt]")
    for name, fn in items:
        cells = []
        for arm, _ in ARMS:
            v = fn(arm)
            cells.append("--" if v is None else f"${v:+.3f}$")
        L.append(rf"\hspace{{3pt}}{name} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
Path("runs/table4_subspaces.tex").write_text("\n".join(L) + "\n")
print("\n".join(L[10:]))
print("\nwrote runs/table4_subspaces.tex")
