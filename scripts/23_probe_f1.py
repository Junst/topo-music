"""Is 24-way accuracy the right summary for Table 1, and is the split right?

Two questions, one script.

1. GiantSteps keys run from 108 to 704 clips, a 6.5x imbalance, so accuracy is
   weighted towards the common keys. Macro F1 weights every key equally.

2. The clips are chunks: 7035 of them come from 1763 tracks, named
   <track>-<chunk>.wav, so roughly four clips share a track. A shuffled k-fold
   puts chunks of one track on both sides of the split and can read back the
   track rather than the key. Script 09 used sklearn's default cv=5, which does
   not shuffle and therefore keeps most chunks of a track together by accident.
   Neither is a guarantee, so this groups by track id explicitly and reports the
   shuffled numbers alongside, to show what the leak is worth.

Note on what is deliberately *not* used: the MIREX key score gives partial
credit for perfect-fifth, relative and parallel confusions. That is exactly the
relation the geometry columns measure, so scoring the probe that way would
blend the two things this paper separates.
"""
import json
import sys
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

m9 = SourceFileLoader("m9", "scripts/09_clip_key_geometry.py").load_module()

DATASET = "gs"
if sys.argv[1:2] == ["--fmak"]:
    DATASET, sys.argv = "fmak", sys.argv[:1] + sys.argv[2:]
TAG = "" if DATASET == "gs" else f"_{DATASET}"

ARMS = sys.argv[1:] or [
    "cqt", "cqt_norm", "cqt_fold", "pq_stft", "chroma", "encodec_32k",
    "mert_L4", "mert_L12", "mert_L16", "mert_L24",
    "muq_L2", "muq_L6", "muq_L8", "muq_L12"]


def track_groups():
    """Rebuild script 09's row order and return the track id of each clip."""
    g = []
    for sp in ["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"]:
        for line in (m9.GS / sp).read_text().splitlines():
            r = json.loads(line)
            km = m9.parse_key(r["label"])
            p = m9.GS / Path(r["audio_path"]).relative_to("data/GS")
            if km is not None and p.exists():
                g.append(p.stem.rsplit("-", 1)[0])
    return np.array(g)


# FMAK has one excerpt per track, so a stratified split is already
# track-disjoint and no grouping variable is needed
G = track_groups() if DATASET == "gs" else None
print(f"{DATASET}: " + (f"{len(G)} clips, {len(set(G))} tracks" if G is not None
                        else "one excerpt per track"), flush=True)

out = {}
print(f"{'arm':<12}{'acc_grp':>9}{'F1_grp':>8}{'acc_shuf':>10}{'F1_shuf':>9}", flush=True)
for a in ARMS:
    if a in ("random_feat", "orthocode"):
        # synthetic controls, rebuilt exactly as script 19 rebuilds them, so
        # the control block carries the same summary as every other row
        s11 = SourceFileLoader(
            "s11", "scripts/11_probe_weight_from_cache.py").load_module()
        gz = np.load("runs/clipkey_cqt.npz")
        Y = gz["Y"]
        Z = s11.synth(a, Y, gz["keys"], np.random.default_rng(0))
    else:
        f = Path(f"runs/clipkey_{a}{TAG}.npz")
        if not f.exists():
            continue
        d = np.load(f)
        Z = d["Z"].astype(np.float64)
        Z = Z[:, :Z.shape[1] // 2]
        Y = d["Y"]
    row = {}
    if G is None:
        plans = [("grouped", StratifiedKFold(5, shuffle=True, random_state=0), {})]
    else:
        assert len(Y) == len(G)
        plans = [("grouped", StratifiedGroupKFold(5, shuffle=True, random_state=0),
                  dict(groups=G)),
                 ("shuffled", StratifiedKFold(5, shuffle=True, random_state=0), {})]
    for tag, splitter, kw in plans:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        pred = np.empty_like(Y)
        for tr, te in splitter.split(Z, Y, **kw):
            pred[te] = clf.fit(Z[tr], Y[tr]).predict(Z[te])
        row[tag] = dict(acc=float((pred == Y).mean()),
                        macro_f1=float(f1_score(Y, pred, average="macro")))
    out[a] = row
    sh = row.get("shuffled", {})
    print(f"{a:<12}{row['grouped']['acc']:>9.3f}{row['grouped']['macro_f1']:>8.3f}"
          f"{sh.get('acc', float('nan')):>10.3f}{sh.get('macro_f1', float('nan')):>9.3f}",
          flush=True)
# one file per arm when a single arm is requested, so the arms can be run in
# parallel and merged; the merged file is what script 19 reads
dst = (Path(f"runs/probe_f1_{ARMS[0]}{TAG}.json") if len(ARMS) == 1
       else Path(f"runs/probe_f1{TAG}.json"))
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
