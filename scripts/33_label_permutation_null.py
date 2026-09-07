"""A supervised null for the key-fitted projection.

The paper already reports two controls on the rank-4 subspace: a matched
random orthonormal subspace, and an unsupervised rank-4 PCA. Neither is a
control on the fitting procedure. A reader can still ask whether shrinkage LDA
on any twelve-class partition of these clips would produce low-rank directions
that happen to fold octaves, since the recovered subspace holds well under one
percent of NSynth variance and there is a great deal of room down there.

So: permute the GiantSteps key labels and refit, changing nothing else. The
permutation is over tracks rather than clips, which keeps the label
distribution and the several-clips-per-track structure intact and destroys
only the correspondence between audio and key. Everything downstream is the
pipeline of script 13 with the same seed, the same train split and the same
instrument-disjoint evaluation, so the permuted values and the reported one are
the same measurement.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from importlib.machinery import SourceFileLoader

m13 = SourceFileLoader("m13", "scripts/13_tonal_subspace.py").load_module()
m9 = m13.m9

ARMS = ["mert_L4", "mert_L12", "mert_L24", "muq_L2", "muq_L6", "muq_L12",
        "matpac_L6", "pupujepa", "encodec_32k", "cqt", "pq_stft", "chroma"]


def gs_track_ids():
    """Track of each GiantSteps clip, in script 09's row order."""
    ids = []
    for sp in ["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"]:
        for line in (m9.GS / sp).read_text().splitlines():
            r = json.loads(line)
            km = m9.parse_key(r["label"])
            p = m9.GS / Path(r["audio_path"]).relative_to("data/GS")
            if km is not None and p.exists():
                ids.append(p.stem.rsplit("-", 1)[0])
    return np.array(ids)


def d_strict(Zn, W, anchors, kmax, eval_inst, seed):
    """Point estimate only; the bootstrap is for the reported value, not the null."""
    cs, _ = m13.curve_stats(Zn @ W, anchors, kmax, eval_inst,
                            np.random.default_rng(seed), n_boot=1)
    return float(cs["d_strict"]["v"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/label_permutation_null.json")
    a = ap.parse_args()

    tracks = gs_track_ids()
    lab, gsplit = m13.gs_key_labels()
    out = {}
    for arm in a.arms:
        p = Path(f"runs/nsynth_emb_{arm}.npz")
        q = Path(f"runs/clipkey_{arm}.npz")
        if not (p.exists() and q.exists()):
            print(f"[{arm}] missing cache, skipped", flush=True)
            continue
        t0 = time.time()
        nz = np.load(p, allow_pickle=True)
        Zn = nz["Z"].astype(np.float64)
        inst, anchors, kmax = nz["inst"], nz["anchors"], int(nz["kmax"])
        names = nz["insts"]
        gz = np.load(q)
        Zg = gz["Z"].astype(np.float64)
        Zg = Zg[:, :Zg.shape[1] // 2]
        Yg = gz["Y"]
        assert len(tracks) == len(Yg) and (lab == Yg).all(), "GS row order drifted"

        # script 13's instrument split, same seed, so the null is matched
        rng = np.random.default_rng(a.seed)
        uinst = np.array(sorted(set(inst.tolist())))
        fit_inst = set(rng.choice(uinst, len(uinst) // 2, replace=False).tolist())
        eval_inst = np.array([i for i in sorted(set(anchors[:, 0].tolist()))
                              if names[i] not in fit_inst])
        tr = gsplit < 2

        obs = d_strict(Zn, m13.lda_directions(Zg[tr], Yg[tr] // 2, a.dim),
                       anchors, kmax, eval_inst, a.seed)

        # one tonic per track, shuffled between tracks, then broadcast back
        utr, inv = np.unique(tracks, return_inverse=True)
        tonic_of_track = np.zeros(len(utr), dtype=np.int64)
        for j in range(len(utr)):
            tonic_of_track[j] = (Yg[inv == j] // 2)[0]
        prng = np.random.default_rng(a.seed + 1)
        null = np.empty(a.n_perm)
        for i in range(a.n_perm):
            y = tonic_of_track[prng.permutation(len(utr))][inv]
            null[i] = d_strict(Zn, m13.lda_directions(Zg[tr], y[tr], a.dim),
                               anchors, kmax, eval_inst, a.seed)
            if i and i % 50 == 0:
                print(f"  [{arm}] {i}/{a.n_perm}  {time.time()-t0:.0f}s", flush=True)
        p_val = float((null >= obs).mean())
        out[arm] = dict(observed=obs, n_perm=int(a.n_perm), dim=int(a.dim),
                        null_mean=float(null.mean()), null_std=float(null.std()),
                        null_lo=float(np.percentile(null, 2.5)),
                        null_hi=float(np.percentile(null, 97.5)),
                        null_max=float(null.max()), p=p_val,
                        null=[float(v) for v in null])
        print(f"[{arm}] observed {obs:+.3f} | permuted mean {null.mean():+.3f} "
              f"[{np.percentile(null, 2.5):+.3f},{np.percentile(null, 97.5):+.3f}] "
              f"max {null.max():+.3f} | p={p_val:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
