"""Clip-level key geometry -- the test the frame-level GS probe could not do.

Why the frame-level probe failed. Key is a distributional property over time; a
20 ms frame barely carries one. nMI(code; tonic) came out 0.035-0.107 against
0.691 for nMI(code; pitch) on NSynth, so a null fifths correlation there meant
"this design cannot see key", not "there is no key geometry". Quantising frames
destroys the very thing being measured.

So: one embedding per clip (mean-pooled), and two things measured in order.

1. POWER FIRST. A 24-way linear probe on the clip embeddings. Absence of
   geometry is only interpretable if the representation demonstrably encodes
   key at all. Chance is 1/24 = 4.2%.

2. GEOMETRY. Average the clip embeddings within each of the 24 keys to get 24
   key centroids, then ask whether the distances among those centroids follow
   known musical key structure. Averaging within class is what removes the
   timbre/genre variance that swamped the frame-level test, and 24 centroids is
   exactly the object the circle of fifths and Krumhansl's key torus describe.

   Reference structures, all reported separately rather than folded into one
   invented "torus metric":
     - Krumhansl-Kessler key distance: 1 - corr(profile_i, profile_j) over the
       standard KK major/minor profiles rotated to each tonic. This is the
       empirical human key-similarity structure and the primary statistic.
     - circle-of-fifths distance between tonics.
     - chromatic distance between tonics.
     - mode agreement (major vs minor).
   Each against a permutation null over key labels, and each with per-clip mean
   log-mel distance partialled out, same control as E2.

Two poolings, because key is a temporal-distributional property and a plain
mean may discard exactly the part of MERT that carries it: `mean`, and
`mean+std` (concatenated). Attention pooling would be the next step and is
deliberately not taken here -- it would turn a control experiment into a
modelling one.

On the genre confound: GiantSteps is entirely electronic dance music, so every
clip pair is effectively within-genre already and the usual genre stratification
has little left to remove. Production style still varies, which is what the
log-mel partial is for.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GS = Path("/lustre/dataset/musicdataset/marble/GS")
CQT_SR, MERT_SR = 16000, 24000
TONIC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
ENH = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# Krumhansl & Kessler (1982) probe-tone profiles.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def parse_key(lab):
    parts = lab.strip().split()
    if len(parts) != 2:
        return None
    t, mode = parts
    t = ENH.get(t, t)
    if t not in TONIC or mode.lower() not in ("major", "minor"):
        return None
    return TONIC[t], 0 if mode.lower() == "major" else 1


def kk_profiles():
    """24 profiles, index = tonic*2 + mode."""
    P = np.zeros((24, 12))
    for t in range(12):
        P[t * 2 + 0] = np.roll(KK_MAJOR, t)
        P[t * 2 + 1] = np.roll(KK_MINOR, t)
    return P


def circ12(d):
    m = np.asarray(d) % 12
    return np.minimum(m, 12 - m)


def partial_spearman(x, y, z):
    rxy, rxz, ryz = (spearmanr(x, y).statistic, spearmanr(x, z).statistic,
                     spearmanr(y, z).statistic)
    den = np.sqrt(max(1 - rxz ** 2, 1e-12) * max(1 - ryz ** 2, 1e-12))
    return float((rxy - rxz * ryz) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import librosa
    rows = []
    for sp in a.splits:
        for line in (GS / sp).read_text().splitlines():
            r = json.loads(line)
            km = parse_key(r["label"])
            p = GS / Path(r["audio_path"]).relative_to("data/GS")
            if km is not None and p.exists():
                rows.append((p, *km))
    print(f"[{a.arm}] {len(rows)} clips", flush=True)

    is_codec = a.arm.startswith(("encodec", "dac"))
    model = taps = codec = None
    layer = None
    if is_codec:
        from topo.codecs import load
        codec = load(a.arm, device=a.device)
        native_sr = codec.sr
        CB = torch.from_numpy(codec.codebooks)                    # [L,K,d]
    elif a.arm.startswith("mert_L"):
        from transformers import AutoModel
        layer = int(a.arm.split("_L")[1])
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M",
                                          trust_remote_code=True).to(a.device).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        taps = {}

        def tap(i):
            def fn(_m, _in, o):
                taps[i] = o[0] if isinstance(o, tuple) else o
            return fn
        model.feature_projection.register_forward_hook(tap(0))
        for i, lyr in enumerate(model.encoder.layers):
            lyr.register_forward_hook(tap(i + 1))
        native_sr = MERT_SR
    else:
        native_sr = CQT_SR

    def batch_embed(paths):
        ws = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32", frames=int(a.seconds * 44100),
                            start=int(2.0 * 44100))
            if x.ndim > 1:
                x = x.mean(1)
            ws.append(librosa.resample(x, orig_sr=sr, target_sr=native_sr)
                      if sr != native_sr else x)
        n = min(len(w) for w in ws)
        ws = [w[:n] for w in ws]
        if a.arm == "chroma":
            # Explicit music-theoretic positive control. A chromagram is octave-
            # folded by construction, so if 24 key centroids can show a circle of
            # fifths to this method at all, they must show it here. A null on
            # every learned arm means nothing without this one coming out
            # positive -- it is what separates "no geometry" from "broken metric".
            F = [librosa.feature.chroma_cqt(y=w, sr=native_sr, hop_length=512).T
                 for w in ws]
        elif a.arm == "cqt":
            F = [librosa.amplitude_to_db(np.abs(librosa.cqt(
                w, sr=native_sr, hop_length=512, fmin=librosa.note_to_hz("C1"),
                n_bins=84, bins_per_octave=12)), ref=np.max).T for w in ws]
        elif is_codec:
            codes = codec.encode(torch.from_numpy(np.stack(ws)).unsqueeze(1))  # [B,L,T]
            # the quantised latent the codec actually emits: sum over RVQ levels
            lat = sum(CB[l][codes[:, l]] for l in range(CB.shape[0]))          # [B,T,d]
            F = list(lat.numpy())
        else:
            taps.clear()
            with torch.no_grad():
                model(torch.from_numpy(np.stack(ws)).to(a.device))
            F = list(taps[layer].float().cpu().numpy())
        E = [np.concatenate([f.mean(0), f.std(0)]) for f in F]   # [mean | std]
        mel = [librosa.power_to_db(librosa.feature.melspectrogram(
            y=w, sr=native_sr, n_fft=2048, hop_length=512, n_mels=a.n_mels)).mean(1)
            for w in ws]
        return E, mel

    t0 = time.time()
    Z, S, Y = [], [], []
    for b0 in range(0, len(rows), a.batch):
        br = rows[b0:b0 + a.batch]
        E, M = batch_embed([r[0] for r in br])
        Z.extend(E); S.extend(M)
        Y.extend([r[1] * 2 + r[2] for r in br])
        if b0 % (a.batch * 60) == 0:
            print(f"{b0+len(br)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    Z = np.stack(Z).astype(np.float64)          # [N, 2D] = [mean | std]
    S = np.stack(S).astype(np.float64)
    Y = np.array(Y)
    print(f"[{a.arm}] embeddings {Z.shape} ({time.time()-t0:.0f}s)", flush=True)

    D = Z.shape[1] // 2
    out = {"arm": a.arm, "n_clips": int(len(Y)), "dim": int(D), "poolings": {}}
    POOLINGS = {"mean": Z[:, :D], "mean+std": Z}

    # ---- 1. power ---------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    present = np.array(sorted(set(Y.tolist())))
    iu = np.triu_indices(len(present), 1)
    Csp = np.stack([S[Y == k].mean(0) for k in present])
    dsp = squareform(pdist(Csp))[iu]
    P = kk_profiles()[present]
    kk = 1.0 - np.corrcoef(P)[iu]
    ton = present // 2
    dchr = circ12(np.abs(ton[:, None] - ton[None, :]))[iu]
    dfif = circ12(np.abs(((ton * 7) % 12)[:, None] - ((ton * 7) % 12)[None, :]))[iu]
    dmode = (np.abs((present % 2)[:, None] - (present % 2)[None, :]))[iu].astype(float)
    out["n_keys_present"] = int(len(present))

    for pool_name, ZP in POOLINGS.items():
        print(f"\n[{a.arm}] pooling = {pool_name}  dim={ZP.shape[1]}", flush=True)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        acc = cross_val_score(clf, ZP, Y, cv=5, scoring="accuracy")
        acc_t = cross_val_score(clf, ZP, Y // 2, cv=5, scoring="accuracy")
        rec = {"probe": dict(key24_acc=float(acc.mean()), key24_std=float(acc.std()),
                             key24_chance=1 / 24, tonic12_acc=float(acc_t.mean()),
                             tonic12_chance=1 / 12)}
        print(f"  24-way key {acc.mean():.3f} (chance .042) | "
              f"12-way tonic {acc_t.mean():.3f} (chance .083)", flush=True)

        C = np.stack([ZP[Y == k].mean(0) for k in present])
        dz = squareform(pdist(C))[iu]
        rng = np.random.default_rng(a.seed)
        res = {}
        for nm, dd in [("kk_key_distance", kk), ("fifths", dfif),
                       ("chromatic", dchr), ("mode_mismatch", dmode)]:
            obs = float(spearmanr(dd, dz).statistic)
            null = np.empty(a.n_perm)
            for i in range(a.n_perm):
                Dp = squareform(pdist(C[rng.permutation(len(present))]))[iu]
                null[i] = spearmanr(dd, Dp).statistic
            res[nm] = dict(rho=obs, null_mean=float(null.mean()),
                           null_std=float(null.std()),
                           z=float((obs - null.mean()) / (null.std() + 1e-12)),
                           p=float((np.abs(null) >= abs(obs)).mean()),
                           rho_given_spec=partial_spearman(dd, dz, dsp))
            print(f"  {nm:<18} rho={obs:+.3f}  z={res[nm]['z']:+5.1f}"
                  f"  p={res[nm]['p']:.3f}  rho|spec={res[nm]['rho_given_spec']:+.3f}",
                  flush=True)
        rec["centroid_geometry"] = res
        out["poolings"][pool_name] = rec

    p = Path(a.out or f"runs/clipkey_{a.arm}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    np.savez_compressed(p.with_suffix(".npz"), Z=Z.astype(np.float32),
                        keys=present, Y=Y)
    print("wrote", p, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
