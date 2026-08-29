"""L1'-c input: attach musical meaning to each code index.

NSynth gives single notes labelled with pitch (MIDI) and instrument family.
Encoding them through a codec tells us, for every code index in every RVQ
level, the distribution of pitches and instruments that select it.

Only the steady-state part of each note is used ([0.2 s, 2.5 s]); NSynth notes
are a 3 s note plus 1 s decay, and the attack transient would smear the
pitch/timbre attribution we are about to measure.

Caveat recorded on purpose: NSynth audio is 16 kHz, so every codec here is fed
upsampled audio with no energy above 8 kHz. That biases *timbre* attribution
(bright/dark distinctions live above 8 kHz) and must be stated in any writeup.
It does not bias pitch attribution, which is the primary L1'-c statistic.

Output: runs/probe_<codec>.npz with
    pitch_counts  [L, K, n_pitch]   int32
    fam_counts    [L, K, n_family]  int32
    src_counts    [L, K, n_source]  int32
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topo.codecs import load

NSYNTH = Path("/lustre/dataset/musicdataset/marble/nsynth")
WIN = (0.2, 2.5)          # steady-state window, seconds
NSYNTH_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True)
    ap.add_argument("--split", default="nsynth-test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.loads((NSYNTH / a.split / "examples.json").read_text())
    keys = sorted(meta)
    if a.limit:
        keys = keys[:: max(1, len(keys) // a.limit)][: a.limit]

    pitches = sorted({meta[k]["pitch"] for k in keys})
    fams = sorted({meta[k]["instrument_family"] for k in keys})
    srcs = sorted({meta[k]["instrument_source"] for k in keys})
    p_ix = {p: i for i, p in enumerate(pitches)}
    f_ix = {f: i for i, f in enumerate(fams)}
    s_ix = {s: i for i, s in enumerate(srcs)}

    c = load(a.codec, device=a.device)
    L, K, _ = c.codebooks.shape
    pc = np.zeros((L, K, len(pitches)), np.int32)
    fc = np.zeros((L, K, len(fams)), np.int32)
    sc = np.zeros((L, K, len(srcs)), np.int32)

    rs = torchaudio.transforms.Resample(NSYNTH_SR, c.sr) if c.sr != NSYNTH_SR else None
    lo, hi = int(WIN[0] * NSYNTH_SR), int(WIN[1] * NSYNTH_SR)

    t0 = time.time()
    for b0 in range(0, len(keys), a.batch):
        bk = keys[b0 : b0 + a.batch]
        wavs = []
        for k in bk:
            x, sr = sf.read(NSYNTH / a.split / "audio" / f"{k}.wav", dtype="float32")
            assert sr == NSYNTH_SR, sr
            w = torch.from_numpy(x[lo:hi])
            wavs.append(rs(w) if rs is not None else w)
        n = min(w.shape[-1] for w in wavs)
        wav = torch.stack([w[:n] for w in wavs]).unsqueeze(1)     # [B,1,T]

        codes = c.encode(wav).numpy()                              # [B,L,T']
        for j, k in enumerate(bk):
            pi, fi, si = (p_ix[meta[k]["pitch"]], f_ix[meta[k]["instrument_family"]],
                          s_ix[meta[k]["instrument_source"]])
            for l in range(L):
                u, cnt = np.unique(codes[j, l], return_counts=True)
                pc[l, u, pi] += cnt.astype(np.int32)
                fc[l, u, fi] += cnt.astype(np.int32)
                sc[l, u, si] += cnt.astype(np.int32)

        if b0 % (a.batch * 20) == 0:
            done = b0 + len(bk)
            print(f"{done}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)

    out = Path(a.out or f"runs/probe_{a.codec}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pitch_counts=pc, fam_counts=fc, src_counts=sc,
        pitches=np.array(pitches), families=np.array(fams), sources=np.array(srcs),
        codec=a.codec, split=a.split, n_clips=len(keys),
    )
    print("wrote", out, "in", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
