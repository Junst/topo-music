# topo-music

Does topographic self-organisation buy anything for music representations?

**Status: the RVQ-codebook branch was stopped 2026-08-29 on a pre-registered
kill criterion.** Read [`PREREG.md`](PREREG.md) Appendix B for the verdict, then
[`DESIGN.md`](DESIGN.md) §0 for the prior-art collisions that shaped the scope.

## The result worth reading first

Not the kill — this gap:

    Moran's I (pitch | codebook geometry) = 0.659   (z = 67)
    Moran's I (pitch | code index)        = 0.034   (p = 0.09)

EnCodec gets no pitch supervision, yet pitch is a principal axis of its codebook
*geometry* — which is then discarded into an arbitrary categorical ID, the one
thing a downstream LM actually sees. And the axis is **absolute pitch height,
not pitch class**: C3 and C4 are not neighbours. The equivalence relation a
codec learns looks acoustic, not musical.

## The kill

The codebooks that music generation systems ship (`encodec_32khz` — MusicGen's
codec; `dac_44khz`; `encodec_24khz`) are already pitch-topographic in codebook
space, so the training objective this project was built to justify has nothing
left to induce. 2 of 3 codecs cross the pre-registered threshold.

Two secondary results matter more than the kill:

- **No octave equivalence anywhere.** Chroma distance, controlling for absolute
  pitch distance, has partial Spearman |rho| <= 0.10 against code distance. The
  organisation is substantially more consistent with acoustic pitch height than
  with music-theoretic chroma. (Stated at that strength on purpose — spectral
  distance has not yet been regressed out, so "spectral similarity alone
  explains it" is *not* claimed. That is E2 below.)
- **RVQ depth does not factorise** into pitch-then-timbre. Both nMI(pitch) and
  nMI(family) decay monotonically with depth in all three codecs.

## Where this goes next

The question is no longer "does topography emerge" but **which topology emerges
under which training objective**:

| | pitch height | chroma | fifths | octave equivalence |
|---|---|---|---|---|
| codec (EnCodec / DAC) | strong | absent | — | absent |
| music SSL (MERT) | ? | ? | ? | ? |

- **E2 — spectral control.** Regress spectral distance out and check whether the
  pitch effect survives. Needed before any "it's just spectral similarity" claim.
- **E3 — MERT, same probe, same metrics.** If MERT also gives
  rho_abs >> rho_chroma, the "emergent harmonic topology" idea is close to dead
  too. If chroma and octave equivalence appear in MERT but not in the codec, the
  subject becomes *representation objectives converting acoustic topology into
  musical topology* — a better question than the one this repo started from.

## Layout

    DESIGN.md    scope, prior art (FSQ / Topographic VAE / Toiviainen), the ladder
    PREREG.md    claim, thresholds, verdict, reproducibility. Written before results.
    topo/codecs.py                     codec loading + codebook extraction
    scripts/02_geometry.py             L1'-a/b  index order, intrinsic dim, 2D-embeddability
    scripts/03_probe_nsynth.py         NSynth pitch/instrument attribution per code
    scripts/04_semantic_topography.py  L1'-c/d  Moran's I, octave test, depth profile
    runs/                              all numbers behind the verdict
    runs/verify/                       independent recomputation (PREREG Appendix C)

## Reproducing

    uv venv --python 3.11 .venv
    uv pip install torch torchaudio transformers numpy scipy scikit-learn \
                   librosa soundfile matplotlib umap-learn
    export HF_HOME=...
    .venv/bin/python scripts/02_geometry.py
    for c in encodec_32k dac_44k encodec_24k; do
      .venv/bin/python scripts/03_probe_nsynth.py --codec $c --batch 32
      .venv/bin/python scripts/04_semantic_topography.py --probe runs/probe_$c.npz
    done

NSynth is read from `/lustre/dataset/musicdataset/marble/nsynth`. Its 16 kHz
sample rate biases every timbre number; see `PREREG.md` §4.
