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

## E2 / E3 — the follow-ups, now run

Full numbers in [`runs/E2_E3_results.md`](runs/E2_E3_results.md). Eleven arms,
all quantised the same way (k-means K=1024 over the same 4096 NSynth notes) so
that discrete-vs-continuous is not a confound.

| | pitch height | chroma | fifths | octave equivalence |
|---|---|---|---|---|
| log-CQT (input floor) | **strongest of all** | absent | absent | absent |
| music SSL (MERT, 7 layers) | strong, **grows with depth** | absent, **shrinks with depth** | absent | absent |
| codec (EnCodec / DAC) | present | absent | absent | absent |

**Music SSL does not produce musical topology either.** rho_chroma <= 0.107
everywhere, rho_fifths <= 0.054. Through MERT's depth the ratio
rho_abs / rho_chroma climbs 3.7x -> 22.7x: training *sharpens* the acoustic
topology and *erodes* the chroma structure the input had. And log-CQT is the
most pitch-organised arm of the eleven, so the topography is largely inherited
from the input rather than built by any objective.

**But it is not "just spectral similarity" — an earlier draft of this README said
so and was wrong.** Controlling per-code log-mel distance barely touches MERT
(rho 0.668 -> 0.618 at L16); it removes ~40% of EnCodec's effect and all of
DAC's. The claim now stands only where the control supports it.

### What is still open

Octave equivalence is properly tested with single notes and is absent. The
Tonnetz / circle-of-fifths question is **not** properly tested here — those are
chord- and key-level relations and NSynth is isolated notes. GiantSteps
(`marble/GS`) is on disk if that test is worth running.

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
