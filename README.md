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

**Music SSL does not produce musical topology either.** Chroma and fifths
correlations remained small everywhere (rho_chroma <= 0.107, rho_fifths <=
0.054). Through MERT's depth the ratio rho_abs / rho_chroma climbs
3.7x -> 22.7x: training *sharpens* the acoustic topology and *erodes* the chroma
structure the input had. And log-CQT is the most pitch-organised arm of the
eleven — with Moran's I 0.760 already in the input, a network reaching 0.805 did
not invent the phenomenon.

But MERT is not merely inheriting it either: rho_abs 0.668 at L16 against
log-CQT's 0.465. The accurate statement is that **MERT transforms an inherited
acoustic topology by selectively amplifying pitch-height organisation** — the
opposite of converting acoustic geometry into musical geometry.

**And it is not "just spectral similarity" — an earlier draft of this README said
so and was wrong.** Controlling per-code log-mel distance barely touches MERT
(0.668 -> 0.618 at L16); it removes ~40% of EnCodec's effect and all of DAC's.
Stated at the strength the control supports: *MERT's pitch-height organisation
cannot be explained by coarse log-mel spectral similarity alone.* Harmonic
spacing, resolved harmonics, spectral envelope and F0 periodicity are not
controlled, so this is not evidence that MERT learns abstract pitch.

## Where it lands

    Music representations are topographic, but not musically topographic.

Neither neural audio codecs nor a strong self-supervised music encoder show
evidence of emergent chromatic, octave-equivalent or circle-of-fifths topology
under these probes; their geometry is consistently dominated by absolute pitch
height. In MERT, increasing depth strengthens the dominance of pitch height over
pitch class rather than converting acoustic organisation into music-theoretic
organisation.

That is a more useful finding than a null. C3 and C4 are not, to these models,
the special equivalence relation that a musician takes them to be — and the
question worth asking next is **why music SSL does not form octave / chroma
invariance**, which points straight at what pitch-shift and contrastive
augmentation do to representation geometry.

The question this repo should have been asking from the start is not "does
musical topology emerge" but *what topology actually emerges across audio
representation objectives* — with CQT -> codec -> music SSL as the axis. The
spectral control already orders that axis: rho_pitch|spec is -0.054 for DAC,
0.185 for EnCodec, 0.618 for MERT L16.

### Still open

Octave equivalence is properly tested with single notes and is absent. Chord-
and key-level topology is tested separately on GiantSteps (`scripts/06_probe_gs.py`,
`runs/gs_*.npz`) because NSynth's isolated notes cannot carry a key.

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
