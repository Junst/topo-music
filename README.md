# topo-music

When is musical structure visible in the geometry of an audio representation?

The project began as "does topographic self-organisation buy anything for music
representations?" and that branch was stopped 2026-08-29 on a pre-registered
kill criterion ([`PREREG.md`](PREREG.md) Appendix B). What replaced it is a
sharper question, and the current work is registered in
[`PREREG_SCALES.md`](PREREG_SCALES.md).

## The result worth reading first

The same MERT encoder gives two answers about whether it represents pitch class,
depending only on how much audio is pooled before you measure.

    note level  (0.2-2.5 s, NSynth)   Delta_strict = -0.005    no octave equivalence
    clip level  (20 s, GiantSteps)    rho_fifths   = +0.468    circle of fifths

Neither is a measurement error. Both are pre-registered statistics with positive
controls that fire (analytic fifths control rho = +0.977; chromagram control
Delta_12 = +0.697) and matched nulls that do not.

Three follow-ups say what is going on, and none of them is what we expected:

- **A projection fitted only on GiantSteps track keys creates note-level octave
  equivalence in NSynth.** No notes, no NSynth, no octave information enters the
  fit. Delta_strict goes from -0.005 to +0.223 for MERT L12 using **0.13%** of
  its variance, and from +0.004 to +0.466 for a log-CQT. Matched random
  subspaces give ~0. [`runs/SUBSPACE_results.md`](runs/SUBSPACE_results.md)
- **But deleting that subspace changes nothing.** The clip-level fifths geometry
  survives removal of its own best subspace in every high-dimensional arm
  (<= 12% drop). Tonal structure is low-energy *and redundantly distributed*, so
  a low-dimensional projection denoises it rather than isolating it.
- **Key is decodable from a log-CQT better than from MERT (.492 vs .456) with
  exactly zero fifths geometry in its centroids (-0.035) -- and its classifier
  weights show that geometry strongly (+0.560).** Controls rule out the
  artefact reading: random features fall to chance and show none of it, and
  orthogonal per-key codes decode key *better* than any real arm (.651) while
  showing none of it either.

So musical information, musical geometry, and decodability are three different
properties. The information is present and linearly accessible even in the
spectrogram; whether it appears as geometry depends on temporal scale and on
which fraction of a percent of the variance you look at.

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

### One correction, from the key-level test

The statement above ("no evidence of emergent chromatic, octave-equivalent or
circle-of-fifths topology") is too flat, and the clip-level GiantSteps test says
so. Full numbers in [`runs/KEY_geometry_results.md`](runs/KEY_geometry_results.md).

| arm | rho_fifths | z | p |
|---|---|---|---|
| chroma (music-theoretic baseline) | **+0.451** | 8.90 | <0.001 |
| log-CQT (input floor) | -0.010 | -0.24 | 0.814 |
| MERT L4 | **+0.141** | 4.01 | <0.001 |
| MERT L24 | +0.074 | 2.03 | 0.038 |

Mean chroma recovers the circle of fifths cleanly and recovers *only* that
(rho_chroma = 0.001), so the test has power and a null elsewhere means
something. log-CQT has no fifths structure — octave folding is required, and an
unfolded spectral input does not have it. **MERT builds a weak circle-of-fifths
geometry that its input lacks** (0.141 at L4). That is a real positive, and it
corrects the flat claim.

It is still about a third of the explicit-chroma effect and it *decays with
depth* (0.141 -> 0.074), the same direction as everything else here. And whether
it is tonal or sociological is unsettled: in EDM key correlates with subgenre.
That rho_chroma stays ~0 while rho_fifths is positive argues for the tonal
reading, but separating them needs a key-balanced corpus or transposed tracks.

So the accurate version of the headline: the *strong* form of emergent harmonic
topology — a dominant chromatic axis a topographic objective could amplify — is
not supported. A faint one exists in mid-depth MERT and is discarded by deeper
layers.

**Also worth recording as a trap:** rho_mode is ~0.3 and highly significant in
log-CQT and every MERT layer, but only 0.070 and non-significant in chroma.
Major/minor is separating those centroids through timbre and production, not
pitch-class content. Reported without the chroma control it would look like a
music-theoretic finding.

### Still open

The frame-level GS probe (`scripts/06_probe_gs.py`, `runs/gs_*.npz`) is kept for
the record but should not be cited: nMI(code; tonic) was 0.035-0.107, so its
fifths null had no power. The clip-level test (`scripts/08_key_geometry.py`)
supersedes it.

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
