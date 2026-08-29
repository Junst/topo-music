# Pre-registration: L1' — are existing audio RVQ codebooks already topographic?

Written 2026-08-29, **before any L1' result existed**. `scripts/02_geometry.py`
was running at the time of writing and had produced no output; no probe run had
been launched. Nothing below may be edited once the first result is read;
revisions go in a dated appendix stating what changed and why.

`stem-set-jepa` fired its kill criterion and was honoured. That is the standard
this document is held to.

---

## 1. Why this is the first experiment

The project ("topographic codebook": lay RVQ codes out on a grid so that
adjacency is perceptual adjacency) is only worth doing if today's codebooks are
**not already** organised that way. If they are, the objective buys nothing and
the paper has no premise. So the premise is tested before anything is built.

## 2. Claim under test

> In the RVQ codebooks that music generation systems actually use, a code's
> musical content (preferred pitch, preferred instrument) is **not** a smooth
> function of the code's position in codebook space, and is unrelated to its
> index.

## 3. Codecs

| codec | why |
|---|---|
| `facebook/encodec_32khz` | MusicGen's codec. Music-trained. Primary. |
| `descript/dac_44khz` | DAC, widely reused in music systems. Primary. |
| `facebook/encodec_24khz` | general-audio control. If structure appears here too, it is not a music fact. |

## 4. Probe

NSynth `nsynth-test` (4096 single notes, MIDI pitch 9–119, 10 instrument
families), steady-state window [0.2 s, 2.5 s]. Recorded limitation: NSynth is
16 kHz, so all codecs are fed audio with no energy above 8 kHz. This biases
**timbre** attribution and is stated wherever a timbre number is reported. It
does not bias pitch attribution, which is the primary statistic.

## 5. Statistics and thresholds

Primary statistic: **Moran's I of the per-code preferred-pitch field on a
k=10 NN graph in codebook space**, against a 200-draw label-permutation null.
Secondary: the same on the best free 2D layout (UMAP) — an upper bound, since a
free 2D embedding is strictly easier than a discrete grid.

| outcome | condition | verdict |
|---|---|---|
| **A. already topographic** | Moran's I ≥ 0.30 **and** z ≥ 10 on ≥2 of 3 codecs, on the level carrying most pitch nMI | **KILL.** Codebooks are already organised; no objective needed. |
| **B. informative, not organised** | nMI(code; pitch) ≥ 0.15 on ≥1 level **and** Moran's I < 0.15 | **PROCEED.** The information exists, the geometry does not. This is the case the project needs. |
| **C. not informative** | nMI(code; pitch) < 0.05 on every level of every codec | **HOLD.** Codes do not carry pitch under this probe. Fix the probe (polyphonic / longer / higher-bandwidth audio) before concluding anything about codebooks. Not a kill. |
| **D. mixed** | anything else | Written up as-is. No post-hoc threshold moves. |

Thresholds are set here, in ignorance of the results, and are not to be tuned
afterwards.

## 6. Secondary, non-decisive

- **L1'-d, RVQ depth factorisation.** nMI(code; pitch) and nMI(code; family)
  per level. Hypothesis: coarse levels carry pitch, deep levels carry timbre.
  This is *exploratory*. It does not gate the project and no threshold is set;
  if it holds it supplies a factorisation axis needing no augmentation and no
  labels, which would replace the group machinery the project would otherwise
  have to build.
- **Octave equivalence.** Whether chroma distance predicts code distance beyond
  absolute pitch distance. Exploratory; a positive result here is a finding,
  a negative one is not evidence against anything.

## 7. Known confounds, declared in advance

- **Index order is trivially arbitrary.** k-means init guarantees it. The
  index-order statistics are reported to document the premise, not as a result.
- **Codebook dimensionality differs by codec.** DAC quantises in an 8-D
  projected space; EnCodec in 128-D. A 2D layout is a far smaller ask from 8-D.
  Moran's I is compared *within* codec across levels, never across codecs, and
  any cross-codec statement must carry this caveat.
- **Usage is long-tailed.** Codes seen <20 times are dropped. This is set here
  and not varied to move a result.

---

## Appendix A — 2026-08-29, written before the full probe returned

A 512-clip preliminary probe of `encodec_32k` (`runs/_pre_encodec_32k.npz`,
1/8 of the registered probe) gives, on RVQ level 0:

    Moran's I (preferred pitch | codebook space, k=10) = +0.585,  z = 44.2
    Moran's I (preferred pitch | best 2D layout)       = +0.629,  z = 42.9
    Moran's I (preferred pitch | index order)          = +0.010,  z =  0.2, p = 0.42
    nMI(code; pitch) = 0.463

That is **outcome A** on this codec (threshold: I >= 0.30 and z >= 10). Outcome
A requires 2 of 3 codecs; the full probe is running and decides.

The §2 claim was that a code's musical content is "not a smooth function of the
code's position in codebook space, **and** is unrelated to its index." The
second half holds cleanly. The first half is refuted on this codec.

**Observation, not a threshold change.** The two halves come apart in a way §2
conflated: pitch is smooth over the *embedding*, and absent from the *index* an
LM actually predicts. Whether that gap is worth anything is a new question, not
a rescue of this one, and it does not move the criterion. Outcome A stands as
written and is applied to the full result.

---

## Appendix B — 2026-08-29, verdict

Full probe: 4096 NSynth notes per codec, 200-draw permutation null.
Decision rule applied exactly as written in §5: Moran's I of the preferred-pitch
field on a k=10 NN graph in codebook space, on **the level carrying most pitch
nMI**, on >=2 of 3 codecs.

| codec | max-nMI level | nMI(pitch) | Moran's I | z | meets A (I>=0.30, z>=10) |
|---|---|---|---|---|---|
| encodec_32k | L0 | 0.378 | **0.659** | 67.1 | **yes** |
| encodec_24k | L1 | 0.339 | **0.398** | 44.0 | **yes** |
| dac_44k | L0 | 0.165 | 0.291 | 19.9 | no (misses by 0.009) |

Two of three. **Outcome A fires. The project is killed.**

`dac_44k` missing the threshold by 0.009 is recorded and changes nothing: the
rule was ">=2 of 3", it is met by the other two, and the threshold is not moved
in either direction after the fact.

### What was refuted

§2 claimed a code's musical content is "not a smooth function of the code's
position in codebook space, **and** is unrelated to its index."

- **Second half holds.** Moran's I on index order is 0.034 / 0.028 / -0.020
  (p = 0.09 / 0.23 / 0.72). Indices are an arbitrary permutation.
- **First half is refuted.** Pitch is strongly smooth over codebook geometry in
  both EnCodec variants, and the effect is stronger still on the best 2D layout
  (I = 0.716 for encodec_32k L0). Shipped codebooks are already pitch-
  topographic; a training objective has little left to induce.

### Secondary results, reported because they were pre-registered

- **No octave equivalence anywhere.** Partial Spearman of chroma distance
  against code distance, controlling for absolute pitch distance, is -0.022 /
  +0.029 / -0.007 on the three max-nMI levels; the largest value anywhere is
  +0.103. The organisation is **tonotopic** (pitch height), not chromatic.
  Nothing resembling a Tonnetz, a circle of fifths, or octave circularity is
  present. Absolute pitch distance does correlate (rho up to +0.31).
- **L1'-d, RVQ depth factorisation: not supported.** Both nMI(pitch) and
  nMI(family) decay monotonically with depth in all three codecs; deep levels
  carry less of everything rather than trading pitch for timbre. The pitch:family
  *ratio* does shift toward family with depth in encodec_24k (1.07 at L0 to 0.36
  at L16) but not in encodec_32k, and dac_44k already has family > pitch at L0.
  This was declared exploratory and non-decisive; it is reported as mixed and is
  not used to argue anything.
- **Unexplained:** encodec_24k levels 18-31 show strong *negative* Moran's I
  (to -0.399, z = -40.9). Those levels have nMI(pitch) ~0.03, i.e. below the
  outcome-C informativeness floor, so this is not decision-relevant. It is
  recorded as unexplained rather than interpreted.

### Scope of the kill

This kills the topographic-codebook project: RVQ codebooks as shipped. It does
**not** test topography in continuous SSL encoders (ideas 3 and 4), and L0 (SOM
on raw CQT) was never run.

The octave-equivalence result is stated at the strength the evidence supports:
**the observed organisation is substantially more consistent with acoustic
pitch-height organisation than with music-theoretic chromatic organisation.**
It is *not* claimed that spectral similarity alone explains it. That claim needs
spectral distance regressed out with the pitch effect then vanishing, which is
E2 below and has not been run. An earlier draft of this appendix overstated it.

### The result that is actually interesting

The kill is the least informative thing here. The finding is the gap:

    Moran's I (pitch | codebook geometry) = 0.659
    Moran's I (pitch | code index)        = 0.034,  p = 0.09

EnCodec receives no pitch supervision, yet pitch emerges as a principal axis of
its codebook geometry — and is then discarded into an arbitrary categorical ID.
The codec builds a pitch-structured geometry and throws the structure away at
the interface the LM actually consumes.

And what it organises is absolute pitch height, not pitch class: C3 and C4 are
*not* neighbours. On this evidence a codec learns an **acoustic** equivalence
relation, not a **musical** one.

That reframes the next experiment. It is no longer "does topography emerge"
but **which topology emerges under which training objective**:

| | pitch height | chroma | fifths | octave equivalence |
|---|---|---|---|---|
| codec (EnCodec/DAC) | strong | absent | — | absent |
| music SSL (MERT) | ? | ? | ? | ? |

If MERT also gives rho_abs >> rho_chroma, idea 3 ("emergent harmonic topology")
is close to dead too. If chroma and octave equivalence appear in MERT but not in
the codec, the question becomes *representation objectives convert acoustic
topology into musical topology* — a better question than the one this project
started from.


---

## Appendix C — reproducibility, 2026-08-29

The verdict numbers were recomputed from the stored probes after the fact
(`runs/verify/`). encodec_32k and encodec_24k reproduce exactly. dac_44k differs
in the 4th decimal (used codes 756 vs 752, Moran's I 0.2913 vs 0.2912, z 19.9 vs
19.1): the probe was run twice, once on CPU and once on GPU, and float
non-determinism in the encoder moves a handful of codes across the MIN_COUNT=20
usage cutoff. No conclusion depends on it -- dac_44k sits below the 0.30
threshold either way.
