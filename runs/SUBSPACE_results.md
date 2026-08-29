# Tonal subspace: verdict against `PREREG_SCALES.md`, 2026-08-30

Seven arms. Subspaces are orthonormal bases of `d` shrunk LDA discriminant
directions. Anything measured on NSynth is fitted on one half of the
instruments and evaluated on the other (75 held-out instruments); the GS side
of `key_lda` is fitted on GS.train+val and evaluated on GS.test, against an
ambient baseline recomputed on those same clips.

## S3 -- cross-dataset transfer. **PASS in every arm.**

The decisive cell. `key_lda` sees only GiantSteps track tonics: no notes, no
NSynth, no octave information of any kind. Applied unchanged to the NSynth
transposition curve at `d = 4`:

| arm | ambient Delta_12 | ambient Delta_strict | projected Delta_12 | projected Delta_strict | projected beta_2 | % of NSynth variance |
|---|---|---|---|---|---|---|
| chroma (control) | +0.661 | +0.662 | +1.254* | +0.878* | -0.012* | 31.9 |
| cqt | +0.004 | +0.004 | +0.432* | **+0.466*** | +0.030* | 1.85 |
| encodec_32k | +0.039 | +0.000 | +0.616* | +0.459* | +0.013* | 1.07 |
| mert_L4 | +0.014 | -0.001 | +0.396* | +0.304* | +0.014* | **0.11** |
| mert_L12 | +0.015 | -0.005 | +0.251* | +0.223* | +0.013* | 0.13 |
| mert_L16 | +0.011 | -0.009 | +0.121* | +0.086* | +0.005* | 0.17 |
| mert_L24 | +0.003 | -0.003 | +0.234* | +0.154* | +0.003 | 0.24 |

`*` = instrument bootstrap CI (2000 resamples) excludes 0.

Every ambient Delta_strict is zero or negative. Every projected Delta_strict is
positive with a CI excluding 0, and Delta_12 and beta_2 agree in sign in six of
seven arms (mert_L24's beta_2 CI includes 0). The pre-registered requirement
that the three statistics agree is met.

**C1 random-subspace control, same evaluation instruments, `d = 4`:** cqt
+0.007+-0.008, encodec +0.003+-0.028, mert_L4 -0.002+-0.008, mert_L12
-0.019+-0.018, mert_L16 -0.011+-0.016, mert_L24 -0.003+-0.003. Random
projections do not produce octave equivalence; the learned ones exceed them by
1-2 orders of magnitude. The effect is not a projection artefact.

**C2 absolute-pitch control:** at matched low `d` an LDA fitted to absolute
pitch rather than pitch class gives *negative* Delta_strict (mert_L12: -0.116
at d=1, -0.077 at d=2, against pc_lda's +0.004 and +0.023). It only turns
positive by d>=4 and then costs four times the variance. The octave-invariant
objective is doing the work.

## S2 -- low energy. **PASS, by a wide margin.**

The threshold was 5%. MERT needs **0.11-0.24%** of its variance to carry the
effect; EnCodec 1.1%, log-CQT 1.9%. The chromagram control needs 32%, which is
what a representation that is *mostly* pitch class looks like and calibrates the
others.

MERT is the most compact of the real arms by an order of magnitude. That is a
statement about concentration, not about presence: the structure is in log-CQT
too, just spread over twenty times more variance.

## S1 -- within-dataset recovery. **PASS**, and weak evidence as declared.

`pc_lda` on held-out NSynth instruments, mert_L12: Delta_12 +0.021*,
Delta_strict +0.023*, beta_2 +0.120* at d=2, rising to +0.361*/+0.361*/+0.069*
at d=11, against an ambient Delta_strict of -0.005. Registered in advance as
the weak cell -- it fits and evaluates on the same corpus with an objective
that is octave-invariant by construction -- and it is reported for completeness
rather than leaned on. S3 is the evidence.

## S4 -- mechanism. **FAILS**, and the failure is the interesting part.

Removing the subspace, `z - W W^T z`, was predicted to at least halve the
clip-level rho_fifths. It does not move it. At `d = 4`, complement vs
split-matched ambient:

| arm | pc_lda ambient | pc_lda complement | drop | key_lda ambient | key_lda complement | drop |
|---|---|---|---|---|---|---|
| mert_L4 | +0.445 | +0.446 | 0% | +0.118 | +0.114 | 3% |
| mert_L12 | +0.468 | +0.471 | -1% | +0.127 | +0.119 | 6% |
| mert_L16 | +0.395 | +0.398 | -1% | +0.101 | +0.096 | 6% |
| mert_L24 | +0.294 | +0.295 | 0% | +0.063 | +0.055 | 12% |
| encodec_32k | +0.177 | +0.139 | 21% | -0.023 | -0.047 | - |
| chroma | +0.475 | +0.143 | 70% | +0.400 | +0.053 | 87% |

Only the chromagram collapses, and it is 12-dimensional, so removing four
directions removes a third of the space -- that is a dimensionality effect, not
evidence of localisation. In every high-dimensional representation the geometry
survives its own best subspace being deleted.

So the hypothesis as written is half right. Tonal structure **is**
low-energy and **is** invisible in the ambient metric, but it is **not**
confined to a compact subspace. It is redundantly distributed: many overlapping
sets of directions carry it, a low-dimensional projection *denoises* it rather
than isolating it, and deleting any one of them leaves the rest.

## An asymmetry worth reporting

The reverse transfer -- `pc_lda` fitted on NSynth single-note pitch class,
applied to GiantSteps key centroids -- needs `d >= 8` and then works in every
learned or codec representation, exceeding the ambient value:

| arm | ambient rho_fifths | d=4 | d=8 | d=11 |
|---|---|---|---|---|
| mert_L4 | +0.445 | +0.016 | **+0.662** | +0.662 |
| mert_L12 | +0.468 | -0.070 | +0.656 | +0.655 |
| mert_L24 | +0.294 | -0.095 | +0.585 | +0.593 |
| encodec_32k | +0.177 | +0.448 | +0.682 | +0.669 |
| chroma | +0.475 | +0.410 | +0.528 | +0.484 |
| **cqt** | -0.035 | -0.126 | **-0.164** | -0.172 |

log-CQT is the exception, and it is the arm where the *forward* transfer was
strongest (+0.466). Pitch-class directions estimated from isolated notes in a
log-CQT are particular frequency bins, and they do not survive the move to
polyphonic mixtures; the same directions estimated in a learned representation
do. That is a concrete sense in which learned representations are more abstract
than the spectrogram, measured rather than asserted -- and it is invisible to
any single-dataset probe.

## What the four results say together

Musical information, its geometric expression, and decodability are three
different things. Tonal structure is present and linearly accessible in every
representation tested including the log-spectrogram; it occupies well under 1%
of the variance of a learned encoder; it does not appear in the ambient metric
at the note level; and it becomes visible either by pooling over seconds of
audio or by projecting onto a fraction of a percent of the variance. Neither
operation creates it. Both reveal it.
