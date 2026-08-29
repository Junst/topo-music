# Pre-registration: does tonal geometry depend on temporal scale?

Written 2026-08-29, **before** the folding ablation and the subspace analysis
were run. The clip-level key results and the note-level transposition results
it reacts to are already in (`runs/KEY_geometry_results.md`,
`runs/transpose_*.json`) and are stated here as the motivating facts, not as
predictions.

## 1. The tension being explained

Two measurements on the same MERT-v1-330M encoder disagree about whether it
represents pitch class:

- **Note level (NSynth, real recordings, no DSP pitch shift).** No octave
  equivalence. Delta_strict <= 0 at every layer -- an octave-transposed note is
  *not* more similar than a 9-11 semitone transposition -- while the chromagram
  positive control gives Delta_12 = +0.697. The three octave statistics
  (Delta_mid, Delta_strict, beta_2) disagree in sign for every learned arm,
  which by the rule fixed in advance marks the small positive beta_2 values as
  curve-fitting artifacts rather than octave equivalence.
- **Clip level (GiantSteps, 7035 clips, 24 keys).** Clear circle-of-fifths
  geometry among key centroids: rho_fifths = +0.468 at L12 (z = +8.9),
  +0.455 after partialling out mel-spectral centroid distance, against an
  analytic fifths positive control that recovers rho = +0.977 and a chromagram
  control at rho = +0.475.

Both cannot describe the same geometry. This pre-registration commits to the
hypothesis that resolves them and to what would refute it.

## 2. Hypothesis

Write the representation as `z = z_chroma + z_rest`, with `z_chroma` a
low-dimensional linear subspace carrying pitch-class identity. If
`Var(z_chroma) << Var(z_rest)`, then cosine distance in the ambient space is
governed by `z_rest`, so octave equivalence is invisible at the note level;
averaging thousands of frames within a key cancels much of `z_rest`, so
`z_chroma` dominates the centroid geometry at the clip level.

The hypothesis is therefore **not** "the model has no tonal structure" and
**not** "the model has tonal structure"; it is that tonal structure exists but
is geometrically low-energy, and that temporal pooling is what exposes it.

## 3. Predictions, with thresholds fixed now

Subspaces are orthonormal bases `W` of `d` LDA discriminant directions,
`d` in {1,2,3,4,6,8,11} (11 is the ceiling: any linear map to a 12-class target
has rank <= 11). Statistics are the same ones used for the ambient curve, with
instrument-level bootstrap CIs (2000 resamples) and 2000-permutation nulls.

- **S1 recovery.** A pitch-class LDA fitted on NSynth instruments *disjoint*
  from the evaluation instruments yields, at some `d <= 8`, all three of
  Delta_12 > 0, Delta_strict > 0, beta_2 > 0 with bootstrap CIs excluding 0 --
  where the ambient representation gives Delta_strict <= 0.
- **S2 low energy.** That subspace holds **< 5%** of the total variance of the
  same embeddings (reported on both NSynth notes and GiantSteps clips).
- **S3 transfer -- the decisive cell.** A subspace fitted *only* to predict the
  tonic of GiantSteps tracks, never on notes and never on NSynth, applied
  unchanged to the NSynth transposition curve, gives Delta_strict > 0 with CI
  excluding 0. S1 alone is weak evidence: it fits and evaluates on the same
  dataset and an objective that is octave-invariant by construction, so a
  positive S1 with a negative S3 will be reported as "recoverable within
  dataset, does not transfer", not as support for the hypothesis.
- **S4 mechanism.** Removing the subspace, `z - W W^T z`, drops the clip-level
  rho_fifths to **at most half** its ambient value on the held-out GiantSteps
  split.

## 4. Controls that can void the predictions

- **C1 random subspaces.** Ten random orthonormal `d`-dimensional subspaces per
  `d`. Projection changes the metric whatever the directions are. If random
  subspaces also produce Delta_strict > 0 at the same `d`, S1 is a projection
  artifact and is void regardless of its own CI.
- **C2 absolute-pitch LDA.** A subspace fitted to absolute pitch rather than
  pitch class must not produce octave equivalence exceeding the pitch-class
  subspace at matched `d`. If it does, the objective is not what is doing the
  work.
- **Self-test.** The pipeline was validated before use on a synthetic cache
  with a planted low-energy pitch-class circle: it recovered Delta_12 = +0.007,
  Delta_strict = +0.012, beta_2 = +0.026 where the ambient metric gave ~0 and
  random subspaces gave negative values. So a null on real data is a null of
  the model, not of the method.

## 5. Kill criterion

If S1 fails at every `d <= 11` while C1 also fails, the masked-subspace
hypothesis is refuted. The honest conclusion then is that no linear subspace of
this family carries note-level octave equivalence, and the clip-level fifths
geometry must be produced by pooling statistics rather than by a chroma
subspace. That result is reported as the finding, not rescued.

## 6. Folding ablation, pre-registered separately

The observation that `chroma` shows fifths geometry (rho = +0.475) while `cqt`
does not (rho = -0.035) was initially written up as "octave folding is the one
component that differs". That is not established: librosa's `chroma_cqt` also
drops the dB scaling and normalises each frame. So the 2x2 is run with one
shared CQT front end:

| | dB (ref=max) | per-frame L-inf |
|---|---|---|
| **84 bins, unfolded** | `cqt` (have: rho = -0.035) | `cqt_norm` |
| **12 bins, octave-summed** | `cqt_fold` | `chroma` (have: rho = +0.475) |

- **F1.** If folding is what matters, `cqt_fold` >> `cqt` and `chroma` >>
  `cqt_norm`.
- **F2.** If normalisation is what matters, `cqt_norm` >> `cqt` while
  `cqt_fold` ~ `cqt`. In that case the paper says normalisation, and the
  octave-folding sentence is removed.
- If both matter, the wording stays at "octave folding is the principal
  representational difference relevant to our hypothesis" and both cells are
  reported.
