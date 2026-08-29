# What actually makes musical geometry visible, 2026-08-30

Four sweeps, run in this order, each one prompted by the previous one failing.

## 1. Temporal aggregation is not the answer

`scripts/14`. Window length swept from one frame to the full 20 s clip, with
one window drawn per clip at every length so the number of clips behind each of
the 24 key centroids is identical throughout. rho_fifths:

| arm | 13-32 ms | 0.5 s | 2 s | 20 s | ratio |
|---|---|---|---|---|---|
| chroma | +0.460 | +0.452 | +0.469 | +0.475 | **1.03x** |
| cqt | -0.023 | -0.022 | -0.042 | -0.035 | - |
| mert_L4 | +0.276 | +0.411 | +0.419 | +0.445 | 1.61x |
| mert_L12 | +0.276 | +0.422 | +0.439 | +0.468 | 1.70x |
| mert_L24 | +0.108 | +0.242 | +0.268 | +0.294 | 2.72x |

A single MERT frame already carries 59% of the 20 s value, and the chromagram
is flat to within 3%. There is no knee and no threshold. The note/clip gap is
not a story about how much audio is pooled.

## 2. Averaging over examples is the answer -- for GiantSteps

`scripts/16`. The clip-level measurement averages twice, and only one of those
was being varied. Holding the window at the full clip and sweeping the number
of clips per key centroid:

| arm | n=1 | n=8 | n=32 | n=128 | all (~293) | ratio |
|---|---|---|---|---|---|---|
| chroma | +0.077 | +0.205 | +0.366 | +0.454 | +0.475 | 6.2x |
| cqt | +0.026 | -0.030 | -0.030 | -0.036 | -0.035 | - |
| mert_L4 | +0.035 | +0.115 | +0.287 | +0.422 | +0.445 | 12.7x |
| mert_L12 | +0.035 | +0.111 | +0.292 | +0.438 | +0.468 | **13.4x** |
| mert_L24 | +0.016 | +0.034 | +0.161 | +0.264 | +0.294 | 18.4x |

13x against 1.7x. And at n = 1 -- one clip per key, the condition the
note-level test is actually in -- mert_L12 sits at +0.035 +- 0.042, which is
nothing. The clip-level geometry disappears when the averaging does.

## 3. But averaging does *not* rescue NSynth. The unified account is wrong.

`scripts/17`. If low between-item SNR were the whole story, then averaging n
notes of the same pitch into a pitch centroid should make octave equivalence
appear on NSynth the way it appears on GiantSteps. It does not:

| arm | Delta_strict, n=1 | n=8 | all | chroma reference |
|---|---|---|---|---|
| mert_L4 | +0.0016 | +0.0075 | +0.0081 | |
| mert_L12 | -0.0039 | +0.0054 | +0.0064 | |
| mert_L16 | -0.0074 | -0.0001 | +0.0005 | |
| mert_L24 | -0.0043 | -0.0010 | -0.0011 | |
| cqt | +0.0008 | +0.0022 | +0.0021 | |
| chroma | **+0.5660** | +0.5980 | +0.6155 | already there at n=1 |

Everything saturates by n = 4 and saturates at ~zero. MERT L24 stays negative.
The chromagram, by contrast, has octave equivalence in *single notes* -- it
needs no averaging at all.

So the mechanism is not one thing. Averaging removes nuisance variance that is
**independent of the class**: on GiantSteps, production, instrumentation and
subgenre vary across tracks of the same key and cancel in the mean. It cannot
remove nuisance that is a **function of the class**: on NSynth, register and
spectral envelope covary with pitch by construction, so every pitch centroid
keeps them no matter how many notes go into it. Averaging over instruments
cancels the instrument, never the octave.

## 4. What removes class-coupled nuisance is a projection

`scripts/13`. The `key_lda` subspace is fitted only on GiantSteps track tonics,
so the directions it keeps must be the ones that predict a pitch-class property
-- which forces them to be pitch-height-invariant. Applied to NSynth notes
with no averaging at all, at 0.11-0.24% of MERT's variance:

| arm | ambient Delta_strict | projected (d=4) |
|---|---|---|
| mert_L4 | -0.001 | +0.304* |
| mert_L12 | -0.005 | +0.223* |
| mert_L16 | -0.009 | +0.086* |
| mert_L24 | -0.003 | +0.154* |
| cqt | +0.004 | +0.466* |

Matched random subspaces give ~0. So the octave-equivalent structure is present
in the ambient representation; it is neither created by the projection nor
recoverable by averaging.

## The account these four support

Musical geometry is hidden in two different ways and they need two different
operations to undo.

- Nuisance variance **independent of the class** hides it from any estimate
  built on few examples. Averaging over examples removes it. This is the whole
  of the GiantSteps key effect, and it is why a 20 ms window works as well as a
  20 s one: what is missing at n = 1 is not audio, it is samples.
- Nuisance variance **coupled to the class** hides it from the ambient metric
  permanently. No amount of averaging touches it, because it does not average
  away. A low-dimensional linear projection removes it, and one fitted on an
  entirely different dataset and task does so at a fraction of a percent of the
  variance.

The note-versus-clip framing was wrong, and so was the temporal-scale framing
that replaced it. Both were reading a property of the *estimator* and a
property of the *metric* as a property of the representation.
