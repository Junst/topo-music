# E2 (spectral control) and E3 (codec vs CQT vs MERT), 2026-08-29

All arms quantised identically: k-means K=1024 over frames from the same 4096
NSynth notes, so the codec/CQT/MERT comparison is not confounded by whether the
representation is discrete. Codec arms use their own shipped codebooks. Metrics
are unchanged from L1'.

## E3 — which topology emerges under which objective

Row = representation, at the level carrying most pitch nMI.

| arm | nMI(pitch) | Moran I (pitch) | z | rho_abs | rho_chroma | rho_chroma\|pitch | rho_fifths | rho_fifths\|pitch |
|---|---|---|---|---|---|---|---|---|
| **cqt** (input floor) | **0.691** | **0.760** | 65.1 | 0.465 | 0.092 | 0.098 | 0.003 | 0.005 |
| mert_L0 | 0.475 | 0.679 | 61.7 | 0.334 | 0.089 | 0.080 | 0.050 | 0.038 |
| mert_L4 | 0.578 | 0.805 | 69.8 | 0.499 | 0.107 | 0.104 | 0.050 | 0.054 |
| mert_L8 | 0.556 | 0.777 | 63.7 | 0.482 | 0.102 | 0.102 | 0.044 | 0.042 |
| mert_L12 | 0.545 | 0.803 | 70.0 | 0.581 | 0.077 | 0.073 | 0.024 | 0.028 |
| mert_L16 | 0.537 | 0.789 | 70.8 | **0.668** | 0.076 | 0.080 | 0.007 | 0.013 |
| mert_L20 | 0.470 | 0.752 | 63.0 | 0.656 | 0.043 | 0.039 | 0.011 | 0.014 |
| mert_L24 | 0.420 | 0.759 | 63.0 | 0.550 | **0.024** | 0.016 | 0.008 | 0.012 |
| encodec_32k | 0.378 | 0.659 | 67.1 | 0.312 | 0.025 | 0.018 | 0.046 | 0.041 |
| encodec_24k | 0.339 | 0.398 | 44.0 | 0.041 | 0.075 | 0.074 | -0.008 | -0.008 |
| dac_44k | 0.165 | 0.291 | 19.9 | 0.078 | 0.040 | 0.036 | 0.018 | 0.015 |

**Music SSL does not produce musical topology.** Chroma and fifths correlations
remained small in every arm, MERT included: rho_chroma <= 0.107 and
rho_fifths <= 0.054 across all eleven. ("Small", not "at noise" -- these
statistics carry no permutation null, unlike the Moran's I column, and the
distinction should survive into any writeup.) The pre-agreed reading was "if
MERT also gives rho_abs >> rho_chroma then idea 3 is close to dead". It does, at
every depth.

**Depth makes it worse, not better.** The ratio rho_abs / rho_chroma rises
monotonically through MERT:

    L0  3.7x    L4  4.7x    L8  4.7x    L12 7.6x
    L16 8.8x    L20 15.2x   L24 22.7x

Training does not convert acoustic topology into musical topology. It sharpens
the acoustic one (rho_abs 0.334 -> 0.668) while eroding the little chroma
structure the input had (0.089 -> 0.024).

**The input already carries the topography.** log-CQT is the *most*
pitch-organised arm of all eleven by nMI and Moran's I -- 0.691 and 0.760, above
every model. This is the L0 experiment the ladder called for, and it forecloses
the claim the project was built on: with 0.760 already present in the input, a
network reaching 0.805 (mert_L4) did not invent the phenomenon.

    Deep music representations inherit a strong pitch topology already present
    in spectral input representations.

**But MERT is not merely inheriting it.** Its rho_abs reaches 0.668 at L16
against log-CQT's 0.465, so the network is doing something -- just not the
something that was hoped for:

    MERT transforms an inherited acoustic topology by selectively amplifying
    pitch-height organisation.

## E2 — no, it is not "just spectral similarity"

Partial Spearman of code distance against pitch distance, controlling per-code
mean log-mel distance.

| arm | rho_pitch | rho_spec | **rho_pitch \| spec** | rho_spec \| pitch |
|---|---|---|---|---|
| cqt | 0.465 | 0.720 | 0.319 | 0.670 |
| mert_L12 | 0.581 | 0.427 | **0.542** | 0.359 |
| mert_L16 | 0.668 | 0.444 | **0.618** | 0.323 |
| mert_L20 | 0.656 | 0.372 | **0.617** | 0.251 |
| mert_L24 | 0.550 | 0.363 | 0.501 | 0.261 |
| encodec_32k | 0.312 | 0.352 | 0.185 | 0.250 |
| encodec_24k (L0) | 0.162 | — | -0.010 | — |
| dac_44k | 0.078 | 0.341 | **-0.054** | 0.336 |

An earlier draft of this project's writeup said the observed topography "is
explained by spectral similarity alone, and that is not a new fact". **That was
wrong**, and the control that would have caught it had not been run.

- **MERT: not coarse-spectral.** Controlling log-mel barely touches it -- 0.668
  to 0.618 at L16. The defensible statement is exactly this and no more:
  *MERT's pitch-height organisation cannot be explained by coarse log-mel
  spectral similarity alone.* It does **not** license "MERT learns abstract
  pitch": harmonic spacing, resolved harmonics, spectral envelope interactions
  and F0 periodicity are all acoustic cues that survive this control.
- **EnCodec: partly.** 0.312 to 0.185 -- roughly 40% of the effect is spectral,
  the rest is not.
- **DAC: entirely.** 0.078 to -0.054. Its weak pitch effect is spectral position
  and nothing else.
- **cqt: not interpretable.** log-mel and log-CQT are near-collinear; the control
  removes the representation. Reported for completeness only.

## Limits that must travel with these numbers

- **Octave equivalence is properly tested here and is absent.** Single notes are
  sufficient to ask whether C3 and C4 land near each other. They do not.
- **The Tonnetz / circle-of-fifths claim is only weakly tested.** Those are
  relations among *chords and keys*, and NSynth is isolated single notes. The
  fifths statistic here asks only whether codes preferring pitch classes a fifth
  apart sit near each other. A proper test needs a polyphonic or key-labelled
  probe -- GiantSteps (`marble/GS`) is on disk for exactly this.
- **log-mel is a coarse spectral descriptor** (64 bins). "Controls for spectral
  similarity" means "controls for 64-bin log-mel distance", not for all spectral
  information. The E2 conclusion is stated at that strength.
- NSynth is 16 kHz; every timbre number is biased by the missing band above
  8 kHz (PREREG §4).


## The spectrum this actually reveals

Ordering the arms by how much of their pitch geometry survives the spectral
control turns the three representation families into a clean progression:

| representation | rho_pitch \| spec |
|---|---|
| DAC | -0.054 |
| EnCodec | 0.185 |
| MERT L16 | 0.618 |

The training objective determines not just *how much* pitch structure a
representation has but *what kind*. DAC's is spectral position under another
name; MERT's is not.

So the question the project should have been asking was never "does musical
topology emerge" -- answered, no -- but:

    What topology actually emerges across audio representation objectives?

with CQT -> codec -> music SSL as the axis, and spectral / pitch-height /
chroma / timbre organisation as the things measured along it. The one-line
summary of everything above:

    Music representations are topographic, but not musically topographic.
