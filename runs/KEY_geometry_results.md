# Clip-level key geometry on GiantSteps, 2026-08-29

## Why this exists

The frame-level GS probe (`scripts/06`) asked a 20 ms code which key it
prefers. That was the wrong granularity and the numbers said so:
nMI(code; tonic) came out 0.035-0.107, against 0.42-0.69 for pitch on NSynth. A
circle-of-fifths null computed on a label that weak has no power and is not
evidence of anything. `runs/topo_gs_*.json` is retained but should not be cited.

This is the same question at the granularity key actually lives at: pool each
clip to one embedding, average clips by key into 24 centroids, test the geometry
of those centroids. 23 well-estimated points instead of ~1000 noisy ones.

## Results

24 key centroids (23 populated), 2000-draw permutation null over key labels.

| arm | rho_fifths | z | p | rho_chroma | z | p | rho_mode | z | p |
|---|---|---|---|---|---|---|---|---|---|
| **chroma** (music-theoretic baseline) | **+0.451** | +8.90 | <0.001 | +0.001 | +0.01 | 0.973 | +0.070 | +1.47 | 0.123 |
| **cqt** (input floor) | -0.010 | -0.24 | 0.814 | -0.013 | -0.31 | 0.731 | +0.354 | +8.42 | <0.001 |
| mert_L4 | **+0.141** | +4.01 | <0.001 | -0.044 | -1.29 | 0.172 | +0.282 | +8.16 | <0.001 |
| mert_L16 | +0.120 | +3.21 | 0.006 | -0.038 | -1.02 | 0.283 | +0.329 | +8.84 | <0.001 |
| mert_L24 | +0.074 | +2.03 | 0.038 | -0.024 | -0.66 | 0.493 | +0.304 | +8.45 | <0.001 |

## Reading

**1. The test has power, and the control proves it.** Mean chroma recovers the
circle of fifths cleanly -- rho 0.451, z 8.90 -- and recovers *only* that:
chromatic (semitone) distance is 0.001. That is the Krumhansl signature, and it
means a null elsewhere is informative rather than a failure of method.

**2. log-CQT has no fifths structure at all** (-0.010, p 0.81). Circle-of-fifths
geometry requires octave folding, and an unfolded spectral representation does
not have it. So unlike pitch height, this is *not* something a model could
inherit from the input.

**3. MERT does build a weak circle-of-fifths geometry that its input lacks.**
rho 0.141 at L4 (z 4.01) against the input floor's -0.010. This is the first
positive result in the project for "a training objective creates musical
structure", and it corrects the earlier flat statement that music SSL shows no
chromatic or fifths organisation whatsoever. It does, faintly, at clip level.

**4. But it is about a third of the explicit-chroma effect, and it decays with
depth**: 0.141 -> 0.120 -> 0.074 from L4 to L24. Same direction as the NSynth
finding. Deeper layers discard it.

**5. The mode effect is a confound, not harmony.** rho_mode is ~0.3 and highly
significant in cqt and every MERT layer -- but only 0.070 and *not* significant
in chroma. Major/minor is therefore not separating these centroids through
pitch-class content; it is separating them through timbre and production.
GiantSteps is entirely EDM, and major-key tracks in it evidently sound
different rather than being harmonically distinguished. Any writeup that reports
rho_mode without this control would be reporting a production artifact as a
music-theoretic finding.

## What this does and does not settle

Settled: the strong form of "emergent harmonic topology" -- that chromatic,
octave-equivalent or circle-of-fifths structure is a dominant organising axis
that a topographic objective could amplify -- is not supported. Octave
equivalence is absent on NSynth under a well-powered probe; the fifths effect
here is weak and shrinking with depth.

Not settled: **whether MERT's 0.141 is tonal or sociological.** In EDM, key
correlates with subgenre, and subgenre correlates with instrumentation and
tempo, which MERT certainly encodes. That rho_chroma is ~0 while rho_fifths is
positive argues for the tonal reading -- a subgenre confound has no reason to
follow the circle of fifths specifically -- but it is not proof. Separating them
needs a key-balanced, genre-diverse corpus, or transposed versions of the same
tracks.
