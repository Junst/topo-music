# Clip-level key geometry on GiantSteps, 2026-08-29

## Why this exists, and what it replaces

The frame-level GS probe (`scripts/06`) asked a 20 ms code which key it
preferred. Wrong granularity, and the numbers said so: nMI(code; tonic) came out
0.035-0.107 against 0.42-0.69 for pitch on NSynth. A null computed on a label
that weak has no power. `runs/topo_gs_*.json` is retained but should not be
cited.

An earlier smoke run of *this* script, on GS.val only, is also superseded. It
reported rho_fifths = +0.141 for MERT L4. The full run over all three splits
(7035 clips, all 24 keys populated) gives +0.445 at the same layer. Any text
still quoting 0.141 is stale by a factor of three; this file is the current
record.

Method: pool each clip to one embedding, average clips by key into 24 centroids,
test the geometry of those centroids against four reference structures, with a
2000-draw permutation null over key labels and a partial Spearman controlling
mel-spectral centroid distance.

## Results (mean pooling, 7035 clips, 24 keys)

| arm | key24 acc | tonic12 acc | rho_KK (z) | rho_fifths (z) | rho_fifths given spec | rho_chromatic | rho_mode |
|---|---|---|---|---|---|---|---|
| **fifths_analytic** (metric control) | .274 | .352 | +0.606 (+9.6) | **+0.977** (+15.8) | +0.977 | -0.085 | +0.053 |
| **chroma** (music-theoretic control) | .451 | .549 | +0.640 (+11.2) | +0.475 (+8.4) | +0.445 | +0.140 | -0.054 |
| cqt (input floor) | **.492** | **.625** | +0.006 (+0.1) | -0.035 (-0.6) | -0.103 | +0.174 | +0.165 |
| encodec_32k | .404 | .513 | +0.176 (+3.6) | +0.177 (+3.8) | +0.218 | +0.041 | +0.098 |
| mert_L4 | .451 | .537 | +0.607 (+12.0) | +0.445 (+8.8) | +0.432 | -0.025 | +0.216 |
| mert_L12 | .456 | .527 | +0.595 (+11.2) | **+0.468** (+8.9) | +0.455 | +0.024 | +0.203 |
| mert_L16 | .440 | .518 | +0.492 (+9.4) | +0.395 (+7.6) | +0.377 | +0.047 | +0.262 |
| mert_L24 | .418 | .500 | +0.375 (+7.9) | +0.294 (+6.3) | +0.266 | +0.049 | +0.328 |

Chance is .042 for key24 and .083 for tonic12. `mean+std` pooling gives the same
ordering and the same conclusions (chroma +0.522, mert_L12 +0.444); it is in the
JSONs and is not reported separately.

## Reading

**1. The metric works, and now that is demonstrated rather than assumed.** The
analytic control places 12 tonics on a unit circle at theta = 2*pi*(7k mod 12)/12
by construction, and the pipeline recovers rho_fifths = +0.977. An earlier draft
treated the chromagram as the fifths control; that was wrong. A chromagram
hard-codes octave equivalence but nothing in it makes d(C, G) < d(C, F#), so it
is an octave/chroma control only. Both controls are now run, and they answer
different questions.

Note also what the analytic arm implies for reading the other columns:
a representation with *pure* fifths geometry and nothing else already scores
rho_KK = +0.606. So rho_KK and rho_fifths are not independent evidence. That
chroma scores higher on KK (+0.640) than on fifths (+0.475) says it carries
scale-content structure beyond the circle alone.

**2. MERT builds circle-of-fifths geometry, and it is not spectral.** +0.468 at
L12 (z = +8.9), and partialling out mel-spectral centroid distance moves it only
to +0.455. Coarse spectral similarity does not account for it.

**3. It does not exceed the chromagram.** chroma +0.475 vs MERT's best +0.468.
So the finding is not that self-supervision discovers tonal structure absent
from a hand-built pitch-class feature; it is that MERT encodes pitch-class
content well enough to reproduce what a chromagram already gives. Two keys a
fifth apart share six of seven scale degrees, so fifths geometry follows fairly
mechanically from pitch-class overlap. This should be stated in any writeup:
the result is about *content being present*, not about an emergent topology
beyond content.

**4. Decodability and geometry come apart, and CQT is the clean case.** log-CQT
has the *highest* key accuracy of any arm (.492, vs MERT's .456) and exactly
zero fifths geometry (-0.035, p = .52). Key is linearly decodable from it while
its metric carries no tonal relation whatsoever. This is the sharpest form of
the dissociation in the project.

What separates cqt from chroma is not settled by this table. Octave folding is
the difference relevant to the hypothesis, but librosa's `chroma_cqt` also drops
the dB scaling and normalises each frame, so folding is confounded with
normalisation here. The 2x2 ablation registered in `PREREG_SCALES.md` section 6
(`cqt_fold`, `cqt_norm`) separates them; until it returns the claim stays at
"octave folding is the principal representational difference relevant to our
hypothesis".

**5. Across depth, tonal relational geometry weakens while major/minor
separability increases.** rho_KK falls 0.607 -> 0.375 and rho_fifths 0.445 ->
0.294 from L4 to L24, while rho_mode rises 0.216 -> 0.328. Stated as a
correlation, not a mechanism: nothing here shows the later layers discard tonal
relations *in order to* gain categorical discrimination, and key accuracy does
not improve along the way (.451, .456, .440, .418), so it is not a trade that
buys better key decoding either.

**6. The rising mode effect is probably not harmony.** rho_mode is ~0.2-0.33 and
highly significant in every MERT layer and in cqt, but it is -0.054 and not
significant in the chromagram control. Major/minor is therefore not separating
these centroids through pitch-class content. GiantSteps is entirely EDM, where
major-key tracks plausibly differ in production and instrumentation rather than
in harmonic organisation. Any reading of the depth trend as "later layers become
mode-oriented abstraction" has to survive this: on the evidence here, the growing
mode axis is at least as consistent with timbre and production as with tonality.

## What this does and does not settle

Settled: key geometry at the clip level is real, survives a spectral control,
and is absent from the unfolded spectral input. The strong form of "emergent
harmonic topology" -- a dominant chromatic axis a topographic objective could
amplify -- is still unsupported: rho_chromatic is null everywhere, and the
fifths effect does not exceed a chromagram.

Not settled, and now the central question: **how clip-level tonal geometry
coexists with the complete absence of note-level octave equivalence in the same
encoder** (Delta_strict <= 0 for every MERT layer, `runs/transpose_*.json`). The
subspace hypothesis and its kill criterion are pre-registered in
`PREREG_SCALES.md`.

Also not settled: whether the fifths effect is tonal or sociological. In EDM,
key correlates with subgenre. That rho_chromatic stays ~0 while rho_fifths is
large argues for the tonal reading -- a subgenre confound has no reason to
follow the circle of fifths specifically -- but a key-balanced, genre-diverse
corpus, or transposed versions of the same tracks, would settle it.
