# L1'-a / L1'-b results  (2026-08-29, codebook weights only, no audio)

Statistics defined in `PREREG.md` §5. `null` = matched-shape Gaussian codebook.

## L1'-a — index order carries no geometry. Premise confirmed.

Spearman rho between |i-j| and ||e_i - e_j|| is +0.014 / -0.001 / +0.014 on the
first level of encodec_32k / dac_44k / encodec_24k, and mean adjacent-index
distance over mean pairwise distance is 0.998 / 1.005 / 1.003. Code indices are
an arbitrary permutation, exactly as k-means init implies. Reported to document
the premise, not as a result.

Exception worth recording: encodec_24k levels 22-31 show rho = +0.08..+0.16.
Those are the deep, barely-used levels; the residual there is small and codes
sit near their initialisation, so index order partially survives. It is an
artifact of disuse, not organisation.

## L1'-b — 2D-embeddability collapses with RVQ depth

| codec | level | PR | trust2d | trust2d null | verdict |
|---|---|---|---|---|---|
| encodec_32k (MusicGen) | 0 | 9.9 / 128 | **0.877** | 0.579 | real low-dim structure |
| encodec_32k | 1 | 72.6 | 0.715 | 0.573 | weak |
| encodec_32k | 2-3 | ~78 | 0.64 | 0.58 | ~chance |
| encodec_24k | 0 | **3.9** / 128 | **0.872** | 0.607 | real low-dim structure |
| encodec_24k | 4+ | 27-69 | 0.64 -> 0.54 | ~0.60 | at or below chance |
| dac_44k | all | 7.3-8.0 / 8 | 0.80-0.87 | **0.89-0.90** | **below its own null** |

Three things follow.

1. **The coarse RVQ level is genuinely low-dimensional.** encodec_24k level 0
   has participation ratio 3.9 out of 128 ambient dimensions, and a free 2D
   layout preserves neighbourhoods far above null (0.872 vs 0.607). A 2D grid is
   a cheap ask *here*.
2. **Deep levels are not 2D-embeddable at all.** By level 4-6 trustworthiness is
   at or below the Gaussian null. Forcing those onto a grid will cost real
   reconstruction quality. This quantifies the L2 risk before L2 is run.
3. **DAC's apparent 2D-embeddability is dimensional, not organisational.** Its
   codes are near-constant-norm (norm std 0.06 at level 4): they lie on a shell
   in an 8-D lookup space and are isotropic on it. trust2d sits *below* the
   matched null at every level. Without the null this would have read as a
   positive; it is not one.

## Estimator note — TwoNN is unreliable here, do not quote it

The reported intrinsic dimensions are wrong in a knowable direction and are kept
only for the record. DAC returns ID 28-36 in an 8-D ambient space, which is
impossible. Cause is not duplicate codes (checked: 0.000 of codes sit within 1%
of the median NN distance). It is that TwoNN assumes locally uniform density,
and these codebooks violate it in two ways: DAC's codes sit on a near-constant-
norm shell, and EnCodec's norms are strongly heteroscedastic (mean 35.8, std
18.0 at encodec_32k L0). Participation ratio and trustworthiness are used
instead; neither assumes uniform density.

## What is still open

L1'-a/b are geometry only. They say nothing about whether codes carry *pitch*.
The PREREG decision (outcome A / B / C) needs the NSynth probe, L1'-c.
