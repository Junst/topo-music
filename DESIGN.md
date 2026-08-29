# topo-music — topographic organisation of music representations

Started 2026-08-29. Scoping conversation is in the session log; this file holds
the parts that must survive it.

The `/scratch2` weekly purge applies. Code goes to a GitHub remote before the
next Sunday; `runs/` and `.venv/` are rebuildable and are not backed up.

---

## 0. Prior work — READ BEFORE WRITING CODE

`stem-set-jepa` lost a project to a collision found after scoping. This section
exists so that does not happen a third time.

| work | what it does | what it leaves open |
|---|---|---|
| **Toiviainen & Krumhansl (1997/2003)** | SOM on key profiles → toroidal tonal map | input is *already tonal* (key profiles), not audio; no encoder, no downstream task |
| **Leman (1995)** | SOM over tonal contexts | same: symbolic/tonal input, pre-deep-learning |
| **SOM-VAE (Fortuin et al., ICLR 2019)** | SOM in a deep discrete bottleneck, time series | not audio, not RVQ, no perceptual/generative payoff |
| **Topographic VAE (Keller & Welling, NeurIPS 2021)** | topographic grid → equivariant capsules from temporal coherence | vision; "group action = translation on the grid" is *theirs*, not ours |
| **Neural Wave Machines (Keller & Welling, ICML 2023)** | follow-up, travelling waves on topographic grids | vision |
| **TDANN (Margalit et al., Neuron 2024)** + ICLR 2025 deep topographic nets | spatial-correlation loss reproduces cortical maps | vision; validated against noisy fMRI, no analytic ground-truth geometry |
| **FSQ (Mentzer et al., ICLR 2024)** | scalar quantisation → codes on a hypercube **lattice** | **a structured codebook already exists.** The lattice axes are arbitrary, not aligned to perceptual factors |

**FSQ is the collision that matters most.** "Structured codebook" is taken. The
delta this project can claim is not *that* there is a geometry but that **the
axes are aligned to musical factors**, and that this buys something measurable.
Any writeup states this in the first page or dies to a one-line review.

### The delta, stated narrowly

Nobody has asked whether the RVQ codebooks that music generation systems
already ship are topographically organised, nor whether *inducing* that
organisation buys (a) codebook health — utilisation, dead codes — or (b)
token-space geometric editing, where moving a token on the grid is a musical
edit and the LM is not retrained.

### Why music and not vision

Music has **analytic ground-truth topology**: circle of fifths, Tonnetz (torus),
Shepard's pitch helix, Chew's spiral array, Krumhansl's key torus. Topographic
vision work can only compare emergent maps to noisy fMRI. This is the argument
that separates the project from "SOM, but modern", and it belongs in the intro.

## 1. What topography has to buy

A topographic constraint costs reconstruction quality. Every such paper must say
what it bought. Three possible payoffs, and the project targets all three:

- **P1 scientific** — the map recapitulates known music geometry. Weak alone.
- **P2 interpretability** — lesion a grid region, only the matched ability
  breaks. Causal, not probing. Impossible in a codebook with arbitrary indices.
- **P3 engineering** — utilisation ↑, dead codes ↓, and geometric token editing.
  Numbers reviewers already accept. P3 is what makes P1 evidence rather than
  decoration.

## 2. Experiment ladder

| | question | gate |
|---|---|---|
| L0 | does tonal topology fall out of raw CQT alone? | if yes, P1 claim is dead |
| **L1'** | **are shipped RVQ codebooks already topographic?** | **PREREG.md — running now** |
| L2 | does a SOM-neighbourhood quantiser cost acceptable reconstruction? | if not, stop |
| L3 | utilisation / dead codes vs baseline | if no gain, drop P3, fall back to P2 |
| L4 | lesion study; geometric token edit without LM retraining | the paper |

L1' comes first because it tests the project's *premise*, costs a day, and can
kill everything downstream.

## 3. Not using EQ-JEPA

Considered and set aside. EQ-JEPA's augmentation-group machinery would supply a
factorisation signal, but it makes this a representation paper and gives up P3.
If RVQ depth turns out to factorise on its own (L1'-d), the group machinery is
not needed at all.
