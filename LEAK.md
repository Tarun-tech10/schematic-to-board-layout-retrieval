# The candidate pools leak the answer — and we are not using it

**Summary.** Score every candidate with a *constant* — using no image data at all — then run
Sinkhorn over the bipartite pool-membership graph. On the 4700 official training pools that
recovers the true layout at rank 1 for **37.8% of queries** (chance: 5%), a chance-corrected
MRR of **0.318**. For scale, our actual image-based model was at 0.175 when this was found.

Reproduce with `python leakcheck.py`.

```
constant scores, no normalisation      raw 0.1799 corr 0.0000
constant + sinkhorn tau=  1.0 it=150   raw 0.4406 corr 0.3179
```

## What it is not

**Not complexity matching.** The organisers' complexity-matched sampler is clean. The true
layout's rank inside its own pool, ordered by complexity, averages 9.52 where uniform is 9.5,
and the histogram over the 20 ranks is flat. A scorer of `-|complexity - pool median|`, which
uses no query information, earns corrected MRR 0.005. (`leak2.py`)

**Not raw candidate frequency.** Usage is tightly balanced: every train layout appears in
19–22 pools, mean exactly 20.0; every test layout in 19–21, mean exactly 20.00. The brief's
claim that "candidate frequency carries no signal about relevance" is true as stated.

**Not pool position.** Order is randomised, as the brief says.

## What it is

The joint structure of *which layout sits in which pool*. The sampler's usage-balancing
correlates a query's distractor set with its own answer strongly enough that a doubly-stochastic
scaling of the 0/1 membership matrix concentrates mass on the true pairs.

The effect belongs to this particular sampler, not to usage-balanced complexity-matched
sampling in general. We wrote our own sampler with the same pool size, the same usage balance
and a matching complexity spread (0.93 against their 0.92, `calib.py`), and the identical
attack on it yields **5.4% top-1, corrected MRR 0.016** — nothing. (`leak3.py`)

Test pools come from the same sampler and show the same usage signature, so it very probably
transfers, though the test graph is far denser relative to its universe (each layout in 20 of
1000 pools, against 20 of 4700 on train) and the effect may be weaker there.

## Why we are not using it

The brief's first prohibition: *"The pairing must be determined from the circuit's own visual
structure."* This determines it from nothing but the shape of the candidate lists. It would
also collapse the moment the sampler is reseeded, so it measures the pool construction rather
than the task.

**Please do not add it back.** If you want the matching constraint, use the clean version below.

## The clean version

The brief states that every test layout is the true match of exactly one query. That is a
property of the *data*, not of the sampler, and it can be used without touching pool
membership: build the dense query x layout score matrix from model scores alone, Sinkhorn
that, and read the pool entries out of it (`dense.py`).

It is leak-free by construction — a constant dense matrix is symmetric under permutation, so
it must score exactly chance, and it does:

```
base                       raw 0.2465 corr 0.0812
dense sinkhorn tau=0.20    raw 0.2515 corr 0.0873
LEAK CHECK constant input  raw 0.1799 corr 0.0000
```

Worth about +0.007 on the scalar scorer. Modest, honest, and it survives a reseeded sampler.

## Consequence for validation

Any experiment that touches pool structure has to be measured on pools we built ourselves
(`valpool.py`), never on the official ones. Scores from the base model are per-(query,
candidate) and pool-independent, so official pools remain the right yardstick for those — and
they are the better estimate of test difficulty, because our own sampler matches distractors
on exactly the board-size and pad-count axes our features use, which makes it adversarially
hard in a way the real one is not.
