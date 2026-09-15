# Schematic to Board Layout Retrieval — solution

## 1. What the score actually pays for

Twenty candidates, one right answer, scored by chance-corrected MRR. Chance is
`(1/20)·Σ 1/r = 0.17989`, and the correction maps that to zero. So raw MRR 0.25 — the true
layout landing around rank 3 on average — is worth 0.085 corrected; raw 0.35 is worth 0.21.
Ties are credited at the expected reciprocal rank over the tied block, so a flat submission
and a flat submission plus noise score identically. There is nothing to gain from
tie-breaking noise and nothing to lose from honest ties.

The evaluation harness here reimplements that exactly, including the tie rule, and confirms
a constant submission scores 0.0000.

## 2. What the data turned out to look like

Both views are KiCad renders, and two facts about the renderer do most of the work.

**Both views have a fixed, known physical scale.** 4542 of 5700 schematics are exactly
1123×794 px, which is A4 landscape at 96 dpi; the rest are A3 (1587×1123), US Letter
(1056×816), A5, A2 — all at 96 dpi. Layouts render at 192 dpi: the nearest-neighbour distance
histogram over detected through-hole pads spikes hard at 19.2 px, which is 0.1 inch, the
standard header pitch, with harmonics at 9.6 and 38.4.

So a layout image's width and height *are* the board outline in millimetres — the median
board is about 70×60 mm — and pad sizes are real millimetres. On the schematic side, KiCad's
2.54 mm pin pitch is 9.6 px, so a symbol's height on the sheet reads out its pin count.

Consequence for the model: every image is rasterised at a fixed **px per millimetre**, never
resized to a fixed box. A 0603 pad is then the same size in every layout, and the network can
learn absolute size rather than relative size. Boards too large for the canvas are shrunk to
fit and the shrink factor is handed to the model as a feature.

**Colour is layer.** Both renders use the stock KiCad palette, verified by isolating each
colour and looking at the mask. Schematics: background `(245,244,239)`, wires `(98,188,96)`,
symbol outline and pins `(177,98,96)`, symbol body fill `(255,255,194)`, labels `(98,98,212)`.
Layouts: `F.Cu (200,52,52)`, `B.Cu (77,127,196)`, through-hole pad and via `(79,172,227)`,
SMD pad `(132,116,220)`, `F.SilkS (242,237,161)`, `B.SilkS (232,178,167)`, mask layers, and
`Edge.Cuts (208,210,205)`, over a white or black background.

So the image is not fed as RGB. It is unmixed into per-layer coverage maps. Each pixel is
projected onto the best-fitting palette direction relative to the background,
`alpha = clip(⟨x−bg, p−bg⟩ / ‖p−bg‖², 0, 1)`, which keeps the mass of an antialiased
one-pixel wire instead of rounding it to background; a nearest-colour assignment loses those
entirely. Colours are quantised to 15 bits and looked up in a precomputed table, so the whole
decomposition is one fancy-index per image. Anything the palette does not explain goes to a
catch-all channel. The corpus of 11 400 images preprocesses in about 200 seconds.

## 3. Structure the raster cannot hold

At 2 px/mm an 0603 pad is under two pixels across, so the raster physically cannot represent
a pad count. Pads are therefore counted at **native** resolution and summarised as scalars:
pads are labelled, grouped into footprints by a 0.45 mm dilation, and turned into a histogram
of pads-per-footprint and a histogram of pad areas in mm². The schematic gets the mirror —
symbol bodies and their areas, the pin stubs in an annulus around each body, and the count of
small two-terminal symbols.

The single best-transferring descriptor found is a connectivity count: components of the wire
mask on the schematic against components of the copper mask on the layout, plus the number of
wire/symbol contact blobs, which is the count of wired pins. Log-log correlation with the
layout side reaches 0.34, against 0.27 for the next best and 0.21 for raw ink area.

These descriptors are worth 0.0814 → 0.1093 corrected on their own through a ridge, they ride
into both towers alongside the raster, and — more usefully — each tower is trained to predict
the *other* view's descriptors. That turns one training pair from a single contrastive bit
into roughly fifty regression targets, which matters a great deal at 4700 pairs.

## 4. Method

Two towers, separate weights, because the views share no visual vocabulary. Each is a
six-stage double-conv CNN over the layer-coverage raster (schematic 288×384×5 at 1.293 px/mm,
layout 224×224×8 at 2.0 px/mm), concatenating global average and max pooling with an MLP over
the descriptors, into a 512-d trunk that feeds both an L2-normalised 256-d embedding and the
auxiliary descriptor head. Score is the dot product.

Losses: symmetric InfoNCE over in-batch pairs; a listwise softmax over sampled members of the
query's own official candidate pool, which trains directly against the organisers' hard
negatives; and the auxiliary cross-view descriptor regression.

Augmentation: layouts get the full dihedral group plus scale and translation jitter, because
a board is the same board rotated or mirrored. Schematics get only mild jitter — a schematic
is read left to right and flipping it is not a transformation the data ever contains.

Inference averages independently seeded models, the layout tower with dihedral test-time
augmentation, and blends in the descriptor ridge.

## 5. The candidate pools leak, and this solution does not use it

Scoring every candidate with a **constant** — no image data at all — and then running Sinkhorn
over the bipartite pool-membership graph recovers the true layout at rank 1 for **37.8%** of
training queries against a 5% chance rate: corrected MRR **0.318**, roughly double what the
actual model achieves.

It is not the complexity matching, which is clean: the true layout's rank inside its own pool
by complexity averages 9.52 where uniform is 9.5, and a scorer of `-|complexity − pool median|`
earns 0.005. It is not raw candidate frequency, which is tightly balanced at 19–22 uses per
layout. It is the joint structure of which layout sits in which pool, and it belongs to this
particular sampler: an independently written usage-balanced, complexity-matched sampler with a
matching spread (0.93 against their 0.92) yields 5.4% top-1 under the identical attack.

This solution does not use it. The brief's first prohibition is that the pairing must be
determined from the circuit's own visual structure, and this determines it from the shape of
the candidate lists alone. It would also vanish if the sampler were reseeded, so it measures
the pool construction rather than the task.

What *is* used is the honest form of the same constraint. The brief states that every test
layout is the true match of exactly one query — a property of the data, not of the sampler. A
Sinkhorn normalisation of the **dense** query×layout score matrix, built from model scores
only and never referencing pool membership, exploits that. It is leak-free by construction: a
constant dense matrix is symmetric under permutation, so it must score exactly chance, and it
does. Worth about +0.007.

## 6. Validation

700 training queries are held out **with their layouts**: a held-out layout is never a
positive during training and is excluded from the pool-negative sampler, so it is never seen
at all. Base scores are per-(query, candidate) and pool-independent, so they are measured on
the official pools. Anything that touches pool structure is measured instead on pools built by
the independent sampler, which was verified not to carry a membership signal.

Worth recording: the independent sampler produces pools that are *harder* than the official
ones for this model, because it matches distractors on exactly the board-size and pad-count
axes the features use, which the real sampler does not.

## 7. Results

| scorer | raw MRR | corrected |
|---|---|---|
| constant / random | 0.1799 | 0.0000 |
| ridge on 29 global scalars | 0.2466 | 0.0814 |
| ridge on scalars + structural descriptors | 0.2695 | 0.1093 |
| two-tower CNN, in-batch InfoNCE only | 0.3236 | 0.1752 |
| full model | TBD | TBD |

## 8. Things that were tried and did not work

- **Per-body pin-stub attribution.** Counting pin stubs in an annulus around each symbol body
  gives a median of 3 pins per body, which is far too low — pin-number text and neighbouring
  symbols corrupt the ring. The global wire/symbol contact count is a better pin proxy and is
  what the descriptors use.
- **Linear double-centering** of the score matrix, the τ→∞ limit of Sinkhorn, is worse than
  Sinkhorn proper (0.097 against 0.137 on the same scores).
- **Training past ~30 epochs** without the auxiliary and pool losses: train loss falls from
  3.87 to 0.57 while validation peaks at epoch 30 and then decays.

## 9. Determinism

Fixed execution plan: no wall-clock guards, no throughput probes, no fallback branches. Seeds
are set for torch, numpy and python; thread counts and `OMP_DYNAMIC`/`MKL_DYNAMIC` are pinned
before numpy and torch are imported; epoch and model counts are constants.

## 10. Files

- `solution.py` — self-contained; `python3 solution.py <public_dir> <submission_csv>`. Writes a
  valid submission before training begins and overwrites it with the model result.
- `submission.csv` — the output of that script on the supplied test set.
- `description.md` — this file.
