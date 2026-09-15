# Schematic â†’ Board-Layout Retrieval

Cross-view retrieval: given a **schematic** image (logical view â€” symbols and nets), pick which of
**20 candidate PCB layout** images (physical view â€” copper, pads, silkscreen) implements the same
design. Metric is chance-corrected **MRR** over the fixed 20-candidate pool
(`chance = (1/20)*sum_{r=1..20} 1/r = 0.17989`, corrected score `max(0,(MRR-chance)/(1-chance))`,
reporting floor 0.02).

This repo is the working code for that task. It is **work in progress** â€” see *Status* at the bottom.

---

## The data

| | count | note |
|---|---|---|
| `public/schematics/<qry_id>.png` | 5700 | 4700 train queries + 1000 test queries, one shared folder |
| `public/train/layouts/<lay_id>.png` | 4700 | train-side candidate universe |
| `public/test/layouts/<lay_id>.png` | 1000 | test-side candidate universe |
| `public/train.csv` | 4700 rows | `id`, `cand01..cand20`, `rel01..rel20` (one `1.0` per row) |
| `public/test.csv` | 1000 rows | `id`, `cand01..cand20` |

Split is disjoint at the level of the *design project*, near-duplicates are removed, and every test
layout is the true match of exactly one test query. So the distractors in a test pool are other
genuine designs, and memorising train pairings is worthless.

## What we worked out about the rendering (the useful part)

Both views are KiCad renders with **fixed, known physical scales** and a **fixed colour palette**.
That is the whole reason this task is tractable.

**1. Schematics render at 96 dpi = 3.7795 px/mm of *sheet*.**
Evidence: 4542/5700 schematics are exactly `1123x794` px = A4 landscape (297x210 mm) at 96 dpi;
530 are `1587x1123` = A3; 209 are `1056x816` = US Letter; `794x559` = A5; `2245x1587` = A2.
Consequence: a symbol's pixel height is its real size on the sheet, and KiCad pin pitch is
2.54 mm = **9.6 px**, so an IC symbol body height is a direct read-out of pin count.

**2. Layouts render at 192 dpi = 7.559 px/mm of *board*.**
Evidence: the nearest-neighbour distance histogram over detected through-hole pads has a dominant
spike at **19.2 px**, which is 0.1 inch = 2.54 mm header pitch, with harmonics at 9.6 and 38.4.
Consequence: layout image width/height *is* the board outline size in mm (median board approx.
70x60 mm), and footprint pad sizes are real millimetres.

**3. Both views use the stock KiCad layer palette, so colour = layer.**
Verified by isolating each colour and eyeballing the mask.

*Schematic:* background `(245,244,239)`, wires `(98,188,96)`, symbol outline + pins `(177,98,96)`,
symbol body fill `(255,255,194)`, labels `(98,98,212)`.

*Layout:* background white or black, `F.Cu (200,52,52)`, `B.Cu (77,127,196)`,
through-hole pad / via `(79,172,227)`, SMD pad `(132,116,220)`, `F.SilkS (242,237,161)`,
`B.SilkS (232,178,167)`, mask layers `(216,100,255)` / `(2,255,238)`, `Edge.Cuts (208,210,205)`.

## Representation

`prep.py` turns every PNG into a **multi-channel coverage raster at a fixed physical scale**:

* Colours are quantised to 15 bits and looked up in a precomputed table, so decomposition is one
  fancy-index per image rather than a per-pixel distance computation.
* Each pixel is *unmixed* against the background â€” `alpha = clip(dot(x-bg, p-bg) / |p-bg|^2, 0, 1)`
  with the best-fitting palette entry â€” so antialiased one-pixel wires keep their mass instead of
  being rounded away to background. Anything no palette entry explains lands in a catch-all channel.
* Each channel is then area-downsampled to a fixed **px-per-mm**, not to a fixed box, so a 0603 pad
  is the same size in every layout. Canvas: schematic `288x384x5` at 1.293 px/mm,
  layout `224x224x8` at 2.0 px/mm, centre-padded; oversized boards are shrunk to fit and the shrink
  factor is passed to the model as a feature.
* Alongside the raster it emits ~29 scalars per image: board/sheet size in mm, ink area per layer in
  mm^2, and connected-component statistics (count / total / median / p90 / max area, count above
  4 mm^2) for pads, vias, silkscreen and symbol bodies.

The whole corpus (11 400 images) preprocesses in **~200 s** on 20 threads.

## Model

`model.py` â€” a two-tower encoder, separate weights per view because the two views share no visual
vocabulary. Each tower is a 6-stage double-conv CNN, then concat(GAP, GMP, MLP(scalars)), then a
256-d L2-normalised embedding. Score is the dot product.

`train.py` â€” symmetric InfoNCE over in-batch pairs, optionally plus a listwise softmax over each
training query's own 20-candidate pool (`--poolw`), which trains directly against the organisers'
hard negatives. Layouts get full D4 plus scale/translate jitter (a board is the same board rotated);
schematics get milder jitter only.

Validation holds out 700 train queries **and their layouts** â€” a held-out layout is never a positive
during training â€” and scores the official 20-candidate pools with the exact tie-aware
expected-reciprocal-rank the leaderboard uses (`ev.py`).

## Results so far

| scorer | raw MRR | corrected |
|---|---|---|
| constant / random (chance) | 0.1799 | 0.0000 |
| ridge on the 29 scalars, cosine score (`probe2.py`) | 0.2468 | **0.0815** |
| ridge, + structural descriptors (pads, footprints, symbols) | 0.2695 | **0.1093** |
| ridge, + net counts and wire/symbol contacts | 0.2818 | **0.1242** |
| two-tower CNN, in-batch InfoNCE only (run a0, best epoch) | 0.3236 | **0.1752** |
| two-tower CNN + aux regression + pool loss | *running* | |

**The candidate pools leak the answer through their own construction — see [LEAK.md](LEAK.md).**
Constant scores plus Sinkhorn over the pool-membership graph reach 37.8% top-1 (corrected MRR
0.318) using no image data whatsoever. We are not using it, and the file explains why, what it
is not (complexity matching and candidate frequency are both clean), and what the honest
version of the matching constraint looks like.

Useful negative result: the organisers' "complexity-matched" distractor pools are only **loosely**
matched â€” pool spread is 0.95x the global spread on every statistic measured. Coarse size and
complexity therefore still carry real signal, which is where the 0.0815 comes from.

## Running it

```
pip install -r requirements.txt
python prep.py  <path-to-public-dir>  cache     # ~200 s, writes ~5.4 GB of uint8 memmaps
python probe2.py                                # scalar-feature baseline
python train.py --epochs 60 --bs 48 --poolw 0 --tag r0
```

`train.py --poolw 1.0 --knn 3` adds a listwise term over sampled members of each query's own
official pool, training directly against the organisers' hard negatives.

**Read [LEAK.md](LEAK.md) before touching anything that uses pool membership.**

## Ground rules (from the brief â€” please keep to these)

The pairing must come from the circuit's own visual structure. Not allowed: external parts lists,
netlists or design-repository indexes; matching images, ids or hashes against any public repo or CAD
dataset; **reading title-block text, project names, authors, dates or silkscreen strings** to match
the two views; id-based or positional hardcoding; tuning against the test relevances; hand-matching
the test set. The palette and scale work above is deliberately text-free â€” everything is measured
from geometry, and at the working resolution (1.3â€“2.0 px/mm) rendered text is not legible.

## Status / next steps

- [x] Verified render scales and palettes; built the fixed-scale layer-coverage representation.
- [x] Evaluation harness with the exact tie-aware MRR, design-disjoint holdout.
- [x] Scalar-feature baseline at 0.0815 corrected.
- [x] Native-resolution structural descriptors (`feats.py`): pads grouped into footprints, a
      histogram of pads-per-footprint and of pad areas, symbol bodies and their pin stubs.
      Worth 0.0814 -> 0.1086 on the scalar scorer; they now ride into both towers too.
- [x] Pool loss made affordable - it samples `--knn` of the 20 candidates, not all 20.
- [x] Audited the pool construction for leakage; found a large one, documented it, declined it.
- [x] Clean matching constraint via dense Sinkhorn (`dense.py`), worth +0.007.
- [x] First CNN run: **0.1752**, best at epoch 30. Past that it overfits hard - train loss
      falls 3.87 -> 0.57 while validation decays to 0.166. 4700 pairs is not many.
- [x] Connectivity descriptors: components of the wire mask against components of the copper
      mask, plus the wire/symbol contact count (= wired pins). Best cross-view correlation
      found, 0.34 log-log, against 0.27 for the next best. Ridge 0.1093 -> **0.1242**.
- [x] `solution.py` written and smoke-tested end to end on a 60-row subset; memmap-backed so
      the 5.4 GB of rasters never has to be resident.
- [ ] **Run b0 in progress**: auxiliary cross-view regression (each tower predicts the other
      view's descriptors, so a pair supervises ~50 targets rather than one contrastive bit)
      plus the pool hard-negative term, dropout 0.2 and decay 3e-4 against the overfitting.
- [ ] Tune the blend weight and Sinkhorn temperature on the holdout (`tune.py`).
- [ ] Final: seed ensemble trained on all 4700 pairs, dihedral TTA, submission.
- [ ] Sweep resolution / px-per-mm â€” the current layout raster may be too coarse to count 0603 pads.
- [ ] Seed ensemble plus D4 test-time augmentation on the layout tower.
- [ ] Blend the CNN score with the scalar-feature scorer.
- [ ] Package as `solution.py` taking `argv[1]`=public dir, `argv[2]`=output CSV, self-contained,
      training from scratch inside the platform's time budget.
