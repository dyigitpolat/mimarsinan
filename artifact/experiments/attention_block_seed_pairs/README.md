# Seed Pairs of the Attention Block

## What the Paper Reports

Table VIII (Results for the ViT-Lite-7/4 Attention Block, Section VI-D) reports a non-inferiority test of ViT-Lite-7/4
with attention on dynamic-weight cores against attention on the host, with a stated margin. This folder holds the
five-seed pairs of the training code on the development split. The row on test images comes from the read of the test
split, whose records are in the zip under `records/frozen/gate_and_test_read_t1_20261109T230030/`.

Printed keys (in `artifact/numbers/index.json`): `vitmargin`, read from `records/excerpts/vit_pairs_result.md`.

## Setup

- **Pairs.** Each pair compares, at one seed, ViT-Lite-7/4 with attention on dynamic-weight cores and with attention
  on the host. The pairs run over seeds 1 to 5 and are counted on the 5,000 images of the development split (named
  E01 in the records), with the margin of the test in percentage points (`records/excerpts/vit_pairs_result.md`, which
  quotes the line the paper reads).
- **Development split.** The same split as the equal-budget pool (`records/06_build/screen_v42/val_split_e01.json`,
  see `artifact/experiments/equal_budget_scenarios/`).

## How It Was Run

1. The pairs, their seeds, their margin and their test were fixed before training by a rules script and a lock script.
   Both are not published, since they hold internal working notes.
2. `scripts/build_harness.py` builds each seed's training harness from the fixed one. The module that holds the
   shared parts is not published, since it holds settings of our machines.
3. `scripts/collect_pairs.py` collects the results, and `scripts/pairs_stats.py` and `scripts/primary_order.py` give
   the one-sided paired statistics against the margin.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/excerpts/vit_pairs_result.md` | the line of the result note that states the margin, word for word | `vitmargin` |
| `records/06_build/screen_v42/val_split_e01.json` | the development split | none directly |

## How to Read the Records

The excerpt gives the line number of the original note and its text: the seeds, the images, the mean difference
between the two arms, the one-sided 95% lower bound and the one-sided p value against the margin.

## Not Included

The pairs' rules and lock, the shared module, the full result note and its record of the five-seed pairs, the training
harness of each seed and the trained networks are not in the public copy, since they hold internal working notes or
settings of our machines. The CIFAR-100
dataset is third-party material, available from its publisher. The authors can be asked for the withheld records and
scripts.
