# Throughput of the Pipelining Branch against COMPASS

## What the Paper Reports

The abstract states that a pipelining branch runs 2.31 times COMPASS's reported throughput at batch 16 (toolchain
estimate). Section V-D (Throughput) reports, on COMPASS's ResNet-18 and middle chip size, 256 arrays, at batch 16, that
the branch copies layers into idle arrays and then streams the batch through each layer group, raising the patched
release's throughput as 89, 1,587 and 1,752 inferences per second, where COMPASS's paper reports 758 under its own
model. At batch 1, weight loading takes 90% of the time and limits the branch to 0.57 times COMPASS's rate. Section
VII names the branch as the one that replicates and pipelines layers.

Printed keys (in `artifact/numbers/index.json`): `thrubrfactor`, `thrubatch`, `thrumarrays`, `thrucmpours`,
`thrubrcopies`, `thrubrpipe`, `thrucmpprior`, `thrubatchone`, `thruonewrites` and `thruonefactor`.

## Setup

Fields are named by their path inside `e3_grid.json`, as in `artifact/numbers/index.json`, with `[*]` for every
entry of a block.

- **Toolchain.** Our pipelining branch at commit `92d90e2c` (`heads`), the branch `throughput-replication` of the
  toolchain repository, seven commits above the patched release (`artifact/VERSIONS.md`). The arm without copies is the
  patched release's design priced on the branch, and the grid's identity check compares the branch's outputs with those
  of an earlier commit of the branch (`identity`).
- **Network.** ResNet-18 for ImageNet at an input side of 224 (`check/[*]/net`, value `r18_224`), built with
  random weights from a fixed seed for a pricing by shape that reads no image (`scripts/w0_price.py`).
- **Chip.** The COMPASS-geometry study chip of the toolchain
  (`src/mimarsinan/mapping/platform/targets/study_compass.json` on the pipelining branch) at COMPASS's three chip sizes,
  `S`, `M` and `L` in the grid's keys. The middle size has 256 cores of 256 axons and 64 neurons with 4-bit inputs
  (`check/[*]/platform_keys/cores` and `check/[*]/platform_keys/activation_bits`).
- **Arms.** Arm A is the patched release's pipelined multiplexing. Arm B1 adds layer copies, with each layer group on at
  most a declared number of arrays (`B1/[*]/declare/layer_replication` and `B1/[*]/declare/wavefront_group_cores`).
  Arm B2 adds the batch pipeline, with as many input slots as the batch (`B2/[*]/declare/batch_pipeline_slots`). The
  pass budget is 131,072 (`B1/[*]/declare/max_schedule_passes`).
- **Reference.** COMPASS's reported throughput per chip size and batch, from its paper (`compass`).
- **Barrier time.** The printed values use the chip description's own barrier time (cells `study:shipped|K<batch>`),
  and the grid also prices barriers of 100 ns and 0 (`study:sync100`, `study:sync0`).

## How It Was Run

1. `scripts/s1_jobs.py` writes the list of runs, one line per process: arm A on the middle chip, and arms B1 and B2 at
   the chip sizes, group limits, pass budgets, buffer settings and batches it lists.
2. Arm A runs through `scripts/w0_slots.py`, which loads `scripts/w0_price.py`, our pricing driver for COMPASS's
   network, and arms B1 and B2 through `scripts/e2_price.py`. Each line runs as its own process on the branch's source
   tree. `w0_price.py` imports two driver modules of our pricing of the deployed selections of Section VI-C, the
   pipelined pricing module and a pairing module. Neither is published.
3. `scripts/s2_identity.py` compares arm A and the headline cells of arm B1 with the outputs of the same lines on the
   earlier commit, every field but the wall clock and the commit.
4. `scripts/e2_check.py` builds arm B2's program at batch 16 on the middle chip and records its stages (`check`).
5. `scripts/s3_table.py` reads every output, the identity comparison, the build record and the earlier grid
   `e2_grid.json`, and writes `e3_grid.json`.

The scripts that ran the processes in parallel are not published.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/thru_branch_92d90e2c/e3_grid.json` | the full grid: arms, chips, batches, barrier times, COMPASS's values, the build record and the identity | all ten |
| `records/excerpts/throughput_branch_compass.md` | the lines of the comparison table at batch 1, word for word | `thrubatchone` |
| `records/tiers/paper_locks/thru/e2/e2_grid.json` | the earlier grid that the table builder reads beside the new outputs | none |

## How to Read the Records

- `e3_grid.json` has one block per arm (`A`, `B1`, `B2`). An arm's key names the chip size (`S`, `M`, `L`), the pass
  budget (`b131072`), the group limit in arrays (`g256`), the buffer (`buf0`) and, for B2, the input slots (`s16`). Its
  `cells` are keyed by barrier time and batch, such as `study:shipped|K16`. In block `A` the cells sit directly
  under the arm's key.
- A cell gives `thr` (inferences per second, toolchain estimate), `factor` (`thr` over COMPASS's reported value for the
  same chip size and batch), `us` (the time of one inference in microseconds, split into `barriers`, `compute`,
  `writes` and `residual`), `passes_per_inference` and `reprogram_passes`.
- The share of the time spent writing weights at batch 1 is `us/writes` over the sum of the cell's `us` fields.
- `compass` lists COMPASS's reported values by chip size and batch, `check` the build of arm B2's program at batch 16,
  `identity` the comparison with the earlier commit, and `heads` the commits the grid ran on.

## Not Included

The tables printed from the grid (of which `records/excerpts/throughput_branch_compass.md` quotes the lines the paper
reads), the two driver modules named above, the per-process outputs and logs, and the scripts that ran the processes are
not in the public copy. The authors can be asked for them.
