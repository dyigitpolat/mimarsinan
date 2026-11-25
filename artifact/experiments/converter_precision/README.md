# Converter Precision

## What the Paper Reports

Section VI-D and Fig. 15(a) report a converter-width grid for two MNIST networks: every grid point matches the
declared converter function exactly on sampled inputs, accuracy is unchanged across core budgets as passes grow, and
converter precision itself changes accuracy. Fig. 15(a) plots the accuracy loss against converter bits, relative to
exact converters, with a trained CIFAR-10 ResNet-20 curve over three seeds and a ResNet-18 curve on CIFAR-100. The
figure reads the records below through its data file (see `artifact/figures/`).

Printed keys (in `artifact/numbers/index.json`): `adcgrid` and `spreadbudgets`, both read from the MNIST grid. No
printed key reads the CIFAR-10 arm or the CIFAR-100 sweep.

This folder holds three parts of one experiment: the MNIST grid, the CIFAR-10 arm and the CIFAR-100 sweep.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`, with `*` for every key or
element of a block.

- **Toolchain.** The patched release, named in the records by its digest beginning `644b4c3bc5d7` (the folder names
  `mnist_644b4c3bc5d7`, `records_644b4c3bc5d7` and `points_644b4c3bc5d7`). `artifact/VERSIONS.md` maps these names to
  the toolchain repository.
- **MNIST grid.** LeNet-5 (`mvm_int_readout_lenet5`) and `t0_43v`, each trained for the converter (`tuned`) and not
  (`untrained`) (`members` of `summary.json`). Converter bits 3, 4, 5, 6, 8 and 32 over three core budgets, resident,
  half and eighth (the keys of `members/*/tuned/budget_spread` and `members/*/tuned/adc_spread`). Correct images are
  counted on 1,500 MNIST training images held out from training (`count_split` and `label` of `summary.json`).
- **CIFAR-10 arm.** ResNet-20, seeds 0, 1 and 2, counted on 1,250 training images held out from training
  (`rows/*/seed`, `count_split` and `label` of `cifar10_summary_r2.json`), at the budgets and converter bits of
  `rows/*/budget` and `rows/*/adc_bits`. Each point names its cores, its scheduling and its checks (`cores`,
  `allow_scheduling` and `certificates` of each `point.json`).
- **CIFAR-100 sweep.** The trained ResNet-18 of CIFAR-100 (member `C1`), with its quantized weights as they are and no
  training for the converter, at converter bits 4, 5, 6 and 8 and 32 as the exact reference, on two core budgets, `B0`
  the single-pass deployment and `B1` a smaller budget with scheduling (the keys of `members/C1/readings/counts` and
  `points/*/pass_count` of `collect_r4_t1.json`). Correct images are counted on 5,000 development images drawn from
  the CIFAR-100 training batch, with no test image (`label` and `points/*/images` of `collect_r4_t1.json`).

## How It Was Run

MNIST grid and CIFAR-10 arm:

1. A specification script and `scripts/configs/spec_grid.json` define the grid. The specification script is not
   published, since it holds settings of our machines.
2. `scripts/cl6_snapshot.py` records each trained network as a snapshot of its working directory, cut before the
   converter step of the pipeline (`start_step` of each `snapshot.json`: Readout Law, or Boundary Quantization for
   `t0_43v`).
3. `scripts/cl6_point.py` builds one grid point on a copy of a snapshot at its converter width and core budget, checks
   it at zero tolerance on sampled inputs and counts correct images. `scripts/cl6_sweep.py` runs the grid.
4. `scripts/cl6_irdiff.py` and `scripts/cl6_entry_check.py` check programs across budgets and the grid's entry
   conditions. `scripts/cl6_routeb.py` and `scripts/cl6_common.py` hold shared parts.
5. `scripts/cl6_summary.py` writes `summary.json`.
6. The CIFAR-10 arm trains one ResNet-20 per seed with the configuration `scripts/configs/cl6_cifar10_resnet20.json`,
   runs its points with the same scripts and collects them into `cifar10_summary_r2.json`. The scripts that fixed the
   arm beforehand, trained each seed and collected the arm are not published, since they hold settings of our
   machines.

CIFAR-100 sweep:

1. `scripts/make_lock.py` fixes the sweep beforehand: points, budgets, widths, split and the digests of every script
   below. The scripts here are the ones it pins, published without their comments, so their digests differ.
2. `scripts/w34_points.py` reads the sweep's points and prints each point's command. A runner checks the fixed sweep
   and runs each point once with `scripts/cl6/cl6_routeb.py` and `scripts/cl6/cl6_point.py`, with the shared helpers of
   `scripts/cl6/cl6_common.py`. The runner is not published, since it holds settings of our machines.
3. `scripts/w34_price.py` prices each point under converter descriptions. No printed value reads these prices.
4. A collection script, which is not published, collects the ten points into `collect_r4_t1.json`: counts per budget
   and width, the accuracy loss against 32 bits, and the spread across budgets.

The three scripts under `scripts/cl6/` are byte-identical to the scripts of the same names under `scripts/`.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/mnist_644b4c3bc5d7/summary.json` | MNIST grid points, checks, invariance across budgets | `adcgrid`, `spreadbudgets` |
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/mnist_644b4c3bc5d7/runs/snap/t0_43v/snapshot.json` | the snapshot record of `t0_43v` | none |
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/mnist_644b4c3bc5d7/runs/snap/mvm_int_readout_lenet5/snapshot.json` | the snapshot record of LeNet-5 | none |
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/cifar10/records_644b4c3bc5d7/cifar10_summary_r2.json` | the CIFAR-10 arm's rows and cells | none (Fig. 15(a)) |
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/cifar10/records_644b4c3bc5d7/runs/s<seed>/points/<budget>/adc<bits>/point.json` | one CIFAR-10 point per seed, budget and width | none (Fig. 15(a)) |
| `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/cifar10/records_644b4c3bc5d7/runs/s<seed>/runs/snap/cifar10_resnet20_s<seed>/snapshot.json` | the snapshot records of the three trained seeds | none |
| `records/frozen/w34_adc_sweep_t1_20261120T183547/11_wave2/explore/cl6/w34/points_644b4c3bc5d7/collect_r4_t1.json` | CIFAR-100 counts and accuracy loss per budget and converter width | none (Fig. 15(a)) |
| `records/frozen/w34_adc_sweep_t1_20261120T183547/11_wave2/explore/cl6/w34/points_644b4c3bc5d7/<nn>_C1_<budget>_adc<bits>/point.json` | one CIFAR-100 point per budget and width, ten in all | none |

## How to Read the Records

- `summary.json`: `points` counts the grid points, `failures` lists any point that failed its check (empty),
  `invariance/cells_zero_spread` of `invariance/cells` the grid cells whose accuracy does not change across budgets,
  and `members/<network>/<arm>` the spreads per width (`budget_spread`) and per budget (`adc_spread`).
- `cifar10_summary_r2.json`: `rows` gives per seed, budget and width the `count` of correct images `of` the counted
  images and `atol0_certified`, `cells` the mean and spread over seeds, and `spread_across_budgets` the spread at equal
  width. A CIFAR-10 `point.json` gives `member`, `adc_bits`, `pass_count` and its `certificates` with their tolerance.
- `collect_r4_t1.json`: `members/C1/readings/counts/<budget>/<bits>` is the number of correct images of 5,000,
  `members/C1/readings/loss_pp/<budget>/<bits>` the loss in percentage points against 32 bits, and
  `members/C1/identity` the check that both budgets give the same count at 32 bits. `points/<id>` gives each point's
  `count`, `images` and `pass_count`.

## Not Included

The MNIST run record, the declarations that fixed the CIFAR-10 arm and the CIFAR-100 sweep beforehand, and the scripts named above as not published are not in the
public copy, since they hold settings of our machines or internal working notes. The MNIST, CIFAR-10 and CIFAR-100
datasets are third-party material, available from their publishers. The authors can be asked for the withheld records
and scripts.
