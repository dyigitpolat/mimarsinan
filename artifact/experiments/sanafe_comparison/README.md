# Comparison on the SANA-FE Simulator

## What the Paper Reports

Section V-B and Table IV report synchronized TTFS run on the SANA-FE simulator, outside Fig. 7(a)'s tests, where it differs from the golden model on some single-pass programs because our SANA-FE export groups cores by latency and so breaks R6's producer order (key `sanafettfs`).

`artifact/numbers/index.md` gives the value with its record and field.

## Setup

- **Toolchain.** The patched release with its SANA-FE back end (`src/mimarsinan/chip_simulation/sanafe/` in the toolchain). The first line of the record gives the code state that ran: `head` (the patched release's label, beginning 644b4c3bc5d7) and `base` (the release). `artifact/VERSIONS.md` gives the commits and the SANA-FE version.
- **Programs.** Six single-pass identity programs named in the record (`chain`, `skew3`, `skip`, `shift_probe`, `seed0`, `seed5`), built from the toolchain's test generators, with no dataset.
- **Comparison.** SANA-FE's outputs against the golden model's synchronized TTFS reference, at zero tolerance and at 1e-6 (the two hardware columns of the record).

## How It Was Run

1. `scripts/sync_ttfs_sanafe.py` builds the six programs, runs each on SANA-FE and on the golden model and prints the counts of differing outputs.
2. `scripts/sync_ttfs_detail.py skip` prints, for the program `skip`, each core's latency group and the outputs that differ.
3. Both probes import `scripts/w33_cross_backend.py`, the cross-backend test code that builds the identity programs and their timing facts, and the generators of `tests/unit/mapping/residency/executed/vehicles.py` in the toolchain.

## Records

Paths are places in the records zip. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w33/runs33/probes_rerun.log` | the output of the two probes | `sanafettfs` |

## How to Read the Records

- Each line after `== sync_ttfs_sanafe.py` names one program. `hardware diffs atol0` counts its outputs that differ from the golden model at zero tolerance. A program counts as differing when that count is above zero.
- The block after `== sync_ttfs_detail.py skip` gives each core's latency (`lat`) and group, then the differing neurons with the reference value and the SANA-FE value.

## Not Included

SANA-FE itself is third-party software and is not part of this copy. It is an optional install of the patched release (`artifact/VERSIONS.md`). Nothing else of this experiment is held back.
