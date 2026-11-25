# Weight Load in the Simulator Comparison

## What the Paper Reports

The paragraph on simulator costs in Section V-D states that CIMFlow's simulator counts latency as one round of reading
on-chip memory, while our estimate counts the weight load through a configuration port, which is at least a stated
share of the estimated latency at the stated points and core counts. Fig. 9(c) therefore also models the weight store
as CIMFlow's on-chip memory.

Printed keys (in `artifact/numbers/index.json`): `latloadshare`, `latloadpoints` and `latloadcores`.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Toolchain.** The patched release, named in each run record by its digest, beginning `644b4c3bc5d7`, in `head`,
  and by the release it was made on in `base_commit`. `artifact/VERSIONS.md` maps these names to the toolchain
  repository.
- **Networks and pairings.** ResNet-18 and ResNet-50, each on two crossbar pairings, `cimflow` and `cimflow512`
  (`file` of `tc_points.tsv`).
- **Budgets and batches.** 9 and 36 cores for the `cimflow` pairing and 144 and 576 for the `cimflow512` pairing
  (`budget` of `tc_points.tsv`), at the batch sizes of the field `K`.
- **Arguments.** The full argument list of each run, among them the core geometry, the weight and input bits and the
  chip description, is kept in `argv` of each run record and in the first line of each run's output.

## How It Was Run

Our estimator ran once per network and pairing at these budgets and batch sizes, on the patched release, and wrote one
line per point to the run's output and one run record. A second pass collected every point into `tc_points.tsv`,
with the weight load's time, the estimated latency and their ratio. The estimator script and its driver are not
published, since they are kept with comparison data that stays out of the public copy.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/runs/cimflow_resnet18_nonblind.jsonl` | the estimate per point, ResNet-18 on the `cimflow` pairing | `latloadshare`, `latloadpoints`, `latloadcores` |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/runs/cimflow_resnet50_nonblind.jsonl` | the same for ResNet-50 | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/runs/cimflow512_resnet18_nonblind.jsonl` | ResNet-18 on the `cimflow512` pairing | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/runs/cimflow512_resnet50_nonblind.jsonl` | ResNet-50 on the `cimflow512` pairing | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/tc_points.tsv` | one line per point: passes, bytes, compute and load times, load share | `latloadshare`, `latloadpoints`, `latloadcores` |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/run_cimflow_resnet18_nonblind.json` | the run record of ResNet-18 on the `cimflow` pairing | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/run_cimflow_resnet50_nonblind.json` | the run record of ResNet-50 on the `cimflow` pairing | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/run_cimflow512_resnet18_nonblind.json` | the run record of ResNet-18 on the `cimflow512` pairing | none directly |
| `records/frozen/latency_diag_t1_20261123T125524/tiers/fidelity/latency_diag/record_644b4c3bc5d7/run_cimflow512_resnet50_nonblind.json` | the run record of ResNet-50 on the `cimflow512` pairing | none directly |

## How to Read the Records

In `tc_points.tsv`, `prog_nom_s` is the weight load's time, `compute_nom_s` the compute time, `L_tc_nom_s` the
estimated latency and `prog_share` the load's share of it, per point named by `file`, `budget` and `K`. A run record
gives `argv`, `head`, `base_commit`, `rc` (the exit code) and `wall_s`.

## Not Included

The estimator script, its driver and the run logs are not in the public copy, since they are kept with comparison
data that stays out of it. CIMFlow is other authors' software, available from its publishers. The authors of this
paper can be asked for the withheld scripts.
