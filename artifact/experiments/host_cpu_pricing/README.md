# Host CPU Pricing

## What the Paper Reports

Section VI-D: a Jetson-class host CPU's energy reorders no chips for any program and corner, but with 8 energy budgets per program and chip spanning its energies (`hostcapsper`), it moves 171 of 376 budgets from met at every corner to corner-dependent (`hostboundcellspub`).

## Setup

- Host: the NVIDIA Jetson Orin Nano 8 GB module in its 15 W mode, every row of its description published or derived and none measured (the host block, which is not in the zip, see below).
- Points and records: the exploration's reference points, priced with and without the host block (`run/out/points_pub.jsonl`), and the deployment records priced the same way (`run/out/records_pub.jsonl`).
- Energy budgets: 8 log-spaced budgets per program and chip over the ends of both energies (`scripts/cl0h_host_study.py`, constant `CAPS_PER_TARGET` and function `cap_grid`).
- Toolchain: the patched release. The run's folder names it by a digest of its source, `644b4c3bc5d7`. `artifact/VERSIONS.md` gives its commit.

Here `run/` stands for `records/tiers/fidelity/ucl0h3/runs/644b4c3bc5d7/run/`.

## How It Was Run

The study is one script, `scripts/cl0h_host_study.py`, with four commands, run in this order on the patched release.

1. `scripts/config_list.py` lists the deployment configurations that the reference points name.
2. `cl0h_host_study.py block` writes the host block (`block_pub.json`) from the toolchain's host description.
3. `cl0h_host_study.py points` prices every reference point through the cost harness `scripts/cost_identity_w1.py`, with and without the host (`points_pub.jsonl`).
4. `cl0h_host_study.py records` prices every deployment record the same way (`records_pub.jsonl`).
5. `cl0h_host_study.py verdicts` orders the chips per corner and counts the energy budgets whose verdict the host moves (`verdicts_pub.json`).
6. A keys script, not published, wrote the run's keys, and `scripts/add_keys_aud7.py` derives `hostcapsper` from the study script.

Each command ran in its own process. The launcher that started them is not published, as it holds settings of our machines.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/tiers/fidelity/ucl0h3/runs/644b4c3bc5d7/run/out/verdicts_pub.json` | chip order and energy-budget verdicts with and without the host | hostboundcellspub |
| `records/tiers/fidelity/ucl0h3/runs/644b4c3bc5d7/run/out/points_pub.jsonl` | every priced reference point | |
| `records/tiers/fidelity/ucl0h3/runs/644b4c3bc5d7/run/out/records_pub.jsonl` | every priced deployment record | |
| `records/tiers/paper_v6/final/audit_fix3/keys/derived_values.json` | the derived budget count with its inputs | hostcapsper |

## How to Read the Records

In `verdicts_pub.json`, `candidate` covers the reference points and `record` the deployment records.
`candidate/energy_cap/host_bound` counts the budgets whose verdict the host moves from met at every corner to corner-dependent, out of `candidate/energy_cap/cells`.
Each line of `points_pub.jsonl` and `records_pub.jsonl` gives a point's energy as a band with the host (`energy`) and without it (`without_host`), and the host's share.
In `derived_values.json`, `decl/hostcapsper` gives the value, its inputs (each with `file`, `sha256`, `field` and `value`) and the arithmetic. Its first input quotes the line of `CAPS_PER_TARGET` by its number in our working copy of the script, which carried comments. The published script, `scripts/cl0h_host_study.py`, has no comments, so the line sits higher in it. Its other two inputs quote the keys of another run of the same study, a file that is not in the zip, for the wording of the budget rule. The derived record keeps their sha256 and values.
The printed count reads the run under `runs/644b4c3bc5d7`.

## Not Included

The record of the run, its declaration, its launcher, the keys script, both keys files, the host block (`block_pub.json`) and the copy of the reference points that the study prices are not published, as they hold settings of our machines and internal working notes.
The plain list of configuration files that the points command read is left out, as it holds only paths of our working tree.
The authors can be asked for them.
