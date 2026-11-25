# Per-Deployment Check

## What the Paper Reports

Section IV-B states that each Section V-C deployment built at several core budgets passes the static check of Algorithm 1 at every budget, on the patched release's LIF deployments too.
It reports the static check's time relative to compilation (key `staticwall`), the tiled IR equal to the untiled IR on every configuration tried (`irtiling`), and the census of the two rounding multipliers of each core (`lmpcensus`).

`artifact/numbers/index.md` gives each value with its record and field.

## Setup

- **Toolchain.** The patched release, whose records carry the label 644b4c3bc5d7 in their folder names (`artifact/VERSIONS.md` gives its commit). The static check is the toolchain's own, `src/mimarsinan/certification/program_check/check.py` at the paper version (function `check_deployment`).
- **Templates and fixtures.** Every template of `templates/tier_0/` and every deployment fixture of `tests/fixtures/deployment_configs/` in the toolchain, at random initialization, each checked against its IR at unbounded geometry. The first table of `w15/summary.md` counts the template rows and the fixture rows, and `rows.jsonl` holds one line per configuration with `ir_source` random_init.
- **Random programs and unit programs.** The last two tables of `w15/summary.md` cover random programs of the base family at several core counts and a list of unit programs, among them ResNet-18 from a trained checkpoint and ViT-Lite-7/4 at random weights.
- **Deployed programs.** The programs of Section V-C's deployments at their core budgets, one record each (the folder `w15/` named below), each with its network, training form, core budget and core geometry in `label`.
- **Census.** The deployed programs of Section V-C's rows, one line each in `census_paper.jsonl`.

## How It Was Run

1. A batch script listed the jobs and ran them in parallel. It is not published because it holds settings of our machines. Each job ran `scripts/w15e6_programs_px.py` with one mode: `configs` for one template or fixture, `fuzz` for a range of random seeds, and `lenet`, `mux`, `w03`, `lanes_r18` and `vit` for the unit programs. The script delegates the templates, fixtures and random programs to `scripts/w15e1_programs_px.py` and imports `scripts/window_rule.py` and a probe module, which is not published because it holds settings of our machines.
2. `scripts/summarize_w15e6_px.py` reads the rows of all jobs and writes `summary.json` and `summary.md`.
3. `scripts/deployments/w15_recheck.py` reruns the static check, read only, on the stored run of each deployed program and writes one record per program. Its arguments are the run folder, the row label and the output file. `scripts/deployments/read_w15.py` prints a table of these records with one tally line.
4. `scripts/multipliers/ulmp_path_census.py` reads the IR of each deployed program and counts the cores whose rounding uses two multipliers.

## Records

Paths are places in the records zip. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w15/summary.md` | verdicts per set and one row per configuration with compile and check times | `staticwall` |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w15/summary.json` | the same summary as data | none |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/e1/w15/configs/rows.jsonl` | one line per configuration with the tiled IR comparison | `irtiling` |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/lmp/census_paper.jsonl` | one line per deployed program with its requantizing cores and its cores with two multipliers | `lmpcensus` |
| `records/frozen/event_census_static_check_t1_20261124T164942/e2e/event_census_static_check/fw15t_644b4c3bc5d7/w15/<row>.json` | the static check of one deployed program, one file per program | the sentence on Section V-C's deployments |

## How to Read the Records

- `summary.md`: `staticwall` is the median of check time over compile time across the configuration rows of the second table.
- `rows.jsonl`: `statuses/ir-tiling` (and `ir_tiling`) is `established` where the tiled IR equals the IR at unbounded geometry, and `ir_tiling_violations` counts the neurons where it does not.
- `census_paper.jsonl`: `law_cores` counts the requantizing cores of a program and `multi_path_cores` those with two multipliers. The printed key sums both over the lines.
- `w15/<row>.json`: `label` names the row. `pipeline` holds the check the toolchain ran when it built the program, and `rerun` the same check run again on the stored program. In each, `theorem1`, `proposition2`, `statuses/<condition>` and `failures` give the verdicts. `established` is the rerun's verdict. The field `pipeline` is established, with no failure, on every program. On the rerun, three untuned LIF programs give Proposition 2 as conditional, and three fine-tuned LIF programs stopped before the rerun because their stored runs could not be loaded (field `error`).

## Not Included

The batch script, the probe module, the per job rows and the run logs are not published because they hold settings of our machines. The stored runs of the deployed programs and the trained checkpoint of the ResNet-18 unit program are not published because of their size and because trained weights stay out of this copy. The authors can be asked for any of these.
