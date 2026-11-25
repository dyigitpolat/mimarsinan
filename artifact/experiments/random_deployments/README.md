# Randomized Test of Cost-Only Parameters

## What the Paper Reports

Section IV-A counts the cost-only parameters our proofs cover and how many of them the randomized tests draw (keys `provenrows`, `drawnrows`, `estimaterows`, `undrawnrows`).
Section IV-A also states that Assumption A fails on none of the drawn deployments (`assumptionaviolated`, `screenaccepted`).
Section V-B reports the randomized test of Theorem 1 with random cost-only parameters (`screencex`, `screenaccepted`, `screenbound`, `boundlevel`), the conditions switched off one at a time (Fig. 7(b) and Table IV: `armdiffcount`, `armdiffrange`, `armhfiveskew`, `armrsix`, `hypsimrange`, `hypcodeoff`, `hypsimonly`, `hypnotrun`) and the measurements on Proposition 2 (`proptwostreamed`, `proptwowindowed`).
Two entries come from an earlier version, as the paper says: the differing deployments that changed a classification decision (`decisionarms`) and the static check entries of Table IV (`staticfailspassed`).

`artifact/numbers/index.md` gives each value with its record and field.

## Setup

- **Toolchain.** The patched release, whose records carry the label 644b4c3bc5d7 in their folder names (`artifact/VERSIONS.md` gives its commit). The two entries from an earlier version come from development trees that are not on a public branch (`artifact/VERSIONS.md`).
- **Programs.** Random programs drawn by the toolchain's own generator, with no trained network and no dataset. The field `coverage` of `report.json` lists what was drawn: six semantics (`coverage/semantics`), windows T from 1 to 8 (`coverage/timesteps`), the network motifs (`coverage/motifs`) and every drawn parameter path (`coverage/drawn_paths`).
- **Arms.** The field `arms` of `report.json` holds 15 arms. The arms `on` and `on_scheduled` keep every condition, and their accepted deployments together are those of `hypothesis_on/admitted` (`hypothesis_on_scope` names the two arms). Each other arm switches one condition off or changes one setting of the test, and its own `admitted` field gives its count of accepted deployments.
- **Long windows.** The arm `on_long_window` draws synchronized deployments at windows longer than declared (`hypothesis_on_beyond_declared_window_scope` of `report.json`).
- **Model level.** Conditions that cannot be switched off in the code are switched off in a separate simulator that shares no code with the toolchain. Its runs are `screen_1000_100.json`, equal to the field `model_level` of `report.json`. The field `deterministic/invocation` gives the arms and the number of programs drawn with every condition on (`n_on`) and with one condition off (`n_off`).

## How It Was Run

1. A launch script ran the driver `scripts/random_deployments_px.py`. It is not published because it holds settings of our machines. It called the driver's `screen` command three times: once for the arms that keep every condition (`on`, `on_scheduled`, `on_values_classes`, `on_long_window`), once for the arms that switch one condition off, each until it showed differing deployments (option `--until-teeth`), and once for the arms `on_sign_split`, `on_sign_split_scheduled` and `on_ttfs_offload`. It then called the driver's `report` command, which merges the runs with the separate simulator's runs (option `--model-level`) into `report.json`.
2. The driver calls the toolchain's own test code at the paper version, `scripts/random_deployments.py` and the package `src/mimarsinan/certification/randomized/` (generator, runner, arms). It builds each program as the toolchain's mapping builds it, through `scripts/window_rule.py`, and `scripts/error_site.py` records where an error without a named rejection arose.
3. `scripts/cost_only_live.py` reads the toolchain's registry of declared parameters and writes the registry record `reg/live.json`. The toolchain tests `tests/unit/certification/randomized/test_cost_draws.py` and `tests/unit/certification/randomized/test_hypothesis_table.py` pin that registry.
4. `scripts/pm_screen_px.py` draws position multiplexing on random programs that carry a shared weight bank, which the generator does not draw, and writes `w50/summary.jsonl`, one line per program family.
5. A short script read `report.json` and wrote the condition counts with their inputs and arithmetic into `derived_values.json`. It is not published because it holds internal working notes. Each count's inputs and arithmetic are in the record itself.
6. The separate simulator is `scripts/x3a_golden/` in the toolchain repository at the paper version.
7. Earlier version: `scripts/earlier_version/w16_build.py` built toolchain trees with one condition switched off, and `scripts/earlier_version/w16_runs.py` ran them. A tabulation script, not published because it holds internal working notes, wrote the table of results behind the two earlier entries.

## Records

Paths are places in the records zip. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w12/screen/report.json` | the test's report: arms, coverage, outcomes, bounds, model level runs | `screenaccepted`, `screencex`, `screenbound`, `assumptionaviolated`, `armdiffcount`, `armdiffrange`, `armhfiveskew`, `armrsix`, `hypsimrange`, `proptwostreamed`, `proptwowindowed`, `drawnrows`, `undrawnrows` |
| `records/exactness_block/644b4c3bc5d7_r3/derived/derived_values.json` | condition counts with their inputs and arithmetic | `hypcodeoff`, `hypsimonly`, `hypnotrun` |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/reg/live.json` | the cost-only parameters, one row each with its part and the test that draws it | `provenrows`, `drawnrows`, `estimaterows`, `undrawnrows` |
| `records/frozen/exactness_block_t1_20261125T031754/shakedown/phase_b_rc4a/SD-08/cost_only/live_rc4a.json` | the same registry from an earlier reading | `provenrows` |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w50/summary.jsonl` | position multiplexing test, one line per program family | `drawnrows`, `undrawnrows` |
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w50/rows_verdict.json` | outcome counts of that test by class | none |
| `records/10_wave1/g6_exactness_contract/x3a_golden_runs/screen_1000_100.json` | runs of the separate simulator | `hypsimrange`, through the report |
| `records/excerpts/defect_block.md` | word for word lines of the earlier version's defect record | `decisionarms`, `staticfailspassed` |

## How to Read the Records

- `report.json`, field `hypothesis_on`: `admitted` counts accepted deployments, `counterexamples` those differing from the single-pass program, `upper_bound_95` is the one-sided 95% upper bound on the differing share, `instances` counts all drawn deployments and `outcomes` their classes (`refused_unnamed` counts rejections without a stated reason).
- `report.json`, field `denominators`: `switched_off_with_paired_controls`, `model_level_only` and `per_hypothesis/<condition>/code_teeth` give the conditions switched off and their counts. `arms/<arm>/counterexamples` and `arms/<arm>/admitted` give each arm's counts.
- `report.json`, field `network_relation/per_semantics/<semantics>`: `differ` and `instances` compare the single-pass program with the network N (Proposition 2).
- `reg/live.json`: `cost_only_set_size` is the number of cost-only parameters, `partition_sizes` splits them by part, and each row's `screen` field names the test that draws it.
- `derived_values.json`: each key under `ex` has `value`, `inputs` (each with `file`, `sha256`, `field` and `value`) and `arithmetic`, which reads "(internal note withheld)" on some keys that no printed key of this experiment reads. An input's `file` names a place under `records/exactness_block/644b4c3bc5d7_r3/`, and the frozen copy of the same folder named above holds `report.json` with the same sha256.
- `defect_block.md`: each block quotes the original lines with their line numbers.

## Not Included

The launch script and the run logs are not published because they hold settings of our machines. The script that derived the condition counts, the tabulation script of the earlier version and the earlier version's defect record are not published because they hold internal working notes. The excerpt above quotes the lines the paper reads. The authors can be asked for any of these.
