# Multi-Pass Execution at Equal Budgets

## What the Paper Reports

Section VI-C, Table VII and Fig. 14 report whether multi-pass execution fits a more accurate network into equal area
and energy budgets: the selections, costs and gains in four budget scenarios at the nominal point of the system cost
model, the 48 further energy-budget scenarios and the 288 latency-budget scenarios, the limits of the gains and the
sensitivity range. The five-seed paired tests on test images of the batch-64, batch-16 and loose batch-1 pairs come
from the read of the test split, whose records are in the zip under
`records/frozen/gate_and_test_read_t1_20261109T230030/` and `records/frozen/gate_and_test_read_t1_20261124T231409/`.
This folder holds the cost side, the pool and the pairs on the development split.

Printed keys read from the records below include `cellsgain`, `cellsgainzs`, `cellsother`, `cellsgainbox`,
`boxpoints`, the keys beginning `pz`, `zs`, `th`, `lat`, `fall`, `tile`, `chip` and `temp`, `jointllatdown`,
`launchrecorddown`, `batchk`, `batchh`, `batchone`, `fopool`, `seeds`, `seedsword`, `boundlevel` and `pzfiveronel`.
`artifact/numbers/index.json` lists each with its record and field, and the values derived from them (for example
`pzenergyratiok`, `capareak` and `latgain`) with their derivation records.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Pool.** 48 CIFAR-100 ResNets over three depths, five widths and six per-stage weight-bit vectors (`members` and
  `architectures` of `records/06_build/headline_v46/headline_rules.json`), each trained by its training code with
  fixed-step quantizers (`weight_quantizer_class/declared`) on the CIFAR-100 training batch (`dataset`). The quoted
  lines of the pool definition that `fopool` reads are in `records/excerpts/headline_check_rules.md`.
- **Development split.** 5,000 images held out of the CIFAR-100 training batch, 50 per class, drawn with a fixed seed
  (`split` of `headline_rules.json`, the list itself in `records/06_build/screen_v42/val_split_e01.json`). The
  training recipe is in `recipe`.
- **Selections.** Seed 0's development-split accuracy, from `measured_table` of
  `records/06_build/headline_v46/headline_result.json`.
- **Scenarios.** The four budget scenarios are named K64 (batch 64), H10.1 (batch 16), R1' (loose batch 1) and R1L
  (tight batch 1). Their declared cells are quoted in `records/excerpts/headline_check_rules.md`.
- **Pairs.** Five seeds, 1 to 5, read on the development split (`records/excerpts/headline_pairs_result.md`). The
  tight batch-1 pair was registered after test accuracies were known, and its record gives both reads (`test/read`,
  `test/counts`, `development/read` and `development/counts` of
  `records/06_build/headline_v46_pairs/r3_nominal/b/pairs_r3b_record.json`).
- **System cost model.** Each reading of the re-read names its setting in `meta` and the values of the cost model's
  parameters at that setting in `point` (for example `point/p_tile`, `point/p_global` and `point/store` of
  `records/tiers/paper_v6/fix5/study_build/reread/out/read/P0.json`, the nominal point).

## How It Was Run

1. **Pool training.** `scripts/pool/make_lock.py` fixes the pool before training, a training script trains one member
   (with `scripts/pool/headline_models.py`, `scripts/pool/screen_models.py`,
   `scripts/pool/screen_quant.py`, `scripts/pool/lane_quant.py`, `scripts/pool/stage_bits.py`,
   `scripts/pool/screen_data.py` and a shared module), `scripts/pool/price_measured.py` prices the measured members
   and `scripts/pool/collect.py` writes `headline_result.json`. `scripts/pool/reuse.py` checks reused runs. The
   training script, the shared module and the script that wrote the pool's rules are not published, since they hold
   settings of our machines.
2. **Pairs.** The five-seed pairs were fixed before training by a rules script and a lock script, which are not
   published, since they hold internal working notes. `scripts/pairs/collect_pairs.py` collects the pairs, with the
   shared parts in a module not published, since it holds settings of our machines, and
   `scripts/pairs/pairs_stats.py` and `scripts/pairs/primary_order.py` give the one-sided paired t test with Holm adjustment.
   `scripts/pairs/tight_batch1/select_r3.py`, `scripts/pairs/tight_batch1/make_lock_r3.py` and
   `scripts/pairs/tight_batch1/derive_r3.py` select, fix and derive the tight batch-1 pair.
3. **System cost model.** The rule modules under `scripts/cost_model/` (`closure_rule.py`, `noc_census.py`,
   `noc_rule.py`, `period_rule.py`, `plan_layout.py`, `state_rule.py`, `static_rule.py` and `sysmem_rule.py`). Their
   common module and the rule of the store's state are not published, since they hold internal working notes.
4. **Re-read of the scenarios.** `scripts/reread/p0_plan7.py` writes the list of settings, `scripts/reread/r_whole.py`
   reads every named setting (`out/read/*.json`), `scripts/reread/g_grid.py` reads the dense grid and the box of
   sensitivity settings, `scripts/reread/t_thresh.py` bisects the thresholds of first change (`out/thresh/*.json`),
   `scripts/reread/d1_fall.py` explains the one fall, `scripts/reread/g0_gate7.py` checks the setting of record,
   `scripts/reread/p1_probe.py` probes the cost model, an analysis script writes the analysis and
   `scripts/reread/k_keys.py` writes `keys.json`. `scripts/reread/rcommon7.py` and `scripts/reread/gridlib.py` are
   their shared code. The analysis script and their shared library are not published, since they hold internal
   working notes.
5. **Per-combination re-read of the four scenarios.** A script that reads the combinations, not published since it
   holds internal working notes, `scripts/reread_cells/c2_keys.py`, `scripts/reread_cells/c3_tables.py` and
   `scripts/reread_cells/c4_brackets.py` write `keys_costliest.json` and `brackets_costliest.json`.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/tiers/paper_v6/fix5/study_build/reread/keys.json` | every key of the re-read with its value, display and source field | the `pz`, `zs`, `th`, `lat`, `fall`, `tile`, `chip` and `temp` keys |
| `records/tiers/paper_v6/fix5/study_build/reread/out/read/P0.json` | the reading at the nominal point | `cellsgain` |
| `records/tiers/paper_v6/fix5/study_build/reread/out/read/zero_ruled.json` | the reading without static power | `cellsgainzs` |
| `records/tiers/paper_v6/fix5/study_build/reread/out/read/box_*.json` | the readings of the box of sensitivity settings | `cellsgainbox`, `boxpoints` |
| `records/tiers/paper_v6/fix5/study_build/reread/out/read/*.json` | the readings of the other named settings | through `keys.json` |
| `records/tiers/paper_v6/fix5/study_build/reread/out/thresh/*.json` | thresholds of first change | the `th` keys |
| `records/tiers/paper_v6/fix5/study_build/reread/out/d1_fall.json` | the cause of the one fall | none directly |
| `records/tiers/paper_v6/fix5/study_build/reread/out/g0_gate7.json` | the check of the setting of record | none directly |
| `records/tiers/paper_v6/fix5/study_build/reread_cells/out/keys_costliest.json` | the four scenarios over the cost model's combinations | `pzpassesronelc`, `pztimeronelc`, `pzmemronelc`, `pzgainronelc`, `thtileronelcd`, `thchipronelc` |
| `records/tiers/paper_v6/fix5/study_build/reread_cells/out/brackets_costliest.json` | the brackets of those combinations | none directly |
| `records/06_build/headline_v46/headline_rules.json` | the pool, split and recipe | none directly |
| `records/excerpts/headline_check_rules.md` | the declared cells and members, word for word | `batchk`, `batchh`, `batchone`, `fopool` |
| `records/06_build/headline_v46/headline_result.json` | the measured table of seed 0 | none directly |
| `records/06_build/headline_v46/headline_lock.json` | the pool's lock | none directly |
| `records/06_build/headline_v46/masks.json` | the admissible masks | none directly |
| `records/06_build/screen_v42/val_split_e01.json` | the development split | none directly |
| `records/excerpts/headline_pairs_result.md` | the result of the five-seed pairs, word for word | `seeds`, `seedsword`, `boundlevel` |
| `records/06_build/headline_v46_pairs/r3_nominal/b/pairs_r3b_record.json` | the tight batch-1 pair's record | `pzfiveronel` |
| `records/06_build/headline_v46_pairs/r3_nominal/selections_r3.json` | the selections of the tight pair | none directly |
| `records/06_build/codesign/v47_composed_r7/site_table.json` | the site table of the system cost model | none directly |

## How to Read the Records

Each entry of `keys.json` has `name`, `value`, `display` and `record_field`. The last names the analysis file and the
path inside it that the value was read from. The analysis file is not in the public copy, so `value` and `display`
are the public values, and the readings in `out/read` and `out/thresh` hold the costs behind them. A reading names
its setting in `meta`, its parameter values in `point`, and holds `energy`, `latency` and `fronts` per scenario.

## Not Included

The analysis of the re-read, the parameter file and data tables of the system cost model other than the site table named
above, the plan of the per-combination re-read, the source of the area unit, the full rules and result notes of the pairs (quoted in part in the excerpts
above), some readings of named settings, the per-combination table and the scripts named above as not published are
not in the public copy, since they hold internal working notes or settings of our machines. The trained networks are
not included either. The CIFAR-100 dataset is third-party material, available from its publisher. The authors can be
asked for the withheld records and scripts.
