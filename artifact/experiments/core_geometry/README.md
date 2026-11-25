# Core Geometry Study

## What the Paper Reports

The co-design paragraph of Section VI-C and Section VIII report our confirmatory study on the release: of the
pre-registered tests, a stated number confirm that co-designed core geometry speeds up one network over chips sized
for a reference network, depending on row reading and cell-bit energy.

Printed keys (in `artifact/numbers/index.json`): `geoconfconfirmed`, `geoconftests`, `geoconfnet` and `geoconfref`.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Toolchain.** The release, as Section VI-C states. `artifact/VERSIONS.md` maps this name to the toolchain
  repository.
- **Plan.** The pre-registered plan fixed the seed, the networks (the 48 networks of the equal-budget pool), the batch
  sizes, the budget scenarios, the grid of core geometries (rows, cell columns and cell bits), the reference network
  and the accuracy source (the measured table of the equal-budget pool on the development split, see
  `artifact/experiments/equal_budget_scenarios/`). `records/13_geometry_study/plan_r5.json` is published in part. It
  holds only the statement of the reference network (`declared/reference_anchor_identity`), which `geoconfref` reads.
- **Tests.** The plan's family of tests, each with its metric, its target network and its target accuracy.

## How It Was Run

1. A plan builder wrote the plan before the run. It is not published, since it holds settings of our machines, and
   its output is the plan, published in part.
2. `scripts/geo_harness_r5.py` builds and prices one design. An assembly script, not published for the same reason,
   collects the priced designs into the sweep. `scripts/pd_summarize_r5.py` summarizes the physical design
   points.
3. `scripts/fits_r2.py` fits the cost terms and `scripts/geo_readings.py` holds the readings the analysis shares.
4. `scripts/geo_confirm_r5.py` runs the confirmatory tests, `scripts/score_r5.py` scores them and
   `scripts/geo_confirm_analysis_r5.py` analyses them.
5. `scripts/derive_fix.py` derives the label counts of record (`derived.json`).

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/13_geometry_study/confirm_analysis_fix/derived.json` | label counts, confirmed networks, bands | `geoconfconfirmed`, `geoconftests`, `geoconfnet` |
| `records/13_geometry_study/plan_r5.json` | the pre-registered plan | `geoconfref` |

## How to Read the Records

In `derived.json`, `label_counts` counts the tests by label (confirmed, not confirmed, not determinable,
unconfirmable), `confirmed_target_members` names the networks of
the confirmed tests, and `six` gives the setting, scenario and target of each confirmed test.

## Not Included

The run's declarations, the rest of the plan, the plan builder, the assembly script, the priced sweep and the routed tiles of the physical
design are not in the public copy, since they hold settings of our machines. The authors can be asked for them.
