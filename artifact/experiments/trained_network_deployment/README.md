# Deployment of Trained CIFAR-100 Networks

This folder documents how we deployed the trained CIFAR-100 networks of Section V-C end to end, at the core budgets of Table VI and Fig. 8, and where their records lie in the records zip. The test-split columns of Table VI come from the pre-registered test read, documented in `artifact/experiments/test_read`.

## What the Paper Reports

- Table VI (Deployed Accuracy of Trained Networks), CIFAR-100 rows: the Dev. column (correct images of the development split) and the budget marks (outputs equal to the single-pass deployment's at each marked budget).
- Fig. 8 (scheduled passes against the core budget): the planned pass counts at the declared core budgets and the executed budgets whose output codes equal the single-pass deployment's.
- Section V-C, "Core budgets", "Pass counts" and "Other rows": the budget shares of the single-pass core count $M$, the shared-kernel variant of the co-designed chip, and the converted rows (integer, LIF and fine-tuned LIF).
- Section IV-B, the sampled run's declared input counts.

Printed keys that read these records (`artifact/numbers/index.md` gives each value with its record and field):

| Printed keys | Where | Records |
|---|---|---|
| `rbcountrone`, `rbcountrtwo`, `rbcountrthree`, `rbcountrfour`, `rbcountrfive`, `rbcountrtenone`, `rbcountrtenhalf`, `rbcountdensenet`, `rbcountpyramid`, `rbcountvitlite`, `rbcountvitop` | Table VI, Dev. column | integer rows |
| `rbcheckrone` to `rbcheckvitop` (the same eleven suffixes) | Table VI, budget marks | integer rows and their tables |
| `ptqintcount` | Table VI, converted integer row | integer rows |
| `lifcount`, `lifcheck` | Table VI, LIF row | LIF rows and their tables |
| `liftunedcount`, `liftunedcheck` | Table VI, fine-tuned LIF row | fine-tuned LIF rows |
| `ladderdeclared` | Section V-C, Fig. 8 | data of Fig. 8 |
| `budgethalf`, `budgeteighth` | Table VI header | declared budgets |
| `trcertsamplesmost`, `trcertsamplesfew` | Section IV-B | declared input counts of the sampled run |
| `eimages`, `carryimages`, `ptqweights`, `vittwintol` | Sections IV and V | declared settings that the paper states |

We document five sets of records in this one folder, because a reader meets them as one table and one figure: the integer rows, the budget ladder of Fig. 8, the shared-kernel rows, the LIF rows and the fine-tuned LIF rows.

## Setup

- **Toolchain.** Each row result names the toolchain it ran on in `inputs.toolchain`: `base_commit` (the release), `patch_sha256` (the fixes on top of it) and `commit` (beginning 644b4c3bc5d7, the label of the patched release in the records). `artifact/VERSIONS.md` gives the commits, and it says, as Table VI's note does, how the rows marked a relate to an earlier version.
- **Networks.** The field `member` of a row names its network, and the collected tables give the network's name in `rows[*].label`. C1 to C5 are CIFAR ResNet-18 at width multipliers 1, 2^-1/2, 2^-1, 2^-3/2 and 2^-2. R10w100 and R10w050 are CIFAR ResNet-10 at 1 and 1/2. S1 is DenseNet-BC-100 with growth rate 12, S4 is PyramidNet-110 (alpha 84), and ViT is ViT-Lite-7/4. Each network was trained by its own code outside the toolchain. The row names its checkpoint and the checkpoint's digest (`inputs.checkpoint`, `inputs.checkpoint_sha256`).
- **Row names.** A row is named `<member>.<target>.<route>.<budget>.<geometry>`. The target is the program form: `T-INT` an integer program, `T-LIF` a LIF program, `T-VAL8-INT` ViT-Lite-7/4 with attention on the host CPU, and `T-VAL8-OP` ViT-Lite-7/4 with attention products on dynamic-weight cores. The route is `b` (or `b-op` for the latter) for a network trained with quantization and deployed with unchanged weights, `a` for a float network converted after training, and `at` for the converted LIF network fine-tuned in the toolchain. The geometry is the core's axons by neurons.
- **Chips.** Cores of 256 axons by 256 neurons at budgets B0 to B4, and the co-designed chip's cores at B5 and its variants (256 by 64, or 128 by 64 for ResNet-18 at 2^-3/2 and ResNet-10 at 1/2), as the row name and `packing_declaration.cores_geometry` of each result give them. Weights and activations are 4-bit, and 8-bit for the ViT-Lite-7/4 rows (`packing_declaration.weight_bits` and `deployment.activation_bits` of each result, the LIF rows giving only the weight bits). The LIF rows run 15 time steps of synchronized LIF (`platform_constraints.simulation_steps` and `deployment_parameters.spiking_variant` of their `config.json`).
- **Budgets.** B0 is the single pass (`deployment_parameters.allow_scheduling` false in `config.json`). B1 and B2 use the default plan at $M/2$ and $M/8$ cores, rounded up, B3 the minimum-rewrite plan (`platform_constraints.residency_plan.step` "floor") and B4 the full-rewrite plan ("stateless"), both at $M/2$. B5 is the co-designed chip and B5L its shared-kernel variant. B5i, B5m and B5u are further co-designed variants, each with its own `config.json`. Each result gives its core count in `deployment.declared_count` and the passes of a scheduled row in `record.schedule.pass_count`. The tables give the single-pass core count of each network in `discovery_rows`.
- **Data and seeds.** The development split of 5,000 CIFAR-100 training images held out from training (`inputs.provider.provider_name` `E01_CIFAR100_DataProvider`, its digest in `inputs.provider.split_sha256`, and `K9.images` of each result). Seed 0 (`seed` of each `config.json`). Each row seals the test archive (`inputs.sealed_paths`).
- **Sampled run.** Its declared input count is `samples` of `K4_K5_pipeline.K4_value_parity_gate` in each integer row's result.
- **Budget ladder of Fig. 8.** The planner places each network at $\lceil M/d \rceil$ cores for each $d$ of the declared ladder (`planned.declared_ladder.d` of the figure's data file). The ladder rows execute three of these budgets for ResNet-18, DenseNet-BC and PyramidNet-110 (L4, L16 and L32) and two for ViT-Lite-7/4 on the minimum-rewrite plan (L4f and L8f), as the field `row` of `ladder_rows.jsonl` lists them.

## How It Was Run

1. The pre-registered row list and one configuration template per row were generated from the declarations of members, targets and budgets, with `scripts/driver/templates.py` and `scripts/driver/configs.py`. Each row's own configuration is its `config.json`. The declarations module and the script that built the row list are not published because they hold internal working notes.
2. A row runner ran the rows in dependency order, and a per-row runner ran each one. Both are not published because they hold settings of our machines. Per row, the toolchain's pipeline ran from the checkpoint (`scripts/driver/inventory.py` names each network's builder and checkpoint, and a quantizer map script, not published because it holds internal working notes, writes its quantizer state). Then `scripts/driver/harvest.py` measured the packed program on the development split in shards and merged them, `scripts/driver/harness_reference.py` evaluated the checkpoint again with byte copies of its training code, and `scripts/driver/no_update.py` checked that no weight changed. The row wrote `config.json` and `result.json`.
3. `scripts/driver/collect.py` collects the rows of a set into one table and compares the outputs of each budget row with its single-pass row bit for bit (field `K10` of the table).
4. `scripts/ladder/make_calls.py` and `scripts/ladder/plan_ladder_paper.py` plan every network at the declared ladder with the toolchain's planner, and `scripts/ladder/assemble_paper.py` assembles the planned ladder that the data of Fig. 8 carries. The ladder rows themselves ran through items 2 and 3.
5. The training code of the networks is in `scripts/driver/harness_copy` (its model, quantizer and data files), a copy that `scripts/driver/harness_copy/copies.py` checked byte for byte against the training folders when the rows ran. Its two training scripts are not published because they hold settings of our machines.

Most driver files import the declarations module and a shared module of paths and machine settings, and neither is published. The driver files therefore document the procedure but do not run as published.

The toolchain is not copied here. The planner calls name toolchain code at the paper version: `src/mimarsinan/deployment_record/introspection/scaling_ladder.py`, `src/mimarsinan/mapping/verification/layout_verification_scheduling.py`, `src/mimarsinan/mapping/support/operand_lowering.py` and `src/mimarsinan/pipelining/pipeline_steps/config/torch_mapping_step.py`. The tuners of the fine-tuned LIF row (`training_legs.json`) are in `src/mimarsinan/tuning`. Every row result lists the digest of each driver file in `inputs.driver_code_sha256`. Our published copies of all scripts are stripped of comments and docstrings, so their digests differ from the recorded ones.

## Records

Paths are places in the records zip. `<row>`, `<budget>` and `<member>` stand for a row name, a budget and a network. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records. The other folder names are our working names, kept so that each path matches the references inside its records: `routeb` and `tlif_tuned` hold the deployment rows, and `WI-6` to `WI-9` name the collected tables.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/rows_r20/<row>/result.json` | one integer row: counts, checks, schedule, versions | `rbcount*`, `rbcheck*`, `ptqintcount` |
| `records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/rows_r20/<row>/config.json` | the row's configuration | settings |
| `records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/tables/r20/t1/WI-6/table.json` | collected table of the convolutional rows, with `K10` | `rbcheck*` |
| `records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/tables/r20/t1/WI-7/table.json` | collected table that adds the ViT-Lite-7/4 rows | `rbcheckvitlite`, `rbcheckvitop` |
| `records/frozen/routeb_tlif_untuned_t1_20261124T164838/e2e/routeb/rows_r20/C5.T-LIF.a.<budget>.256x256/result.json` | LIF rows at B0, B1 and B3 | `lifcount`, `lifcheck` |
| `records/e2e/routeb/rows_r20/C5.T-LIF.a.<budget>.256x256/config.json` | their configurations | settings |
| `records/frozen/routeb_tlif_untuned_t1_20261124T164838/e2e/routeb/tables/r20/t1/WI-8/table.json` | collected table of all rows, the converted and LIF rows included, with `K10` | `lifcheck` |
| `records/frozen/routeb_tlif_untuned_t1_20261124T164838/e2e/routeb/tables/r20/t1/WI-9/table.json` | the same table with the LIF rows as its only converted rows | none |
| `records/frozen/tuned_tlif_t1_20261124T164742/e2e/tlif_tuned/rows_r20/C5.T-LIF.at.<budget>.256x256/result.json` | fine-tuned LIF rows at B0, B1 and B3 | `liftunedcount`, `liftunedcheck` |
| `records/e2e/tlif_tuned/rows_r20/C5.T-LIF.at.<budget>.256x256/config.json` | their configurations | settings |
| `records/e2e/tlif_tuned/rows_r20/C5.T-LIF.at.<budget>.256x256/training_legs.json` | their fine-tuning passes | settings |
| `records/figures_v3/data/fig_invariance.json` | data of Fig. 8: the declared ladder, the planned rows and the executed rows | `ladderdeclared` |
| `records/frozen/routeb_budget_ladder_t1_20261120T183320/tiers/paper_locks/r20b/ladder/run/tables/t2/ladder_rows.jsonl` | the executed ladder rows, one line each | Fig. 8 |
| `records/frozen/routeb_budget_ladder_t1_20261120T183320/tiers/paper_locks/r20b/ladder/run/rows/<row>/config.json` | the configuration of each executed ladder row | settings |
| `records/frozen/routeb_layer_sharing_t1_20261120T183214/tiers/r20/b5l_run/runs/<row>/deployment_cost_report.json` | cost reports of the shared-kernel rows | none |

## How to Read the Records

- `result.json`: `K9.correct` of `K9.images` is the Dev. count. `static_program_check.theorem_1_failed` false means the static check established Theorem 1 on the program. `K4_K5_pipeline` holds the sampled run, with its declared input count in `samples`. `K13` compares the deployment's classes with those of the training code. `K7_window_counts.verdict` is the LIF rows' spike-count certificate.
- `table.json`: `rows[*]` gives each row's checks in one line, and `K10.<row>` compares a budget row with its B0 row over `outputs` output codes (`outputs_differ`, `max_abs_delta`, `passed`). A Table VI mark is a budget row with `passed` true. The fine-tuned LIF rows carry the same comparison as `K10` in their own `result.json`.
- `fig_invariance.json`: `networks[*].rows[*]` gives each planned row (`status` planned) and each executed row with `budget`, `cores` and `passes`, and `planned.declared_ladder.d` the declared divisors.
- `ladder_rows.jsonl`: each line gives `row`, `cores`, `planned_cores`, `passes`, `planned_passes` and the comparison `K10` with the row's B0 row.

## Not Included

The row runner, the per-row runner, the shared module of machine settings, the two training scripts and the launch records are not published because they hold settings of our machines. The declarations module, the row list builder, the quantizer map script, the locks of the rows and of the ladder, the planner's own output file and our design notes are not published because they hold internal working notes. The results of the executed ladder rows and the training code outputs of the shared-kernel networks are held back for the same reason, and `ladder_rows.jsonl` and the data of Fig. 8 carry what the paper reads from them. The trained checkpoints and the CIFAR-100 images are not published, the first because trained weights stay out of this copy and the second because the dataset is third-party material. The authors can be asked for any of these.
