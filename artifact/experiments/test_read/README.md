# Pre-Registered Test Read

This folder documents the two reads of the CIFAR-100 test split that stand behind the test-split numbers of Sections V and VI, and where their records lie in the records zip. Every design choice used the development split only. Each read was pre-registered and locked before it ran (Section V-A, "Data splits"). Read 1 ran on the release and read 2 on the patched release.

## What the Paper Reports

- Table VI (Deployed Accuracy of Trained Networks), CIFAR-100 rows: the Code column (the training code's count of the test images), the Test column (the deployment's count) and the Same class as Code column.
- Table VII and Section VI-C, "Gains and their cost": the mean difference, its one-sided lower bound and the Holm-adjusted $p$ of the batch-64 and batch-16 pairs of selected networks over five seeds, the reading of the pre-registered paired $t$ test, and the gains of the deployed seed-0 programs.
- Table VIII (Results for the ViT-Lite-7/4 Attention Block) and Section VI-D: the non-inferiority reading of ViT-Lite-7/4 with attention on dynamic-weight cores against attention on the host on the test images, the deployed pair at the single pass, and the development-split reading of the same seeds.

Printed keys that read these records (`artifact/numbers/index.md` gives each value with its record and field):

| Printed keys | Where | Read |
|---|---|---|
| `trharnrone`, `trharnrtwo`, `trharnrthree`, `trharnrfour`, `trharnrfive`, `trharnrtenone`, `trharnrtenhalf`, `trharndensenet`, `trharnpyramid`, `trharnvitlite`, `trharnvitop`, `trharnrfivefloat` | Table VI, Code | read 1 |
| `trrbcountrone` to `trrbcountvitop` (the eleven suffixes of the integer rows), `trptqintcount`, `trlifcount`, `trliftunedcount` | Table VI, Test | read 2 |
| `trrbagreerone` to `trrbagreevitop` (the same eleven suffixes) | Table VI, Same class | read 2 |
| `trkpairmean`, `trkpairp`, `trhpairmean`, `trhpairp`, `trkpairlowone`, `trhpairlowone`, `trkpairreadingw` | Table VII, Section VI-C, abstract | read 1 |
| `trdeployedgains` | Section VI-C | read 2 |
| `trvitmean`, `vitmargin` | Table VIII | read 1 |
| `trvitoppair` | Table VIII | read 2 |
| `vitdegraded` | Section VI-D | development split, the same seed pairs |
| `testimages`, `seeds`, `seedsword` | Sections V-A and VI | declared settings |

## Setup

- **Images.** The 10,000 images of the CIFAR-100 test split (`images` of each `test_read_result.json`), checked against the archive's digest before the read.
- **Toolchain.** Read 1 ran on the release. Read 2 ran on the patched release, whose label in the records, 644b4c3bc5d7, names the folder of read 2's records. `artifact/VERSIONS.md` gives the commits.
- **Rows.** Each result lists its deployment rows (`deployment_rows`, the B0 rows of `artifact/experiments/trained_network_deployment` with their converted and LIF rows) and its checkpoint rows (`checkpoint_rows`, the training code's own evaluation of each checkpoint). The certificates run on the first 512 test images of each row (`K6.samples` and `certificates.K4.samples` of each deployment row).
- **Seed pairs.** The pairs of selected networks were fixed on seed 0 and read at seeds s1 to s5 (`n_seeds` and `printed.selection` of the selections' `pairs_result_record.json`). The ViT-Lite-7/4 pairs read the network with attention on dynamic-weight cores against the same network with attention on the host, with the margin of `margin_images` test images (the ViT-Lite-7/4 `pairs_result_record.json`). The development-split pairs of ViT-Lite-7/4 are documented in `artifact/experiments/attention_block_seed_pairs`.

## How It Was Run

1. A lock script wrote the lock of a read once, before the read: the toolchain, the code digests, the digests of each row's configuration and result, and the reported quantities. It is not published because it holds settings of our machines.
2. A read script ran the read. It refused to start unless the lock existed and every digest in it still matched, extracted only the test batch, ran every locked row and wrote `test_read_result.json` and `test_read_done.json`. It is not published because it holds settings of our machines.
3. `scripts/read/eval_checkpoints.py` evaluates each trained checkpoint with its own training code (eval mode, float32, argmax over the logits). A program script ran each deployed program with the executor its development row used, and the int64 golden model on the first images. A shared module, which `eval_checkpoints.py` imports, held the paths, the seals and the digests that every part shares. Both are not published because they hold internal working notes.
4. `scripts/pairs/headline_v46_pairs/collect_pairs.py` and `scripts/pairs/vit_gate_pairs/collect_pairs.py` turn the read's seed-pair counts into the pre-registered readings (paired differences, one-sided bounds, Holm correction) and write `pairs_result_record.json`. They import the pair module `pairs_stats.py`, which lies in `artifact/experiments/equal_budget_scenarios/scripts/pairs` for the selections and in `artifact/experiments/attention_block_seed_pairs/scripts` for ViT-Lite-7/4, and a common pair module and a lock module of the seed pairs, which are not published.

`scripts/read/eval_checkpoints.py` is the version of read 2. Our published copies of all scripts are stripped of comments and docstrings.

## Records

Paths are places in the records zip. Each `records/frozen/gate_and_test_read_<time>` folder is the frozen copy of one read's records, and `06_build` is the folder of the training runs whose seed pairs the read counts.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/gate_and_test_read_t1_20261109T230030/e2e/test_lock.json` | lock of read 1, published in part | `trharn*` (which checkpoint row each Code cell reads) |
| `records/frozen/gate_and_test_read_t1_20261109T230030/e2e/test_read_out/test_read_result.json` | result of read 1 | `trharn*`, pair keys |
| `records/frozen/gate_and_test_read_t1_20261109T230030/06_build/headline_v46_pairs/pairs_result_record.json` | readings of the selection pairs | `trkpairmean`, `trkpairp`, `trhpairmean`, `trhpairp` |
| `records/frozen/gate_and_test_read_t1_20261109T230030/06_build/vit_gate_pairs/pairs_result_record.json` | reading of the ViT-Lite-7/4 pairs | `trvitmean` |
| `records/frozen/gate_and_test_read_t1_20261109T230030/e2e/test_read_done.json` | completion record of read 1 | none |
| `records/frozen/gate_and_test_read_t1_20261124T231409/e2e/test_read_644b4c3bc5d7/test_lock.json` | lock of read 2, published in part | `trrbcount*`, `trptqintcount`, `trlifcount`, `trliftunedcount` (the development-split counts beside) |
| `records/frozen/gate_and_test_read_t1_20261124T231409/e2e/test_read_644b4c3bc5d7/test_read_out/test_read_result.json` | result of read 2 | `trrbcount*`, `trrbagree*`, `trptqintcount`, `trlifcount`, `trliftunedcount`, `trdeployedgains`, `trvitoppair` |
| `records/frozen/gate_and_test_read_t1_20261124T231409/e2e/test_read_644b4c3bc5d7/test_read_done.json` | completion record of read 2 | none |
| `records/excerpts/vit_pairs_result.md` | word for word lines of the ViT-Lite-7/4 result note with the margin | `vitmargin` |
| `records/excerpts/headline_pairs_result.md` | word for word lines of the selections' result note with the seeds | `seeds`, `seedsword` |

## How to Read the Records

- `test_read_result.json`: `deployment_rows[*]` holds one deployed row each, named by `row`. `K9.correct` is its count of the test images, and `K13` counts the images whose deployed class equals the training code's (`agree_qat`, `agree_qat_fp64`). `certificates` holds the checks on the first images. `checkpoint_rows[*]` holds the training code's counts per device (`devices.<device>.counts`). `track_files` names the seed-pair files of read 1 and their validator.
- `test_lock.json`: both locks are published in part. Each holds only the fields that printed numbers read, as its `_note` says. In the lock of read 1, `training_code.printed[*]` names, for each printed Code cell, its checkpoint row (`checkpoint_row`) and device (`device`). In `checkpoint_rows` of read 1 and `deployment_rows` of read 2, an entry that a printed number reads gives the development-split count beside it (`e01_counts` with `kind`, and `e01` with `correct` of `images`), and every other entry is null, so that each entry keeps the place of its row in the lock.
- `pairs_result_record.json` of the selections: `cells.<cell>` gives the differences per seed in test images (`d_images`) and `mean_pp`, `sd_pp`, `lower95_pp`. `holm.<cell>` gives `p_holm` and `rejected`. `development` repeats the readings on the development split, printed beside the test reading and never replacing it.
- `pairs_result_record.json` of ViT-Lite-7/4: `primary.entries.<seed>` gives the correct counts with attention on dynamic-weight cores (`operand_correct`) and on the host (`twin_correct`) and their difference, `margin_images` the margin, and `spread` the accuracies per seed.

## Not Included

The lock script, the read script, the driver's module of machine settings, the common pair module and the run logs are not published because they hold settings of our machines. The program script, the shared read module, the fields of the two locks that no printed number reads, the test counts per seed of the seed-pair checkpoints, the lock module of the seed pairs, the two result notes and the design notes of the read are not published because they hold internal working notes or settings of our machines. The excerpts above quote the lines the paper reads, and `pairs_result_record.json` holds every reading. The trained checkpoints and the CIFAR-100 images are not published, the first because trained weights stay out of this copy and the second because the dataset is third-party material. The authors can be asked for any of these.
