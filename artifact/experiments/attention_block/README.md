# Attention Block of ViT-Lite-7/4

## What the Paper Reports

Section III-A states the share of ViT-Lite-7/4's multiply-accumulates that runs on cores once its attention products run on dynamic-weight cores (key `attnmacshare`).
Section VI-D and Table VIII report the attention block's estimates: the product energy on the SRAM macro and on the generic SRAM point (`productmacenergy`, `productmacgeneric`, with the technology node `genericnode`), the host transfer of operands against the product energy (`hosttransferratio`), and the deployed accuracy difference on the SRAM macro with attention on dynamic-weight cores minus that with attention on the host (`softmaxdelta`).

`artifact/numbers/index.md` gives each value with its record and field.
The non-inferiority rows of Table VIII over five seeds are documented in `artifact/experiments/attention_block_seed_pairs` (development split) and `artifact/experiments/test_read` (test images).

## Setup

- **Toolchain.** The share and the energy terms come from records of the release, which the fixes leave unchanged (Section V-A). The deployed difference comes from the patched release, which its `run.json` names in `head` (beginning 644b4c3bc5d7) and `base_commit` (the release). `artifact/VERSIONS.md` gives the commits.
- **Network.** ViT-Lite-7/4 trained on CIFAR-100, with the softmax attention block and the ReLU attention block (field `instance` of `fig1_vit_block.json`).
- **Data.** The development split, 5,000 images held out of the CIFAR-100 training batch (field `data` of `run.json` and field `n_images` of `aggregate.json`). `val_split_e01.json` gives the split's indices and the rule that drew them (`seed`, `per_class`, `rule`). These runs do not read the test batch.
- **Seeds.** One trained seed (field `deployed_top1_e01/class` of the patched release's `fig1_vit_block.json`).
- **Chips.** The SRAM macro and the generic 22 nm SRAM point (bundle `generic_estimated_22nm`), each with host and on-chip channels (field `energy_terms/per_bundle`).

## How It Was Run

1. `scripts/census_n2.py` builds the network with a dynamic-weight core type declared for the attention products and counts the multiply-accumulates on cores before and after, writing the records under `acceptance4/rows/`.
2. `scripts/deploy_gate.py` deploys the trained programs on the SRAM macro and counts correct decisions on the development split for each form: softmax or ReLU attention, 4, 6 or 8 operand bits, and the same network with attention on the host (the keys of `deployments` in `aggregate.json`, the last marked T). It loads the network and the images through `scripts/import_gate_checkpoint.py` and the modules in `scripts/vit_gate/`.
3. `scripts/aggregate_gate_r2.py` pairs the count runs and writes `aggregate.json`.
4. `scripts/price_n2_r2.py` and `scripts/price_resident_r2.py` price the writes and the energy terms with the toolchain's write model (`price.json` and `price_resident.softmax.O8.json`).
5. A short script read these records and wrote the block's data file `fig1_vit_block.json`. The data file names it in its field `script` and lists the records it read in its field `inputs`. The script is not published because it holds internal working notes.

## Records

Paths are places in the records zip. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records. The folder `n2_cl7_t1_20261107T093743` holds the release's records and the folder `n2d_keys_t1_20261120T183656` those of the patched release.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_fig1_data/fig1_vit_block.json` | the block's data file: share of multiply-accumulates, energy terms, writes | `attnmacshare`, `productmacenergy`, `productmacgeneric`, `hosttransferratio`, `genericnode` |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5/acceptance4/rows/vit_lite_7_4_softmax.json` | multiply-accumulates before and after the dynamic-weight core type, softmax block | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5/acceptance4/rows/vit_lite_7_4_relu.json` | the same for the ReLU block | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5/acceptance/vit_lite_7_4.json` | the declared cores of each form | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_aggregate/acceptance3/aggregate.json` | correct counts per form and their pairs on the release | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_fig1_store/acceptance5/price.json` | write prices with the figure's backing store | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_fig1_store/acceptance5/price_resident.softmax.O8.json` | the same for the resident softmax program at 8 operand bits | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_22nm_store/acceptance5/price.json` | write prices on the generic 22 nm point's store | read through the data file |
| `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_22nm_store/acceptance5/price_resident.softmax.O8.json` | the same for the resident softmax program | read through the data file |
| `records/frozen/n2d_keys_t1_20261120T183656/11_wave2/n2_attention/record_644b4c3bc5d7/fig1_vit_block.json` | the deployed pairs on the development split | `softmaxdelta` |
| `records/frozen/n2d_keys_t1_20261120T183656/11_wave2/n2_attention/record_644b4c3bc5d7/acceptance3/aggregate.json` | correct counts per form and their pairs | read through the data file |
| `records/frozen/n2d_keys_t1_20261120T183656/11_wave2/n2_attention/record_644b4c3bc5d7/run.json` | the count runs, their exits, the toolchain and the data split | none |
| `records/06_build/screen_v42/val_split_e01.json` | indices of the development split | none |

## How to Read the Records

- `on_chip_mac_share`: `before_n2` and `after_n2` are the shares of `total_macs` on cores, and `lowered_product_macs` counts the attention products moved onto cores.
- `energy_terms/per_bundle/<bundle>.<channel>`: `product_mac_mJ`, `operand_write_mJ` and `static_write_mJ` per inference. The bundle `generic_estimated_22nm#profile.host` gives the generic 22 nm point with the host channel.
- `deployed_top1_e01/pairs/softmax.8`: `delta_pp` is the difference in percentage points, `paired_se_pp` its paired standard error, `a_only` and `b_only` the images only one form gets right, and `operand` and `twin` the correct counts with attention on dynamic-weight cores and on the host.

## Not Included

The script that wrote the data file and the build report of the dynamic-weight core type are not published because they hold internal working notes. The trained ViT-Lite-7/4 checkpoint and the CIFAR-100 images are not published, the first because trained weights stay out of this copy and the second because the dataset is third-party material. The authors can be asked for the script, the report and the checkpoint.
