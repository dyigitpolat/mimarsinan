# ImageNet Deployment on NeuRRAM

## What the Paper Reports

Table VI (Deployed Accuracy of Trained Networks, Section V-C) holds a block "ImageNet on NeuRRAM, 5,000 images" with
six rows: ResNet-50 with 4-bit weights, ResNet-50, ResNet-18, MobileNetV2, DenseNet-121 and ViT-B/16. For each network
the table prints the float network's count on the same images (Code), the deployed count (Test), the images whose
deployed class is the float network's (Same class as Code) and the core budgets whose outputs equal those of the
single-pass deployment. Its note states that the ImageNet outputs are compared on 500 images and that 192 cores are
four NeuRRAM chips. Section V-C adds that, except the 4-bit row, the ImageNet rows store each 8-bit published weight on
two 4-bit cell pairs and take 8-bit inputs, settings we chose since NeuRRAM states none.

Printed keys (in `artifact/numbers/index.json`): `inetcoderfiftyw`, `inettestrfiftyw`, `inetsamerfiftyw`,
`inetbudgetsrfiftyw` and the same four prefixes with `reighteen`, `mobilenet`, `densenet` and `vit` for the other
networks, `inettestrfiftyn` for the 4-bit row, and `inetsubsetimages`, `inetktenimages`, `inetchipsword`,
`inetweightbits`, `inetcellpairs` and `inetinputbits` for the block's settings.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`, with `[*]` for every element of
a list.

- **Toolchain.** The patched release. The records name it by the digest `63d7c3274503` of its exported source, made on
  the release commit `e5f6bb37` (`rows[*]/toolchain/commit` and `rows[*]/toolchain/base_commit` of each group record).
  `artifact/VERSIONS.md` maps these names to the toolchain repository.
- **Networks.** Five torchvision classifiers with their published ImageNet-1K weights: ResNet-50
  (`ResNet50_Weights.IMAGENET1K_V2`), ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`), DenseNet-121
  (`DenseNet121_Weights.IMAGENET1K_V1`), MobileNetV2 (`MobileNet_V2_Weights.IMAGENET1K_V2`) and ViT-B/16
  (`ViT_B_16_Weights.IMAGENET1K_V1`) (`members[*]/weights_enum` of `members.json`). The rule that chose them and its
  trace are in the same record (`rule_file`, `selection_trace`). No weight is trained or tuned: the no-update check
  passes on every perceptron of every completed row (`rows[*]/certificates/NU` of each group record).
- **Chip.** NeuRRAM's chip description in the toolchain, 48 cores per chip (`x/executed/C` of each 8-bit group record). The
  rows with 8-bit weights declare cores of 128 x 128 weights, two 4-bit differential pairs per 8-bit weight on 4 cells
  over 2 adjacent rows, and 8-bit inputs (`departures` of each 8-bit group record, its first two entries). The 4-bit
  row keeps the chip description's own geometry of 128 x 256 weights of 2 cells each (`departures` of the 4-bit group
  record, its first entry). Each group record lists the declared departures from the chip's published sources with
  the pages they rest on, among them the digital idealization of the analog products (`departures`).
- **Budgets.** `chip` declares one chip's 48 cores, `chip+pm` the same cores with position multiplexing (one copy of a
  shared kernel serving its output positions in turn), `chipx4+pm` 192 cores with position multiplexing, and `res` a
  resident pool of 4,000,000 cores, the single-pass reference (`budgets`, `rows[*]/budget` and
  `rows[*]/declared_cores` of each group record).
- **Data.** A fixed subset of 5,000 images of the ImageNet validation split (`rows[*]/images` of the row with the
  deployed count) and calibration images from the training split, both fixed by a rule on file names before any network
  was evaluated (`scripts/manifest/build_manifest.py`). The outputs of each budget are compared with the resident
  deployment's on the first 500 images of the subset (`K10/comparisons[*]/images` of each group record).

## How It Was Run

1. `scripts/manifest/build_manifest.py` fixes the subset and the calibration list, and `scripts/manifest/build_dev.py`
   a separate development list that only smoke runs of the driver read.
2. `scripts/members/select_members.py` applies the member rule and writes `members.json`.
3. `scripts/float/float_eval.py` runs each float network on the subset as a sanity check, and
   `scripts/float/float_dev.py` on the development list.
4. A row driver runs one row, one network at one budget, through the toolchain's pipeline session
   (`src/mimarsinan/pipelining/session.py` in the toolchain repository), with images served by
   `scripts/driver/provider.py` and the network built by `scripts/driver/members_builder.py`.
5. `scripts/driver/harvest.py` runs the deployed program on the row's images with the toolchain's value executor in
   double precision and, on the first images, the int64 golden model (`src/mimarsinan/chip_simulation/value_run/`),
   and writes the row's result. `scripts/driver/no_update.py` checks that the pipeline changed no weight.
6. `scripts/driver/structure.py` writes each group's structural pack, and `scripts/driver/collect.py` builds each
   group record from its rows, with the bit-for-bit comparison of every budget's outputs against the resident
   deployment's.

The row driver, its shared module, its configuration of declarations, its lock check, its launch checks and its
launcher are not published, since they carry settings of our machines and internal working notes. The published
modules import them by name, so the published scripts document the procedure but do not run on their own.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/resnet50.neurram.W.group.json` | ResNet-50 with 8-bit weights, all budgets | `inetcoderfiftyw`, `inettestrfiftyw`, `inetsamerfiftyw`, `inetbudgetsrfiftyw` |
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/resnet18.neurram.W.group.json` | ResNet-18, all budgets | `inetcodereighteen`, `inettestreighteen`, `inetsamereighteen`, `inetbudgetsreighteen` |
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/mobilenet_v2.neurram.W.group.json` | MobileNetV2, all budgets | `inetcodemobilenet`, `inettestmobilenet`, `inetsamemobilenet`, `inetbudgetsmobilenet` |
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/densenet121.neurram.W.group.json` | DenseNet-121, all budgets | `inetcodedensenet`, `inettestdensenet`, `inetsamedensenet`, `inetbudgetsdensenet` |
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/vit_b_16.neurram.W.group.json` | ViT-B/16, all budgets | `inetcodevit`, `inettestvit`, `inetsamevit`, `inetbudgetsvit` |
| `records/frozen/w38_imagenet_t1_20261120T224345/imagenet/records/63d7c3274503/resnet50.neurram.N.group.json` | ResNet-50 with 4-bit weights, every budget refused | `inettestrfiftyn` |
| `records/imagenet/members/members.json` | the networks, their weights and the member rule's trace | none directly |

The block's settings (`inetsubsetimages`, `inetktenimages`, `inetchipsword`, `inetweightbits`, `inetcellpairs`,
`inetinputbits`) are listed in `artifact/numbers/index.json` with their records or the section that states them.

## How to Read the Records

- A group record (`*.group.json`) has one entry in `rows` per budget. `rows[*]/state` is `complete` or names a refused
  session, whose `rows[*]/error` gives the toolchain's message. For the row with the deployed count, `correct` is the
  deployed count with ties counted as wrong (`correct_ties_lowest_index` counts a tie for the lowest class index),
  `float_correct_same_images` the float network's count on the same images, `agree_with_float` the images of the same
  class as the float network, `declared_cores` the budget's cores and `paired` the paired comparison with the float
  network.
- `rows[*]/certificates` holds the checks of a completed row with their denominators: `K6` the int64 golden model's
  check on the first images, `K11` the record identity and `NU` the no-update check.
- `K10` holds the comparison of each budget's output codes with the reference budget's (`reference_budget`, the
  resident deployment): `images`, `outputs`, `outputs_differ` and `max_abs_delta` per budget. A budget whose outputs
  equal the reference's has `passed` true and `outputs_differ` 0.
- `departures` lists the declared departures from NeuRRAM's published sources, with the pages they rest on. One entry
  of each group record reads "(internal note withheld)" in this copy.

## Not Included

The per-row result files of each budget, the lists of ImageNet files, the float networks' outputs, the structural
packs and the run logs are not in the public copy, and neither are the row driver and its launch modules named above.
The ImageNet images and the torchvision weights are third-party material, available from their publishers. The
authors can be asked for the withheld records and scripts.
