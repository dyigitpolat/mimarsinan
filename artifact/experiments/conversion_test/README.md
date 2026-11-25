# Conversion Test of torchvision Classifiers

## What the Paper Reports

Section V-A states that our conversion test tries 80 torchvision ImageNet classifiers at random weights. Section V-B
(Conversion test) reports, with Fig. 3(b), that 78 of 80 classifiers convert, with double-precision errors up to
4.3 × 10⁻¹³, 0 rejections and 0 conversions beyond the bound. The caption of Fig. 3 adds that for 2 networks
the comparison cannot decide, as double-precision rounding alone exceeds its bound. The script and data of Fig. 3(b)
are in `artifact/figures/`.

Printed keys (in `artifact/numbers/index.json`): `censusmembers`, `censusconvert`, `censusagree`, `censusrefused`,
`censussilent` and `censusundecidable`.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Toolchain.** The release, commit `e5f6bb37` in the toolchain repository (`run/code_state/head` of `summary.json`).
- **Networks.** Every classifier builder torchvision lists, 80 in all (`blocks/torchvision / verdict/members` and
  `blocks/torchvision / verdict/population_scope` of `summary.json`), each built with random weights from a fixed seed
  (`torch.manual_seed` in `scripts/harness/cov_members.py`).
- **Inputs.** Each classifier is checked once, at the input side a rule fixes before any check runs: 112 x 112 for 64
  classifiers, 224 x 224 for 9, 128 x 128 for 4 and 256 x 256 for 3 (`blocks/torchvision / verdict/sides` and
  `blocks/torchvision / verdict/side_members` of `summary.json`). The rule and the side of every member are in
  `scripts/harness/side_table_w12.json`, whose digest the summary records (`run/side_table_sha16`). The published table
  names record paths by their places in the zip, so its digest differs from that one.
- **Check.** The toolchain's own value-parity probe in double precision: stage 1 compares the converted model with the
  native model, stage 2 the identity value program, each at the probe's own bound (`rows[*]/stage`, `rows[*]/bound` and
  `rows[*]/relative_error` of `summary.json`). A classifier that agrees is counted as converting, and its host MAC share
  is counted from the converted program (`rows[*]/host_mac_share`).

## How It Was Run

1. `scripts/harness/cov_members.py` lists the members, every classifier builder torchvision lists and a set of
   attention and recurrent test models, and builds each with random weights.
2. `scripts/harness/cov_side.py` fixes each member's input side once, in `scripts/harness/side_table_w12.json`.
3. A job script runs one check of one member: the verdict check calls the toolchain's
   `probe_witness_conversion` (`src/mimarsinan/torch_mapping/converter.py`, with its stages in
   `src/mimarsinan/torch_mapping/value_parity/`), and for a converting member it counts the host MAC share with the
   toolchain's counters (`src/mimarsinan/mapping/verification/onchip_fraction.py` and
   `src/mimarsinan/mapping/verification/onchip_majority.py`).
4. `scripts/harness/cov_record.py` writes each record with the code state of the toolchain it imported and a digest of
   the harness's own code (`run/code_state/harness_sha16` of `summary.json`).
5. A summarizer counts the classes and writes `summary.json`, with its sentence and a second
   counting path that must agree with the first (`table_i_second_path` and `table_i_second_path_self_test`).
6. Three checks, the verdict checks of ViT-H/14 and RegNetY-128GF and the supplementary check of RegNetY-128GF, were
   run again with more memory. A merge script replaced exactly these records and ran the summarizer again on the merged
   records, which gives the `summary.json` below.

The job script, the summarizer, the script that ran the checks in parallel under memory limits, its planner and its
supervisor, and the merge script are not published, since they hold settings of our machines. No torchvision verdict check reached a limit
(`blocks/torchvision / verdict/not_checked_memory` and `blocks/torchvision / verdict/not_checked_wall` of
`summary.json`, both empty).

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/census_baseline_t1_20261109T230030/tiers/fcap/runs/r5b_merged/summary.json` | class counts, worst agreement, sides, one row per member and check, and the sentence | `censusmembers`, `censusconvert`, `censusagree`, `censusrefused`, `censussilent`, `censusundecidable` |

## How to Read the Records

- `blocks/torchvision / verdict` of `summary.json`: `members` (the classifiers checked), `class_counts` (`converts`
  and `converts with CPU share` are the converting classifiers, `not decidable at the line` the classifiers whose
  comparison cannot decide), `worst_agreement_error` (the largest relative error of a converting classifier), and
  `rejected_by_root_and_stage` (empty: no rejection). A class that does not occur, such as a value mismatch, has no
  entry in `class_counts`.
- `rows` of `summary.json`: one entry per member and check with `member`, `check` (`verdict`, or `supplement` for
  stage 1 alone at 224 x 224), `population` (`torchvision` or `test model`), `input_shape`, `class`, `stage`,
  `relative_error`, `bound`, `host_mac_share` and `host_mac_kinds`.
- The blocks of the `supplement` check and of the test models are records of the same run that the paper does not
  report.

## Not Included

The per-member records of the run (one line per member and check, with the probe's settings and the software
versions), the merge record and the scripts named above as not published are not in the public copy, since
they hold settings of our machines. The authors can be asked for them.
