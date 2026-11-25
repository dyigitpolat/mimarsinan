# Co-Design and Sequential Design at Matched Accuracy

## What the Paper Reports

The co-design paragraph of Section VI-C compares co-design with sequential design, which selects chip or network first
from co-design's candidates, without an energy budget and under the toolchain's estimate. At each accuracy target
studied, the better of the two orders matches co-design's energy and time, and with chip energy at its high reading the
better order costs more at targets near a stated accuracy.

Printed keys (in `artifact/numbers/index.json`): `iaratiok`, `iahighleast`, `iahighmost`, `iahightargets` and
`iahighnear`.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Toolchain.** The patched release, whose digest, beginning `644b4c3bc5d7`, the result names in `RR_commit` of
  `result.json`. `artifact/VERSIONS.md` maps these names to the toolchain repository.
- **Candidates.** The 48 CIFAR-100 networks of the equal-budget pool (see
  `artifact/experiments/equal_budget_scenarios/`), priced on the study chips over a ladder of core counts and batch
  sizes. Accuracy is the count of correct images of 5,000 on the development split (`members/accuracy/total` of
  `rules.json`).
- **Rules.** The rules were declared before any computation: co-design, the sequential orders, the better of them at
  each target, the accuracy targets and the budget scenarios. `rules.json` is published in part. It holds only the
  fields that printed numbers read: the count above and the template of the paragraph's sentence (`sentences/main`
  and `sentences/class_words`).
- **Area reading.** The area reading of the main result is `main_area_reading` of `result.json`, repeated in
  `paper_text/main_area_reading`.

## How It Was Run

1. A lock script pinned the rules and every input by its digest before the run.
2. A driver, which reads JSON inputs only and imports no toolchain module, read the priced candidates, formed co-design and both sequential orders at each
   accuracy target in each context of the rules, and wrote `result.json` with the sentences the paper's paragraph
   reads.

Both scripts are not published, since they hold internal working notes. The rules that define every quantity the
driver computes are not in the published part of `rules.json`.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/iso_accuracy_t1_20261121T155409/e2e/iso_accuracy/runs/record_644b4c3bc5d7/result.json` | the matched-accuracy run of record | `iaratiok`, `iahighleast`, `iahighmost`, `iahightargets`, `iahighnear` |
| `records/frozen/iso_accuracy_t1_20261121T155409/e2e/iso_accuracy/rules.json` | the rules declared before the run, published in part | the same keys |

## How to Read the Records

In `result.json`, `paper_text/cells/<scenario>/main` holds the sentence of the main reading per scenario, with the
ratio of sequential to co-designed cost, and `paper_text/cells/<scenario>/values` the values the sentence prints.
The fields that the keys of the high energy reading read are listed with each key in `artifact/numbers/index.json`. `result.json` is a large file and is best read as a stream.

## Not Included

The lock of the run, its manifest of inputs with their digests, the rest of the rules, the priced candidates the driver
read, the lock script and the driver are not in the public copy, since they hold internal working notes. The authors
can be asked for them.
