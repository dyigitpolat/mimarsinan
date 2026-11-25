# FPGA Runs of the Spiking Core

## What the Paper Reports

Section V-B: on the FPGA, an Alveo U55C board, ODIN at two core geometries runs MNIST networks in passes, matching the golden model's pass outputs in 250 of 250 checks and its predicted classes in 128 of 128 (`cardrealizations`, `cardchecks`, `cardpredicted`).
Section V-A states that the FPGA ran a development build's programs, byte-identical to the release's.
Section V-D ("Pass time on the FPGA") and Fig. 10 compare measured pass times with the timing model of the estimate.
The model was fixed before measuring and tested on 4 and 2 published cores (`cardheldoutcounts`).
Of 5 predictions fixed before measuring (`cardexpectlocked`), 4 were confirmed (`cardexpectheld`).
The fifth failed, with errors of +0.2% to +6.2% on the published core and +9.0% to +298% on the wider generated core (`cardresidenterr`, `cardresidenterrwide`).
All 40 runs on the wider generated core pass their checks (`cardwideclean`), where its earlier bitstream passed 24 of 40 (`cardwidefirstclean`, `cardwidefirstcampaigns`), with at least 0.222% of kernel launches failing (`cardwidefirstrate`).

## Setup

- Board: an Alveo U55C, as the note of Table III states.
- Published core: 256 axon lines, 256 neurons, 4-bit weights (`hw_execution.json`, field `realizations/published/declaration/axon_lines`, and `artifact/figures/data/fig_card.json`, fields `geometries/0/geometry/neurons` and `geometries/0/geometry/weight_bits`).
- Wider generated core: 1,024 axon lines, 256 neurons, 8-bit weights (`hw_execution.json`, field `realizations/widened/declaration/axon_lines`, and `fig_card.json`, field `geometries/1/geometry`).
- Networks: MNIST networks, as Section V-B states. The published part of `hw_execution.json` does not name them.
- Samples: 64 samples per pass for the pass times (`expectations.json`, field `samples` of each entry whose name begins with E3, and the column `sample` of the census tables), and 64 predicted classes per core (`hw_execution.json`, field `realizations/<core>/accuracy/decisions_as_predicted`).
- Predictions: the five predictions fixed before measuring are named E1 to E5 in `expectations.json`, each with one entry per core and per arm or pass. Section V-D states them in words.
- Programs: a development build's programs, byte-identical to the release's (Section V-A). `artifact/VERSIONS.md` gives the release's commit.

## How It Was Run

1. Building the programs. The toolchain at the paper version holds the programs that build each FPGA program and its deployment bundle (`make_deployment_bundle.py`, `deployment_bundle_builder.py` and `gen_chip_rtl.py`, among its scripts), the host side that drives the board (`odin_deployment_executor.py` and `odin_board_driver.py`, in a package beside them), and the Verilog source of the FPGA kernel under `hw/fpga/`. We do not copy them here.
2. Checking the programs. We regenerated every program on the release and compared it byte for byte with the one the FPGA ran.
3. Predicting. Before the runs, `scripts/fidmodel.py` evaluated the timing model on each bundle's census, the input words and events per sample and pass (`records/tiers/fidelity/ufidfpga/census/`), and wrote the predicted device cycles of every pass for both cores, as run and with the program resident.
4. Measuring. The FPGA ran each program in passes, and each run wrote the device cycles of every pass and its checks.
5. Reading. `scripts/read_session_r10.py` reads the runs into `readings.json`, and `scripts/keys_r10.py` derives `keys.json` (fits and errors per pass). Both import two helper modules that read the runs and compute the keys, which this copy does not include. `scripts/s08_expect.py` evaluates each prediction on the readings (`expectations.json`), and `scripts/e2_passes.py` writes the table of measured and fitted time per pass.
6. Deriving the printed values. Two derivation scripts derive the printed counts and error ranges, each naming its inputs, their digests and the arithmetic. They are not published, since they hold internal working notes.
7. Drawing Fig. 10. A data script, not published since it holds internal working notes, builds `artifact/figures/data/fig_card.json`, which holds the measured points and the fit of every pass, and `artifact/figures/scripts/fig_card.py` draws the figure.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/other/card_execution/hw_execution.json` | published in part: both cores' axon lines, pass checks and predicted classes on the FPGA | cardrealizations, cardchecks, cardpredicted |
| `records/hardware/multicore/r5_card_session_3/read/readings/readings.json` | the runs of both cores per core count, arm and pass, with device cycles and fits, and the runs of the wider core's earlier bitstream | cardwidefirstclean, cardwidefirstcampaigns |
| `records/hardware/multicore/r5_card_session_3/read/readings/expectations.json` | each prediction entry, held or not, with its range | |
| `records/hardware/multicore/r5_card_session_3/read/keys/keys.json` | the fits per pass, the errors at the tested core counts and the core counts the model was tested on | cardheldoutcounts |
| `records/tiers/fidelity/ufidfpga/census/stock_census.tsv` | input words and events per sample and pass on the published core | |
| `records/tiers/fidelity/ufidfpga/census/generated_census.tsv` | the same on the wider generated core | |
| `records/tiers/paper_v6/fix5/p0/keys_card/derived_values.json` | derived counts with inputs and arithmetic | cardexpectlocked, cardexpectheld, cardwideclean, cardwidefirstrate |
| `records/tiers/paper_v6/fix5/p0/keys_asm/derived_values.json` | derived error ranges | cardresidenterr, cardresidenterrwide |

## How to Read the Records

`hw_execution.json` is published in part. It begins with a `_note` and holds only the fields that printed numbers read.
In `readings.json`, `realizations/<core>/campaigns` lists the runs with their checks, `realizations/<core>/cells` gives the device cycles and the fit per core count, arm and pass, and `stated_with/generated/0` holds the earlier bitstream's runs on the wider core, with `clean` and `campaigns`.
In `expectations.json`, each entry name gives the prediction, the core (`stock` the published core, `generated` the wider core) and the arm, and `held` gives the outcome.
The arm `as_run` streams the program before each pass, `resident` keeps it on the core.
In `keys.json`, `derived/<core>/rtwo` gives the fits per pass, and `derived/stock/held_out/core_counts` holds the core counts the model was tested on.
In `hw_execution.json`, `realizations/<core>/certificates` counts the pass checks and `realizations/<core>/accuracy/decisions_as_predicted` the predicted classes that the FPGA matched.
Each derived value names its input files, their sha256, the field and the arithmetic. An input's `file` is its place in the zip. Some inputs are not in the zip.
The count of predictions reads the statement fixed before measuring, which is not published. The derived record lists the five names it counted.

## Not Included

The statement of the predictions fixed before measuring, the predicted pass times, the timing model as data, the table of measured and fitted time per pass, and the rest of `hw_execution.json` are not published, as they hold paths of our machines and internal working notes. `scripts/fidmodel.py` holds the timing model in code, and `fig_card.json` holds the points and fits of Fig. 10.
The byte-for-byte comparisons of the regenerated programs, the raw outputs of every run on the board, the run scripts and logs, and the script that collected `hw_execution.json` are not published, as they hold settings and names of our machines.
The two helper modules imported in the reading of the runs, the two derivation scripts, the data script of Fig. 10 and an internal check note that one derived bound cites are not included either.
The authors can be asked for any of them.
