# Figure Scripts and Their Data

This folder rebuilds the sixteen figures of the paper from one data file each. A figure script reads nothing but
its data file, the style kit and the label table. It does not read a record, so a reader can rebuild every figure
without the records zip. The records zip holds the records that the data files were built from.

The folder `out/` holds the sixteen keys files only. The PDF and PNG files of each figure are not in this folder
and come from one command (see Building the Figures).

## Contents

- `scripts/fig_<name>.py`: the sixteen figure scripts.
- `data/fig_<name>.json`: one data file per figure, fifteen in all (Fig. 5, the invariants diagram, draws no
  measured value and has none). `data/MANIFEST.sha256` holds the SHA-256 digest of each data file.
- `_style.py`: the style kit. It sets the printed sizes, the type face, the colours by role and the checks that a
  build runs on every figure.
- `labels.tsv`: the paper's terms. A data file keeps the words of its records, and the kit prints the paper's word
  in place of a whole string that the table lists.
- `make_all.py`: builds all figures or the ones named.
- `out/<name>.keys.json`: the sixteen keys files of our build.
- `tests/test_ink_overlap.py`: a test of the kit's check that two labels do not overlap.
- `scripts/build_fig_<name>_data.py`: eleven scripts that made data files from the records, for provenance only
  (see Data Builders).

## Building the Figures

From this folder:

    python3 make_all.py --no-copy               # all sixteen figures into out/
    python3 make_all.py --no-copy fig_fronts    # one or more figures by name
    python3 tests/test_ink_overlap.py           # the test of the kit, it prints "OK ... 4 checks"

`make_all.py` first compares each data file with `data/MANIFEST.sha256` and stops if one differs. It then runs
each figure script in its own process. A script writes into `out/` three files:
`<name>.pdf` at its printed size, `<name>.png` at 200 dpi and `<name>.keys.json`, which lists every value the figure
draws with the key or the record it comes from. Each script prints one report line that starts with `OK` or `FAIL`.

Always pass `--no-copy`. Without it, `make_all.py` also copies each PDF into `../paper/figures`, the folder of our
paper source, which this repository does not hold, and it stops with an error after the first figure.

With the versions listed below, a build from this folder gave PNG files equal pixel for pixel to the ones we built
for the paper. The PDF bytes of the two diagrams, Figs. 2 and 4, can differ from one build to the next while their
PNG and keys files stay the same. A rebuilt keys file holds the same values as the one in `out/`.

## Requirements

- Python 3.12 with matplotlib 3.11.1, NumPy and Pillow. We built the figures with Python 3.12.3, NumPy 2.4.4 and
  Pillow 12.2.0.
- The diagram library sciviz at commit 945c459 of https://github.com/dyigitpolat/sciviz, for example
  `pip install "git+https://github.com/dyigitpolat/sciviz@945c459ad2eef830d9f4468ad6eae068cee94ebf"`. The kit
  puts the folder `sciviz` beside the checkout of this repository (`../../../sciviz` from this folder) first on
  the import path, so a clone of sciviz placed there is used. Without that folder, the kit imports the installed
  package.
- CairoSVG (we used 2.9.0) and the cairo library, which sciviz uses to write the PDF of a diagram (Figs. 2 and 4).
  When `rsvg-convert` or `inkscape` is on the path, sciviz uses that program instead. The PDF bytes can then
  differ, while the keys file stays the same.
- `pdftoppm` and `pdfinfo` from poppler-utils (we used 24.02.0), which the kit uses for the PNG and the size checks.
- The Noto Sans fonts. The kit loads every `NotoSans*-*.ttf` from `/usr/share/fonts/truetype/noto` (the Debian and
  Ubuntu package fonts-noto-core) or, when the environment variable `FIG_FONT_DIR` is set, from the folder it names.
- Optional: with `FIG_MOCK=1`, the kit also sets each figure on a mock two-column page in `out/mock/`, which needs
  `pdflatex` with the IEEEtran class and the lipsum package. A build without it draws the same figures.

## Reading a Data File and a Keys File

Each drawn value in a data file carries its source: a record with the field in it, or the key of a number the paper
prints. A source of the form `ledger:<key>` names a printed number by its key in our table of printed numbers. The
index in `artifact/numbers` (`index.json` and `index.md`) gives each key's value, record and field. A keys file
lists, for every value a figure draws, what it is (`what`), the value (`value`) and its key (`key`) or record
(`source`).

## Source Paths and the Records Zip

A data file and a keys file name each record by its place in the records zip, a path that begins with `records/`.
A folder under `records/frozen/` holds a dated copy of records. Some data files also name records, notes and drafts
of the paper that the zip does not hold, or a record at the place it had before we froze it, whose copy is then under
`records/frozen/<copy>/` with the same rest of its path. For those, the data file itself holds the values the figure
draws.

## Figures, Data and Records

The table gives, for each figure, the paper's figure number and the records in the zip that its data file was
built from. A path with `<row>` stands for one folder per deployment row, and the count gives the rows the figure
reads.

| Figure file | Paper | Records in the zip |
|---|---|---|
| `fig_capacity` | Fig. 1 | `records/08_significance/probes/arch/runs/capacity_census.json`<br>`records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/capacity_counts.json` |
| `fig_toolchain` | Fig. 2 | none: a diagram of the toolchain's stages (Section III), no measured value |
| `fig_places` | Fig. 3 | `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_fig1_data/fig1_vit_block.json` |
| `fig_classes` | Fig. 4 | `records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/rows_r20/<row>/result.json` (2 rows)<br>`records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/tables/r20/t1/WI-6/table.json` |
| `fig_invariants` | Fig. 5 | none: a diagram of Theorem 1's invariants (Section IV), no measured value |
| `fig_replay` | Fig. 6 | `records/frozen/exactness_block_t1_20261124T165039/exactness_block/644b4c3bc5d7_r3/w13/summary.json` |
| `fig_exact` | Fig. 7 | `records/frozen/exactness_block_t1_20261124T165039/exactness_block/644b4c3bc5d7_r3/w12/screen/report.json` |
| `fig_invariance` | Fig. 8 | `records/frozen/routeb_budget_ladder_t1_20261120T183320/tiers/paper_locks/r20b/ladder/run/tables/t2/ladder_rows.jsonl`<br>`records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/rows_r20/<row>/result.json` (17 rows)<br>`records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/tables/r20/t1/WI-6/table.json`<br>`records/frozen/routeb_tint_supplement_t1_20261121T001425/e2e/routeb/tables/r20/t1/WI-7/table.json` |
| `fig_compare` | Fig. 9 | `records/hardware/multicore/r5_card_session_3/read/readings/expectations.json`<br>`records/hardware/multicore/r5_card_session_3/read/readings/readings.json` |
| `fig_card` | Fig. 10 | `records/hardware/multicore/r5_card_session_3/read/keys/keys.json`<br>`records/hardware/multicore/r5_card_session_3/read/readings/readings.json` |
| `fig_area` | Fig. 11 | `records/figures_v3/data/fig_invariance.json`<br>`records/tiers/fidelity/uarea/build_r6/w9/unit_facts.json`<br>`records/tiers/fidelity/uarea/build_r6/w9_paper_644b4c3bc5d7/k02/fig_area_unit_data_r2.json`<br>`records/frozen/routeb_tint_supplement_t1_20261121T001425/tiers/fidelity/uarea/build_r6/w9_paper_644b4c3bc5d7/k02/fig_area_unit_r2.json` |
| `fig_space` | Fig. 12 | `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/member_map.json`<br>`records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/regime_map.json`<br>`records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/summary.json` |
| `fig_regime` | Fig. 13 | `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/regime_map.json` |
| `fig_fronts` | Fig. 14 | `records/06_build/headline_v46_pairs/r3_nominal/b/pairs_r3b_record.json`<br>`records/tiers/paper_v6/fix5/study_build/reread/keys.json`<br>`records/tiers/paper_v6/fix5/study_build/reread/out/read/P0.json`<br>`records/tiers/paper_v6/fix5/study_build/reread_time/out/time_fronts.json` |
| `fig_chipprops` | Fig. 15 | `records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/cifar10/records_644b4c3bc5d7/cifar10_summary_r2.json`<br>`records/frozen/cl6_t1_20261120T183429/11_wave2/explore/cl6/mnist_644b4c3bc5d7/summary.json`<br>`records/frozen/w34_adc_sweep_t1_20261120T183547/11_wave2/explore/cl6/w34/points_644b4c3bc5d7/collect_r4_t1.json` |
| `fig_block` | Fig. 16 | `records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5/n2b/acceptance/price_n2b.softmax.O8.json`<br>`records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_fig1_data/fig1_vit_block.json`<br>`records/frozen/n2_cl7_t1_20261107T093743/11_wave2/n2_attention/r5_rerun_n2b512/certify_n2b.gate.softmax.O8.i512.json`<br>`records/frozen/provisional/11_wave2/explore/cl5/out/regime_map.json` |

Fig. 6 (spike replay) draws a schematic with illustrative values next to one measured count, and its keys file says
which is which.

## Data Builders

Eleven data files have their builder here: `scripts/build_fig_<name>_data.py` for `area`, `block`, `capacity`,
`chipprops`, `classes`, `exact`, `invariance`, `regime`, `replay`, `space` and `toolchain`. The data files of
Figs. 3, 9, 10 and 14 (`fig_places`, `fig_compare`, `fig_card`, `fig_fronts`) have no builder in this folder. A
builder names the records it reads by their places in the records zip and imports helper modules of our working
repository that this folder does not hold, so a reader does not run it. We publish the builders as provenance: each shows which field of which
record a drawn value comes from and how a derived value was computed. The figure scripts and `make_all.py` never
call them.
