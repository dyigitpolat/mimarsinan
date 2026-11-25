# Area Estimates against Published Chips and Passes

## What the Paper Reports

Section III fits the share $\rho$ of the area model, Eq. (4), for digital spiking cores over 5 chips, refitted without each chip it estimates (`areanspiking`).
Section V-D ("Area against published silicon") and Fig. 11(a) and (b) compare leave-one-out area estimates with published areas.
Table III's chip descriptions draw on 3 of the estimated chips and macros only through the area model (`areaindep`), and the estimates of 4 use their own cell or converter area (`areaownpriced`).
With other papers' areas instead, 8 remain estimable (`areastrictpriced`), the worst within a factor of 2.8 (`areastrict`), where Fig. 11(a) shows 2.31 (`areaworst`).
SRAM array estimates are within 14.2% on 5 chips and macros (`areaarrays`, `areaarraychips`), 2 using their own cell area (`areaownsram`), and the NeuRRAM and HERMES-64 arrays err by 1.75 and 1.48 times (`areaarraynv`).
Section V-D ("Area against passes") and Fig. 11(c) give ResNet-18's area against scheduled passes (`arearesident`, `arealow`, `arealowpasses`, `areapick`, `areapickpasses`) and DenseNet-BC's passes at every budget (`areadensepasses`, `areadenseflat`).

## Setup

- Chips with a published area: those cited in Fig. 11's caption. For each chip, the published area, the estimate, their ratio and the place of the value in its source paper are in `fig_area_unit_r2.json`, field `parity` (entries `published_mm2`, `estimate_mm2`, `ratio` and `published_source`).
- Published figures: the area model reads its published areas and breakdowns from the toolchain itself, each with its source and the quoted passage, in `src/mimarsinan/deployment_record/platform_physics/area_rule/data/evidence/published_figures.json` and `published_points.json` beside it.
- Rejected chips: the area model rejects Loihi 2 and IMAGINE (Section V-D). The share fit keeps TrueNorth, ODIN, MorphIC, ReckOn and Loihi and drops Loihi 2 by name (`fig_area_unit_data_r2.json`, fields `share_fit/kept` and `share_fit/dropped`).
- SRAM arrays: compared on TrueNorth, ODIN, MorphIC, IMAGINE and CAP-RAM (`fig_area_unit_data_r2.json`, field `arrays/sram_labels`).
- Panel (c): the explored deployments of ResNet-18, DenseNet-BC and the other networks of the panel on the generic SRAM description of Table III (the record of panel (c), field `facts/ladder`, one entry per network with its `label`).
- Toolchain: the patched release. The records name it by a digest of its source, `644b4c3bc5d7`, which appears in the folder `w9_paper_644b4c3bc5d7`. `artifact/VERSIONS.md` gives its commit.

## How It Was Run

The area model is part of the toolchain. At the paper version it lives in the toolchain repository under `src/mimarsinan/deployment_record/platform_physics/area_rule/` (`rule.py` holds the core equation, `pricing.py` and `typerule.py` the pricing of a deployment record), and `src/mimarsinan/deployment_record/quantities/from_record.py` reads a deployment record. We do not copy them here.

1. `scripts/w9/e01_configs.py` collects the explored configurations (target, network, budget, cores, passes and weight bits) from the exploration's records.
2. `scripts/w9/p01_price.py` prices each chip description and each deployment record through the toolchain's area functions.
3. `scripts/w9/p02_unit.py` computes the leave-one-out estimates, the share fit and the array comparison from the toolchain's published figures and writes `unit_facts.json`. `scripts/w9/c01_side.py` adds side readings of the explored rows.
4. A build script assembles the figure record and its data file, copying every value by its field and checking it, and `scripts/w9_r2/displays.py` holds the rule by which a value is printed. The same build on the patched release wrote the records under `w9_paper_644b4c3bc5d7/k02/`. The build script is not published, since it holds internal working notes.
5. Fig. 11 is drawn by `artifact/figures/scripts/fig_area.py` from `artifact/figures/data/fig_area.json`, which `artifact/figures/scripts/build_fig_area_data.py` builds from the records below.

The scripts take their input and output paths on the command line, and the build script loads `scripts/w9_r2/displays.py` by its path.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/tiers/fidelity/uarea/build_r6/w9_r2/fig_area_unit_r2.json` | the record of Fig. 11 with its facts and the comparison per chip | areanspiking, areaindep, areaarraychips, areaownsram, areaarraynv |
| `records/tiers/fidelity/uarea/build_r6/w9_r2/fig_area_unit_data_r2.json` | headline counts, factors and the print bounds | areaownpriced, areastrictpriced, areastrict, areaworst, areaarrays |
| `records/frozen/routeb_tint_supplement_t1_20261121T001425/tiers/fidelity/uarea/build_r6/w9_paper_644b4c3bc5d7/k02/fig_area_unit_r2.json` | the same record on the patched release, the copy that the index of printed numbers cites for panel (c) | arearesident, arealow, arealowpasses, areapick, areapickpasses, areadensepasses, areadenseflat |
| `records/tiers/fidelity/uarea/build_r6/w9_paper_644b4c3bc5d7/k02/fig_area_unit_data_r2.json` | the data file of the record above | |
| `records/tiers/fidelity/uarea/build_r6/w9/unit_facts.json` | the area model's figures per chip | |

## How to Read the Records

In `fig_area_unit_data_r2.json`, `a_headline` holds the leave-one-out comparison (`n_priced`, `own_unit_priced`, `independent`, and `strict_unit` for the estimates made with other papers' areas).
`print_bounds/<key>` gives each printed value with the reading it rounds.
`arrays` compares the array group of each chip's stack, estimate over published.
In `fig_area_unit_r2.json`, `parity` holds one entry per chip, `facts/share_fit` the share fit and `facts/ladder/<row>` each network's area and passes per budget: `resident` is the single-pass deployment, `executed/<budget>` a multi-pass one.
Ratios are estimate over published area.
A record's `sources` field names the files it was built from by their places in the records zip, and the table of published areas it names is not in the zip (see below).

## Not Included

The table of published areas from which the figure record was built is not in the public copy, because it points to page images of other authors' papers. The toolchain's published figures and the field `parity` give every published value with its source.
The build script of the figure record is not published, as it holds internal working notes. The shell wrappers that ran these scripts are not published, as they hold settings of our machines.
The authors can be asked for either.
