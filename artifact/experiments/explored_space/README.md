# Explored Space, Regime Map and Write Endurance

## What the Paper Reports

Section VI-A (Fig. 12) reports the explored space: random-weight networks on chip descriptions, the trained networks
whose checked deployments give accuracy to unbuilt points, and the points below the required service life. Section
VI-B (Fig. 13) sorts points beyond capacity into regimes by tiles per core and the write-to-compute ratio. The
write-endurance paragraph of Section VI-D and Fig. 15(b) give the largest weight overflow a non-volatile chip runs
without batching. Table III marks the cost axes each chip description is checked on, and Section I (Fig. 1) counts the
network and chip pairs whose weights exceed the chip's cells. One exploration run writes the records of all of these,
so they share this folder.

Printed keys (in `artifact/numbers/index.json`): `clfivenetworks`, `clfivechips`, `dsechipnames`, `dsenetworks`,
`dseconfigs`, `dseclassesadapted`, `capimagenet`, `clfiveaxessram`, `clfiveaxesneurram`, `clfiveaxeshermes`,
`clfiveaxesgeneric`, `clfiveaxesisaac`, `clfiveaxescompass`, `clfiveaxesisaacstudy`, `clfiveaxesvr`, `batchonelimit`,
`lifefloor`, `rate`, `rramendurance`, `rrampulses`, `pcmendurance`, `regimefree` and `regimewrite`. The keys
`floornvchips`, `floorunder`, `floorcrossk` and `flooracceptedk` read the figure data of Fig. 12 (`artifact/figures/`).
The endurance bands and the pulses per write are published values, and the index names their publications and pages.

## Setup

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Toolchain.** The patched release. The run record names it by its digest, beginning `644b4c3bc5d7`, in the field
  `head` of `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/record_644b4c3bc5d7/run.json`.
  `artifact/VERSIONS.md` maps these names to the toolchain repository.
- **Service life and inference rate of the write law.** `meta/assumptions/min_lifetime_hours` and
  `meta/assumptions/inference_rate_hz` of `records/frozen/provisional/11_wave2/explore/cl5/out/regime_map.json`.
- **Write reading at batch 1.** The endurance cycles, the chip's cells and the cells per weight of the reading behind
  `batchonelimit` are in `declared/endurance_cycles`, `declared/chip_cells` and `declared/cells_per_weight` of
  `records/tiers/paper_t9/vid_reading_t1_20261109T004712/record.json`, with `lifetime_hours` and `rate_hz`.
- **Regime thresholds.** `meta/regime_thresholds/capacity_free_max` and `meta/regime_thresholds/resident_dominant_min`
  of `records/frozen/cl5_exploration_t1_20261110T065255/11_wave2/explore/cl5/out_r5b/regime_map.json`, an earlier run
  of the same exploration whose thresholds the paper prints.
- **Networks and chips.** The field `points` of the regime map of record, each point naming its network, chip
  description, core count and batch size. `networks` and `chips` of `capacity_counts.json` list the pairs of the
  capacity count.

## How It Was Run

Each script reads the outputs of the one before it. The ladder runs the toolchain's own `scripts/scaling_ladder.py`
(toolchain repository at the patched release), which we do not copy.

1. `scripts/cl5_jobs.py` writes the list of runs of the priced tier and of the capacity tier.
2. `scripts/cl5_ladder.py` runs one entry of that list: the toolchain's scaling ladder with a declared weight grid.
3. `scripts/cl5_resolve.py` resolves each point's network identity and computation, and every candidate deployment of
   record, and writes `deployments_resolved.json`.
4. `scripts/plane_facts.py` compares executed and priced programs and labels where they can diverge.
5. A build script assembles the regime map and the capacity counts (`regime_map.json`,
   `capacity_counts.json`). It is not published, since it holds internal working notes.
6. `scripts/cl5b_census.py` writes the chip-first admission census and the endurance demonstration.
7. `scripts/r1001_hcm.py` and `scripts/r1001_record.py` build ResNet-1001 through hard core mapping beside its point
   priced by shape (`r1001_hcm.json`).
8. A deployment script deploys and checks networks at one eighth of the packed tiles. It is not published, since it
   holds internal working notes.
9. `scripts/cl5_summarize.py` and `scripts/cl5_tables.py` write `summary.json` and the report tables.
   `scripts/cl5_readings.py` holds the readings they share.
10. `scripts/cl5_outputs.py` writes the run record `run.json`. `scripts/skip_departures.py` marks entries of the list
    of runs that add nothing.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/regime_map.json` | every explored point with its regime, costs and lifetime | `clfivenetworks`, `clfivechips`, the `dse` keys |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/member_map.json` | trained networks and their random-weight counterparts | the `dse` keys |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/deployments_resolved.json` | checked deployments of record, resolved | the `dse` keys |
| `records/excerpts/exploration_agreement_rule.md` | the agreement rule that counts checked deployments, word for word | the `dse` keys |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/capacity_counts.json` | pairs whose weights exceed the chip's cells | `capimagenet` |
| `records/excerpts/exploration_cost_marks.md` | the lines of the report tables with Table III's cost marks, word for word | the `clfiveaxes` keys |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/summary.json` | summary tables of the regime map | none directly |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/out_644b4c3bc5d7/r1001_hcm.json` | ResNet-1001 through hard core mapping | none directly |
| `records/frozen/cl5_exploration_t1_20261122T161849/11_wave2/explore/cl5/record_644b4c3bc5d7/run.json` | the run record | none directly |
| `records/frozen/provisional/11_wave2/explore/cl5/out/regime_map.json` | the write law's assumptions | `lifefloor`, `rate`, `batchonelimit` |
| `records/tiers/paper_t9/vid_reading_t1_20261109T004712/record.json` | the write reading at batch 1 | `batchonelimit` |
| `records/frozen/cl5_exploration_t1_20261110T065255/11_wave2/explore/cl5/out_r5b/regime_map.json` | the regime thresholds | `regimefree`, `regimewrite` |

## How to Read the Records

A point of `regime_map.json` carries its network, chip description, cores and batch size, its estimated costs and its
regime label. The field `meta` holds the assumptions and thresholds the points were sorted by. The three regime maps
are large files and are best read as a stream. In
`capacity_counts.json`, `probe_estimate/exceed` of `probe_estimate/pairs` gives the count of Section I.

## Not Included

The intermediate outputs of items 3, 4, 6 and 8 of the list above (the resolution of each point, the comparison of
executed and priced programs, the admission census and the deployments at one eighth of the tiles), the full report
tables, the full agreement rule, the build script and the deployment script are not in the public copy, since they hold internal working
notes or settings of our machines. The two excerpts above quote the lines that printed values read. The authors can be
asked for the rest.
