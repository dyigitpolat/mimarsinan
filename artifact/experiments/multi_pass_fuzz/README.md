# Random Multi-Pass Programs

## What the Paper Reports

Section V-B and Fig. 7(a) report random multi-pass programs compared with their single-pass program ι under each semantics inside Theorem 1.
Under synchronized TTFS every program is compared, and under streamed, windowed and synchronized LIF and cascaded TTFS only part of them (keys `gfivesemanticsevery`, `gfivefuzzevery`, `gfivesemanticspart`, `gfivefuzzpart`).
Section VII reports the analytical TTFS run of the same programs (`ttfsanalytic`).

`artifact/numbers/index.md` gives each value with its record and field.

## Setup

- **Toolchain.** The patched release, whose records carry the label 644b4c3bc5d7 in their folder names (`artifact/VERSIONS.md` gives its commit). The field `code_state` of `summary.json` gives the source and test digests of the tree that ran.
- **Programs.** Random networks from the toolchain's test generators, with no dataset. The field `generators` of `summary.json` describes each family. The base family is a random branchy LIF network of 4 to 9 operations with skips, at most two host operations, layer widths 4, 8 or 16 and fan-in at most 24, on cores of 25 axons and 8 neurons. The value family holds integer value networks (MLP, convolutional or residual) with integer inputs from minus 3 to 3.
- **Jobs.** The field `jobs` lists nine jobs, each with its name (`job`), seed count (`seeds`), built programs (`rows`), multi-pass programs (`multipass`) and drawn ranges (`ranges`, among them the core counts `ranges/count`). The printed keys read the job `base_2000x7`: 2,000 seeds at 7 core counts.
- **References.** Each multi-pass program is compared at zero tolerance with the identity program of its tiled IR and with Theorem 1's identity program, the identity program of its IR at unbounded geometry.

## How It Was Run

1. `scripts/u1_fuzz_px.py` builds each random program at each core count, deploys it in passes and runs it under every semantics against both identity programs. It reads the generators from the toolchain's tests at the paper version (`tests/unit/mapping/residency/executed/vehicles.py` and `tests/unit/mapping/residency/executed/runs.py`) and builds programs through `scripts/window_rule.py`, as the toolchain's mapping builds them. `scripts/error_site.py` records where an error without a named rejection arose.
2. `scripts/value_fuzz.py` runs the integer value programs against their identity program and the int64 reference.
3. A chunk runner split each job into ranges of seeds and ran them in parallel. It is not published because it holds settings of our machines. The arguments of both scripts above are the toolchain tree, the output file, the first and last seed and the core counts.
4. `scripts/w13_summarize.py` reads the rows of all jobs and writes `summary.json` and a Markdown table of the same counts. It reads the list of jobs from a run manifest, which is not published because it holds internal working notes. The field `jobs` of `summary.json` repeats each job's settings, and `manifest_sha256` gives the manifest's digest.

## Records

Paths are places in the records zip. Each `records/frozen/<name>_<time>` folder is a frozen copy of one set of records.

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/frozen/exactness_block_t1_20261125T031754/exactness_block/644b4c3bc5d7_r3/w13/summary.json` | per job: seeds, programs, ranges and per semantics comparison counts | `gfivesemanticsevery`, `gfivefuzzevery`, `gfivesemanticspart`, `gfivefuzzpart`, `ttfsanalytic` |

## How to Read the Records

- `jobs/<i>/theorem_programs`: `N` multi-pass programs, `differ` those where any Theorem 1 comparison differs, `runs` the semantics compared, and `clopper_pearson_95` and `rule_of_three` the one-sided bounds.
- `jobs/<i>/runs/<semantics>/against_unbounded_identity` and `against_tiled_identity`: `N`, `differ` and `classes` (`equal`, or `absent` where the semantics was not run on a program).
- `jobs/<i>/runs/analytical_ttfs`: the analytical TTFS run, with `outside_theorem_by_name` true.
- `code_state`: the time and the source and test digests of the toolchain tree that ran.

## Not Included

The chunk runner, the run manifest, the per chunk rows, the Markdown table and the run logs are not published. The runner holds settings of our machines, and the manifest holds internal working notes. The summary above holds every count the paper prints. The authors can be asked for any of these.
