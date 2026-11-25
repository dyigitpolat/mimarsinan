# Regime Robustness to the Cost Constants

## What the Paper Reports

Section VI-B sorts points beyond capacity into regimes by their write-to-compute ratio, Eq. (11), as compute-dominated, mixed or write-dominated, with fixed thresholds of the system cost model at each corner (Fig. 13).
Over 0.2 to 5 times the assumed cost constants, 3,007 of 4,221 tested design points keep their regime (`robregime`, with `costrange` for the range).

## Setup

- Points: the tested design points, sorted into regimes as in Section VI-B.
- Range: the cost constants scaled within 0.2 to 5 times their assumed values (`costrange`, a setting the paper states in Section VI-B).
- Toolchain: the patched release, see `artifact/VERSIONS.md`.

## How It Was Run

1. We priced the tested design points with the system cost model (Section VI-C), with the cost constants scaled within the range above.
2. At every scaling, each point was sorted into its regime by the thresholds of Section VI-B.
3. We counted the points whose regime stays the same over the whole range, out of all tested points.

The regime map of Fig. 13, its thresholds and the explored points are documented in `artifact/experiments/explored_space/README.md`, and Fig. 13 is drawn by `artifact/figures/scripts/fig_regime.py` from `artifact/figures/data/fig_regime.json`.

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| none | the record of this study is not published | robregime |

The index of printed numbers in `artifact/numbers/` gives the value of `robregime` as the paper prints it and names `costrange` as a setting stated in the paper.

## Not Included

The run of this study, its rules fixed before the run, its scripts and its record are not in the public copy.
They are held back under the study's held-out rule, as are the records of the compiler comparison stored with them.
