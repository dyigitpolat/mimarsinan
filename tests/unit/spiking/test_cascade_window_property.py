"""[U-TGAUGE5] The window rule's property test: generated per-cycle cascades with in-segment fan-out at two depths and
inner single-reader gaps (``cascade_trees``), built through the mapping in both walk families, each walk's read equal
to the deployed executor's element by element and each reading core's fold equal to its sources' placed gauges (the
fourth verify's m-5), or the construction refused by name (never the window refusal: the mapping arms the relays).

U-TGAUGE4's generator drew producers as chains over the encoding only, so no in-segment producer was read at two
depths, and built the timing family without relays: on U-TGAUGE4's tree this generator's timing family reads 17 of its
first 100 trees a step low. 240 trees per family, in chunks.
"""

from __future__ import annotations

import pytest

import unit.spiking.cascade_trees as ct

TREES = 240
CHUNK = 40
FOLD_TOLERANCE = 1e-5


@pytest.mark.parametrize("first", range(0, TREES, CHUNK))
@pytest.mark.parametrize("family", ["timing", "count"])
def test_every_generated_cascade_reads_as_the_deployed_executor_through_the_mapping(family, first):
    compared, refused, off, folds = 0, [], [], []
    for seed in range(first, first + CHUNK):
        rec = ct.read(seed, family)
        if "refused" in rec:
            refused.append(rec)
            continue
        compared += 1
        if rec["off"]:
            off.append(rec)
        if rec["fold_error"] > FOLD_TOLERANCE:
            folds.append(rec)
    assert not off, off[:3]
    assert not folds, folds[:3]
    assert not [r for r in refused if "UnbalancedFanInError" in r["refused"]], refused[:3]
    assert compared >= CHUNK - 2, refused  # the generator draws constructions the toolchain builds
