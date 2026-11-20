"""[U-TGAUGE4] The placement rule's property test: generated layouts in both walk families, each walk's read equal to
the deployed program's executor element by element, or the construction refused by name.

``layout_trees.draw_tree`` draws a tree from a seed: producers of one to three perceptrons (or the encoding itself) at
random gauges, one value or one per channel, under a random tree of the toolchain's layout mappers (concatenations
along either axis, flattens, axis moves, gathers, stacks), read as the output, by a core, by a host op, or by a host
op then a core. The timing family runs the cascaded pair; the count family the Soft Core Mapping sequence with the
declared depth-balancing relays, under both LIF semantics. 600 trees per family, in chunks.
"""

from __future__ import annotations

import pytest

import unit.spiking.layout_trees as lt

TREES = 600
CHUNK = 50


@pytest.mark.parametrize("first", range(0, TREES, CHUNK))
@pytest.mark.parametrize("family", ["timing", "count"])
def test_every_generated_layout_reads_as_the_deployed_executor_or_refuses_by_name(family, first):
    compared, refused, off = 0, [], []
    for seed in range(first, first + CHUNK):
        rec = lt.read(seed, family)
        if "refused" in rec:
            refused.append(rec)
            continue
        compared += 1
        if rec["max_abs"] > 1e-6:
            off.append(rec)
    assert not off, off[:3]
    assert compared >= CHUNK - 2, refused  # the generator draws constructions the toolchain builds
