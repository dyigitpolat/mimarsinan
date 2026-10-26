"""[P1 R2a] A placed mapping keeps only the cores it uses: ``HardCoreMapping.map`` drops its reference to the pool it
drew from once placement is done, so a scheduled pass (a fresh pool of every declared core per pass) retains no unused
core, and neither does a stage of the unscheduled build (one pool shared across stages, held by the build until it
returns). The pool itself is the caller's: placement still takes its cores from the caller's list in place, so the
next stage of an unscheduled build draws from what is left, in order.

Before, each pass's mapping kept the unused rest of its pool (``unused_cores``), which nothing reads after placement:
765 to 861 B per pool core per pass (``11_wave2/sched_memory/scout.md``), 12.0 to 13.6 GiB at ResNet-50 224 F.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping, SoftCore
from mimarsinan.mapping.packing.softcore.soft_core_mapper import SoftCoreMapping

from unit.chip_simulation.value_run.integer_programs import build_integer_program
from unit.mapping.packing.test_neuron_split_before_coalescing import SHAPE, _wide_cnn

SMALL, LARGE = 4000, 16000


@pytest.fixture(scope="module")
def program():
    """The wide CNN as an integer value program on 16 x 8 cores (several scheduled passes, neuron splits)."""
    return build_integer_program(_wide_cnn(), SHAPE, num_classes=5, max_axons=16, max_neurons=8,
                                 allow_coalescing=True)


def _neural(hybrid) -> list:
    return [s for s in hybrid.stages if s.kind == "neural"]


def _mappings(hybrid) -> list:
    """Every HardCoreMapping of the program, the retimed level stages' included."""
    out = []
    for stage in _neural(hybrid):
        out.append(stage.hard_core_mapping)
        out.extend(level.hard_core_mapping for level in getattr(stage, "retimed_level_stages", None) or ())
    return out


def _built_and_retained(build):
    """The program ``build()`` returns and the bytes it retains (tracemalloc over the build, the program alive)."""
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        hybrid = build()
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    return hybrid, retained


def _build(program, count: int, passes):
    return program.packed_mapping(axons=16, neurons=8, count=count, split=True, coalescing=True, passes=passes)


class TestAScheduledPassKeepsTheCoresItUses:
    def test_no_pass_keeps_an_unused_core(self, program):
        hybrid = _build(program, LARGE, 64)
        assert len(_neural(hybrid)) >= 2
        for hcm in _mappings(hybrid):
            assert hcm.unused_cores == []
            assert hcm.cores

    def test_retained_bytes_do_not_grow_with_the_declared_count(self, program):
        small, r_small = _built_and_retained(lambda: _build(program, SMALL, 64))
        large, r_large = _built_and_retained(lambda: _build(program, LARGE, 64))
        passes = len(_neural(small))
        assert passes == len(_neural(large)) >= 2
        slope = (r_large - r_small) / ((LARGE - SMALL) * passes)
        assert slope < 50, f"{slope:.1f} B retained per pool core per pass"


class TestAnUnscheduledStageKeepsTheCoresItUses:
    def test_no_stage_keeps_an_unused_core(self, program):
        hybrid = _build(program, LARGE, None)
        for hcm in _mappings(hybrid):
            assert hcm.unused_cores == []

    def test_retained_bytes_do_not_grow_with_the_declared_count(self, program):
        small, r_small = _built_and_retained(lambda: _build(program, SMALL, None))
        large, r_large = _built_and_retained(lambda: _build(program, LARGE, None))
        assert len(_neural(small)) == len(_neural(large))
        slope = (r_large - r_small) / (LARGE - SMALL)
        assert slope < 50, f"{slope:.1f} B retained per pool core"


def _soft_mapping(*shapes):
    soft = SoftCoreMapping()
    for i, (axons, neurons) in enumerate(shapes):
        core = SoftCore(np.ones((axons, neurons)), [SpikeSource(0, j, is_input=True) for j in range(axons)], id=i)
        core.latency = 0
        soft.cores.append(core)
    return soft


def test_placement_draws_from_the_callers_pool_in_place():
    """The pool a caller passes stays the caller's: placement takes its cores from that list, in order, and the
    mapping keeps no reference to it, so a second mapping on the same list (the next stage of an unscheduled build)
    draws the cores the first one left."""
    pool = [HardCore(4, 4) for _ in range(6)]
    order = [id(core) for core in pool]
    first = HardCoreMapping(pool)
    first.map(_soft_mapping((4, 4), (3, 2)))
    assert first.unused_cores == [] and len(first.cores) == 2
    assert [id(core) for core in pool] == [i for i in order if i not in {id(c) for c in first.cores}]
    second = HardCoreMapping(pool)
    second.map(_soft_mapping((2, 4)))
    assert second.unused_cores == [] and len(pool) == 3
    assert not {id(c) for c in second.cores} & {id(c) for c in first.cores}
