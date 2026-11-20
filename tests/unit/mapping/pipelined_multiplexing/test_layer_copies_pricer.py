"""[thru:w1] The pricer of a program with layer copies (PL/thru/design.md (c) 'Pricer', (e) 'Pricer'), hand-counted on
the planner's two-level chain (``test_layer_copies_planner``: 20 x 20 instances, one per 32 x 32 core) on 6 cores,
priced on the candidate plane as the exploration prices it (``collect_noc_fragments``, ``candidate_program_facts``,
the period census). Copies change the passes and the cores each pass loads, never the pricer's form: the passes and
barriers are the schedule census's, the steps one evaluation per pass, and the copies are written cores of their
group's first pass, so the write law counts their bytes as it counts any written core.

Hand count, copies (4, 2): passes 3 (pass 0 runs 4 instances, pass 1 runs 6, pass 2 runs 2), barriers 2, at T = 1
each pass T + L + 1 = 2 steps (L = 0 on a wavefront pass), so 6 steps; the first pass writes the 6 layout cores of
400 cells each, 200 bytes at 4 bits, so 1,200 bytes; the evaluated cores are the 12 instances' (4,800 cells). One copy
per bank: passes 9, barriers 8, 18 steps, 2 written cores (400 bytes), the same 12 evaluated cores."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics, candidate_program_facts

from .test_layer_copies_planner import chain

CORES = 6
BYTES_PER_CORE = 20 * 20 * 4 // 8


def _program(segments: int = 1):
    """The chain once per segment (each segment its own banks), its producers and its runs."""
    one, reads_one = chain()
    specs, reads, runs = [], [], []
    for s in range(segments):
        base = len(specs)
        specs += [LayoutSoftCoreSpec(20, 20, latency_tag=x.latency_tag, segment_id=s, bank_id=x.bank_id + 10 * s,
                                     name=f"s{s}{x.name}") for x in one]
        reads += [frozenset(base + p for p in r) for r in reads_one]
        runs += [s] * len(one)
    return specs, reads, runs


def _priced(copies, segments: int = 1):
    specs, reads, runs = _program(segments)
    census = SimpleNamespace(pair_wires={(p, c): 20 for c, ps in enumerate(reads) for p in sorted(ps)},
                             spec_runs=tuple(runs))
    frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(32, 32, CORES)], census=census,
                                 allow_scheduling=True, allow_neuron_splitting=False, allow_coalescing=False,
                                 max_schedule_passes=64, runs=runs, position_multiplexing=True,
                                 pipelined_multiplexing=True, layer_replication=copies)
    facts = candidate_program_facts(specs, frag, StageSemantics(), timesteps=1, weight_bits=4,
                                    pass_transfer="collapse")
    return frag, facts


class TestTheHandCountedProgram:
    @pytest.mark.parametrize("copies,passes,steps", [(True, 3, 6), (None, 9, 18)], ids=["copies", "one_copy"])
    def test_its_passes_barriers_and_steps(self, copies, passes, steps):
        frag, facts = _priced(copies)
        assert frag.schedule_census == (passes, passes - 1)
        assert facts.programming.schedule_census == (passes, passes - 1)
        assert facts.latency_steps == steps

    def test_the_copies_are_written_cores_of_the_group_s_first_pass(self):
        _frag, facts = _priced(True)
        programming = facts.programming
        assert programming.pass_cores == (6, 6, 2)
        assert (programming.reprogrammed_cores, programming.reprogram_passes) == (6, 1)
        assert programming.reprogrammed_cells == (400,) * 6
        assert programming.reprogrammed_bytes == 6 * BYTES_PER_CORE == 1200
        assert (programming.segment_cores, programming.committed_cells) == (12, 12 * 400)

    def test_one_copy_writes_one_core_per_bank_and_evaluates_the_same_cores(self):
        _frag, facts = _priced(None)
        programming = facts.programming
        assert programming.pass_cores == (2, 1, 2, 1, 2, 1, 2, 1, 1)
        assert (programming.reprogrammed_cores, programming.reprogrammed_bytes) == (2, 2 * BYTES_PER_CORE)
        assert (programming.segment_cores, programming.committed_cells) == (12, 12 * 400)


class TestTheWriteLaw:
    """Two segments of the chain on the 6 cores: with copies each segment's first pass writes all 6 cores, 12 written
    cores do not fit the chip at once, so the period census rewrites both heads every period (2 x 1,200 bytes); one
    copy writes 2 cores per segment, 4 fit the chip, and the program is loaded once (a flat fit rewrites nothing)."""

    def test_the_copies_bytes_are_rewritten_every_period(self):
        _frag, facts = _priced(True, segments=2)
        census = period_census(facts.passes, weight_bits=4, capacity={(32, 32): CORES}).census
        assert [p.payload_bytes for p in census.passes] == [1200, 0, 0, 1200, 0, 0]
        assert [p.written_cells_backing_store for p in census.passes] == [2400, 0, 0, 2400, 0, 0]

    def test_one_copy_fits_the_chip_and_is_loaded_once(self):
        _frag, facts = _priced(None, segments=2)
        census = period_census(facts.passes, weight_bits=4, capacity={(32, 32): CORES}).census
        assert len(census.passes) == 18 and sum(p.payload_bytes for p in census.passes) == 0
