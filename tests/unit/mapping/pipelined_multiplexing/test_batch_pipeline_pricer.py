"""[thru:w3] The pricer of the batch pipeline (PL/thru/design.md (c) 'Pricer', (e) 'Pricer'): a run of a program of B
input slots serves B inputs, and a batch of K inferences runs ceil(K / B) rounds, so every count a plane states per run
(passes, barriers, steps, core initializations, events, carried bytes) times ceil(K / B) / K is the count per inference,
applied only where ``batch_pipeline_slots`` is declared. The live carry is counted per input slot and node, and a peak
above a declared ``carry_buffer_bytes`` is refused by name. The latency gains one term, an input's span from its first
pass to its last (``latency_input_span_s``); the throughput's form is unchanged (1 over the time per inference).

Hand count on the copies' chain of ``test_layer_copies_pricer`` (20 x 20 instances, one per 32 x 32 core, 6 cores,
copies (4, 2), 4-bit weights, T = 1, collapse), priced on the candidate plane as the exploration prices it. Per run at
B absent, 2 and 4: passes 3, 5 and 9 (barriers one fewer), steps two per pass (T + L + 1, L = 0 on a wavefront pass),
evaluated cores per pass (4, 6, 2), (4, 6, 6, 6, 2) and (4, 6, 6, 6, 6, 6, 6, 6, 2), 400 cells each, and per input 8
carried level-0 values of 20 elements (20 B each under collapse at T = 1), at most 4 of them live at once (80 B). An
input's span is 3 passes (6 steps, 2 barriers) and its evaluated cores 16 at B 2 (input a: 4 + 6 + 6) and 18 at B 4
(input b: 6 + 6 + 6)."""

from __future__ import annotations

from fractions import Fraction
from math import ceil
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics, candidate_program_facts

from .test_layer_copies_pricer import CORES, _program

#: Per B (None: absent) the run's passes, evaluated cores and carried values, and an input's span (passes, cores).
RUN = {None: dict(passes=3, cores=12, carried=8), 2: dict(passes=5, cores=24, carried=16),
       4: dict(passes=9, cores=48, carried=32)}
SPAN = {2: dict(steps=6, syncs=2, cores=16), 4: dict(steps=6, syncs=2, cores=18)}
PER_RUN = ("pass_count", "sync_count", "latency_steps", "segment_cores", "macs", "carried_raster_bytes",
           "carry_out_bytes", "carry_in_bytes")


def _platform(b, k, **extra):
    platform = {"cores": [{"max_axons": 32, "max_neurons": 32, "count": CORES}], "weight_bits": 4,
                "simulation_steps": 1, **extra}
    if k is not None:
        platform["inference_batch_K"] = k
    if b is not None:
        platform.update(batch_pipeline_slots=b, carry_input_slots=b)
    return platform


def _facts(b, segments: int = 1):
    specs, reads, runs = _program(segments)
    census = SimpleNamespace(pair_wires={(p, c): 20 for c, ps in enumerate(reads) for p in sorted(ps)},
                             spec_runs=tuple(runs))
    slots = {} if b is None else {"batch_pipeline_slots": b, "carry_input_slots": b}
    frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(32, 32, CORES)], census=census,
                                 allow_scheduling=True, allow_neuron_splitting=False, allow_coalescing=False,
                                 max_schedule_passes=64, runs=runs, position_multiplexing=True,
                                 pipelined_multiplexing=True, layer_replication=True, **slots)
    return candidate_program_facts(specs, frag, StageSemantics(), timesteps=1, weight_bits=4, pass_transfer="collapse")


#: A host census of no host work (every layer on chip), so the headline prices the chip's time.
NO_HOST = dict(host_macs=0, onchip_macs=12 * 400, host_params=0, onchip_params=3 * 400, compute_op_count=0)


def _quantities(b, k, host=None, **extra):
    facts = _facts(b)
    context = candidate_context_from_platform(_platform(b, k, **extra), latency_steps=facts.latency_steps,
                                              programming=facts.programming, carry=facts.carry, passes=facts.passes,
                                              **(host or {}))
    return from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                          context=context)


def _per_run(b) -> dict:
    run = RUN[b]
    return {"pass_count": run["passes"], "sync_count": run["passes"] - 1, "latency_steps": 2 * run["passes"],
            "segment_cores": run["cores"], "macs": 400 * run["cores"], "carried_raster_bytes": 20 * run["carried"],
            "carry_out_bytes": 20 * run["carried"], "carry_in_bytes": 20 * run["carried"]}


class TestTheHandCountedRun:
    @pytest.mark.parametrize("b", [None, 2, 4])
    def test_its_counts_per_run(self, b):
        """At K = B one round serves the batch, so the counts per inference are the run's divided by B."""
        k = 1 if b is None else b
        q = _quantities(b, k)
        want = {key: Fraction(value, 1 if b is None else b) for key, value in _per_run(b).items()}
        assert {key: q.get(key).value for key in PER_RUN} == pytest.approx({key: float(v) for key, v in want.items()})

    @pytest.mark.parametrize("b", [2, 4])
    @pytest.mark.parametrize("k", [1, 4, 5])
    def test_every_count_per_run_times_the_rounds_per_inference(self, b, k):
        q = _quantities(b, k)
        factor = Fraction(ceil(k / b), k)
        assert {key: q.get(key).value for key in PER_RUN} == pytest.approx(
            {key: float(value * factor) for key, value in _per_run(b).items()})

    @pytest.mark.parametrize("k", [1, 4, 5])
    def test_undeclared_the_counts_are_one_input_s_whatever_k(self, k):
        q = _quantities(None, k)
        assert {key: q.get(key).value for key in PER_RUN} == pytest.approx(
            {key: float(value) for key, value in _per_run(None).items()})
        assert not any(q.has(key) for key in ("input_span_steps", "input_span_syncs", "input_span_cores"))

    def test_the_hand_counted_values_at_b_two_and_four(self):
        """Written out: passes per inference 5, 2.5, 3 at B 2 and 9, 2.25, 3.6 at B 4 (K 1, 4, 5)."""
        got = {(b, k): _quantities(b, k).get("pass_count").value for b in (2, 4) for k in (1, 4, 5)}
        assert got == pytest.approx({(2, 1): 5, (2, 4): 2.5, (2, 5): 3, (4, 1): 9, (4, 4): 2.25, (4, 5): 3.6})

    def test_a_declared_b_without_k_withholds_every_count_per_run_by_name(self):
        q = _quantities(2, None)
        for key in PER_RUN:
            assert not q.has(key) and "inference_batch_K" in q.withheld_reason(key), key


class TestTheLiveCarry:
    @pytest.mark.parametrize("b", [2, 4])
    def test_each_input_s_values_are_carried_and_counted_per_slot_and_node(self, b):
        facts = _facts(b)
        assert facts.carry["carried_wires"] == RUN[b]["carried"]
        assert facts.carry["carried_bytes"] == 20 * RUN[b]["carried"]
        assert facts.carry["peak_live_bytes"] == facts.carry["onchip_peak_live_bytes"] == 80

    def test_a_peak_above_the_declared_buffer_is_refused_by_name(self):
        refused = _quantities(2, 4, carry_buffer_bytes=79)
        assert not refused.has("carry_onchip_bytes")
        assert "carry_buffer_bytes 79" in refused.withheld_reason("carry_onchip_bytes")
        held = _quantities(2, 4, carry_buffer_bytes=80)
        assert held.get("carry_onchip_bytes").value == pytest.approx(2 * 320 * Fraction(1, 2))


class TestTheInputSpan:
    @pytest.mark.parametrize("b", [2, 4])
    @pytest.mark.parametrize("k", [1, 5])
    def test_an_input_s_span_is_stated_per_input_never_per_round(self, b, k):
        q = _quantities(b, k)
        assert {key: q.get(f"input_span_{key}").value for key in ("steps", "syncs", "cores")} == SPAN[b]

    def test_the_span_term_beside_the_headline_and_the_throughput_s_form(self):
        physics = get_platform_physics("generic_estimated_22nm")
        q = _quantities(2, 4, host=NO_HOST)
        pricing = price_absolute(q, physics)
        terms = {t.name: t for t in pricing.terms}
        span = terms["latency_input_span_s"]
        cycle, sync = physics.band("t_cycle").nominal, physics.band("t_sync_barrier").nominal
        assert not physics.has("t_core_init")  # the profile states no core init: the span's cores are named unpriced
        assert span.value == pytest.approx(6 * cycle + 2 * sync, rel=1e-12)
        assert "input_span_cores" in span.source
        e2e, throughput = terms["e2e_latency_s"], terms["throughput_inferences_s"]
        assert throughput.value == pytest.approx(1.0 / e2e.value, rel=1e-12)
        from unit.deployment_record.test_ratchets_uenergy import time_rule_failures

        assert not time_rule_failures(pricing.terms)  # a reading of e2e's passes: no static leg of its own

    def test_undeclared_no_span_term(self):
        pricing = price_absolute(_quantities(None, 4, host=NO_HOST), get_platform_physics("generic_estimated_22nm"))
        assert "e2e_latency_s" in {t.name for t in pricing.terms}
        assert "latency_input_span_s" not in {t.name for t in pricing.terms}
        assert "latency_input_span_s" not in {r.name for r in pricing.refusals}


class TestTheWriteLaw:
    """Copies are written cores of their group's first pass with or without the batch pipeline: two segments of the
    chain at B 2 rewrite each head's 6 cores (1,200 B) once per period, the other passes resident."""

    def test_the_copies_bytes_once_per_period_at_b_two(self):
        facts = _facts(2, segments=2)
        census = period_census(facts.passes, weight_bits=4, capacity={(32, 32): CORES}).census
        payload = [p.payload_bytes for p in census.passes]
        assert len(payload) == 10 and payload == [1200, 0, 0, 0, 0, 1200, 0, 0, 0, 0]
