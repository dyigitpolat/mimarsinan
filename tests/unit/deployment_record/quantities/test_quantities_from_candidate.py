"""A candidate layout yields static quantities; the modeled ones say they are modeled."""

import pytest

from mimarsinan.deployment_record.quantities.from_candidate import (
    CandidateQuantityContext,
    from_candidate,
)
from mimarsinan.deployment_record.quantities.probe import probe_quantities
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS

from unit.deployment_record.record_fixtures import make_layout


def _context(**over):
    kwargs = dict(
        timesteps=32,
        activity_factor=0.05,
        weight_bits=4,
        tiles=2,
        neurons_physical=768,
        axons_physical=768,
        host_macs=1000,
        onchip_macs=14000, cells_committed=14000,
        host_params=100,
        onchip_params=900,
    )
    kwargs.update(over)
    return CandidateQuantityContext(**kwargs)


def test_layout_quantities_are_static_facts_of_the_packing():
    quantities = from_candidate(
        layout=make_layout(),
        chip_param_capacity=196608.0,
        total_params=1000.0,
        host_side_segment_count=1,
        context=_context(),
    )
    assert quantities.get("pass_count").value == 2.0
    assert quantities.get("sync_count").value == 1.0
    assert quantities.get("cells_physical").value == 196608.0
    assert quantities.get("total_params").value == 1000.0
    assert quantities.get("pass_count").provenance == "static"
    # [E2] Allocation is a fact of the PASS structure, not of the layout stats:
    # a layout without one claims no allocation rather than reporting the
    # declared chip's core count under the record's name for used cores.
    assert not quantities.has("cores_allocated")


def test_the_programming_census_rides_in_as_the_allocation_and_payload():
    """[U-N] The allocation is its own context field (the census's per-pass cores summed, a wavefront head's dark
    cores included), never the segment cores under a second name: here 9 allocated, 7 initialized."""
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=1.0, total_params=None,
        host_side_segment_count=None,
        context=_context(
            segment_cores=7, cores_allocated=9, reprogrammed_cores=3,
            reprogrammed_bytes=42560, reprogram_passes=1,
        ),
    )
    assert quantities.get("cores_allocated").value == 9.0
    assert quantities.get("segment_cores").value == 7.0
    assert quantities.get("reprogrammed_cores").value == 3.0
    assert quantities.get("reprogrammed_bytes").value == 42560.0
    assert quantities.get("reprogram_passes").value == 1.0


def test_the_candidate_context_allocates_the_census_per_pass_cores():
    """[U-N] The census states its segment cores (dark cores out) and its per-pass cores (every core a pass loads):
    the context's allocation is the per-pass cores summed, the record's ``cores_allocated``."""
    from types import SimpleNamespace

    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform

    census = SimpleNamespace(segment_cores=7, committed_cells=70, reprogrammed_cores=4, reprogrammed_bytes=None,
                             reprogram_passes=1, pass_cores=(4, 3, 2), reprogrammed_cells=(10, 10, 10, 10))
    context = candidate_context_from_platform({}, programming=census)
    assert (context.cores_allocated, context.segment_cores) == (9, 7)
    assert candidate_context_from_platform({}).cores_allocated is None


def test_context_declarations_ride_in_as_static():
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=1.0, total_params=None,
        host_side_segment_count=None, context=_context(),
    )
    assert quantities.get("timesteps").value == 32.0
    assert quantities.get("weight_bits").value == 4.0
    assert quantities.get("tiles").value == 2.0
    assert quantities.get("neurons_physical").value == 768.0
    assert quantities.get("axons_physical").value == 768.0
    assert quantities.get("host_macs").value == 1000.0
    assert quantities.get("onchip_macs").value == 14000.0
    assert quantities.get("total_macs").value == 15000.0, "derived when both halves exist"
    assert quantities.get("onchip_params").value == 900.0


def test_synaptic_events_are_modeled_from_the_activity_assumption():
    """The EDA switching-activity discipline: candidate spike-dependent energy rests on
    a DECLARED activity factor, and the provenance says so."""
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=1.0, total_params=None,
        host_side_segment_count=None, context=_context(),
    )
    events = quantities.get("synaptic_events")
    assert events.value == pytest.approx(14000 * 32 * 0.05)
    assert events.provenance == "modeled"


def test_no_activity_factor_means_no_event_claim():
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=1.0, total_params=None,
        host_side_segment_count=None, context=_context(activity_factor=None),
    )
    assert not quantities.has("synaptic_events")


def test_a_layoutless_candidate_keeps_only_declarations():
    """An area-only search never packs: capacity and context survive, packing facts
    do not."""
    quantities = from_candidate(
        layout=None, chip_param_capacity=196608.0, total_params=None,
        host_side_segment_count=None, context=_context(),
    )
    assert quantities.get("cells_physical").value == 196608.0
    assert quantities.get("neurons_physical").value == 768.0
    for key in ("cores_allocated", "pass_count", "sync_count"):
        assert not quantities.has(key), key


def test_what_a_candidate_never_claims():
    """Measured-only quantities must be absent at candidate time, never estimated
    silently: hop counts need a placement, spike counts need an execution."""
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=1.0, total_params=1.0,
        host_side_segment_count=0, context=_context(),
    )
    # [R1] ``cells_used``/``macs`` moved OUT of this list: the committed
    # rectangle is a static fact of the planned pass structure, and the event
    # model multiplies it — they are claimed when the structure is known.
    for key in ("noc_total_hops", "noc_total_packets", "total_spikes",
                "boundary_events", "host_ops_s",
                "reprogrammed_bytes", "connectivity_entries"):
        assert not quantities.has(key), key


def test_an_empty_context_still_yields_the_packing_facts():
    quantities = from_candidate(
        layout=make_layout(), chip_param_capacity=10.0, total_params=None,
        host_side_segment_count=None, context=CandidateQuantityContext(),
    )
    assert quantities.get("pass_count").value == 2.0
    assert not quantities.has("timesteps")


def test_the_probe_covers_every_catalog_key_with_positive_values():
    """Capability probes ask 'could this axis ever answer' — every key present,
    strictly positive so no availability predicate can be tripped by a zero."""
    probe = probe_quantities()
    for key in QUANTITY_SPECS:
        assert probe.has(key), key
        assert probe.get(key).value > 0.0, key


def test_the_pass_structures_schedule_census_is_the_candidates():
    """[U-PP P-1, SD-13 SDB-B2] The pass structure that supplies the latency steps (the fragments' programming census)
    states the schedule census, so layout statistics planned on other segments (here 2 passes, 1 sync) never answer
    for it; a single pool's census is (0, 0), as its record's; without a pass structure the statistics answer."""
    def census(**over):
        q = from_candidate(layout=make_layout(), chip_param_capacity=1.0, total_params=None,
                           host_side_segment_count=None, context=_context(**over))
        return q.get("pass_count").value, q.get("sync_count").value

    assert census(pass_count=5, sync_count=3) == (5.0, 3.0)
    assert census(pass_count=0, sync_count=0) == (0.0, 0.0)
    assert census() == (2.0, 1.0)


def test_the_candidate_context_reads_the_programming_census_schedule():
    """[U-PP P-1] The programming census carries its fragments' schedule census into the context; a census that
    states none (a duck-typed one) leaves the context's absent."""
    from types import SimpleNamespace

    from mimarsinan.deployment_record.build.payload_sizes import candidate_programming_census
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform

    base = dict(segment_cores=7, committed_cells=70, reprogrammed_cores=4, reprogrammed_bytes=None,
                reprogram_passes=1, pass_cores=(4, 3, 2), reprogrammed_cells=(10, 10, 10, 10))
    stated = candidate_context_from_platform({}, programming=SimpleNamespace(**base, schedule_census=(5, 3)))
    assert (stated.pass_count, stated.sync_count) == (5, 3)
    silent = candidate_context_from_platform({}, programming=SimpleNamespace(**base))
    assert (silent.pass_count, silent.sync_count) == (None, None)
    noc = SimpleNamespace(pass_programs=(), schedule_census=(4, 2))
    assert candidate_programming_census(noc, weight_bits=8).schedule_census == (4, 2)
    assert candidate_programming_census(SimpleNamespace(pass_programs=()), weight_bits=8).schedule_census is None
