"""[U-TGAUGE6] The review's dead read in both LIF semantics and in the timing family (U-TGAUGE5's check, M-2).

C reads A live and a deeper producer Q through an all-zero axon row; turning C's read off leaves Q with no reader.
Under per-hop re-timing (the windowed LIF semantics) Q sat alone at its depth level, so its level
stage held no outputs and the per-cycle executor failed with an unnamed error (U-TGAUGE4's tree read the form equal).
The pass that turns the read off removes Q with its own reads (and a chain of such producers, to a fixed point), so
every semantics reads the form equal through the mapping and with the relays declared off, and the per-hop build holds
no stage without outputs. The forms are ``cascade_windows``' dead read, the dead read beside a live unequal-depth join
and a chain of two producers read only through the dead read. The class's other site, IR pruning (it turns reads off
too), is read under the same per-hop build.
"""

from __future__ import annotations

import pytest

import unit.spiking.cascade_windows as cw

from mimarsinan.mapping.latency.depth_balancing import UnbalancedFanInError

#: (family, LIF semantics): the count family under both LIF semantics, the timing family's cascaded schedule.
READS = [("count", "streamed_lif"), ("count", "windowed_lif"), ("timing", None)]
#: Per form, the cores the pass removes (each producer of these forms is one core).
REMOVED = {"dead_read": 1, "dead_read_join": 1, "dead_chain": 2}
SEEDS = range(0, 6)


def _build(form, family, semantics, seed, how):
    if family == "count":
        return cw.count(form, seed, how, semantics=semantics)
    return cw.timing(form, seed, how)


def _unread(ir) -> list:
    read = {int(s.node_id) for n in ir.nodes for s in n.input_sources.flatten() if s.node_id >= 0}
    read |= {int(s.node_id) for s in ir.output_sources.flatten() if s.node_id >= 0}
    return sorted({int(n.id) for n in ir.nodes} - read)


def _stages_without_outputs(read) -> list:
    """Every neural stage of the deployed build, and each per-hop level stage, whose mapping holds no output source."""
    hybrid = getattr(read.executor, "hybrid_mapping", None)
    empty = []
    for stage in getattr(hybrid, "stages", None) or []:
        for unit in [stage] + list(getattr(stage, "retimed_level_stages", None) or []):
            if unit.kind == "neural" and len(unit.hard_core_mapping.output_sources) == 0:
                empty.append(unit.name)
    return empty


@pytest.mark.parametrize("family,semantics", READS, ids=["count_streamed", "count_windowed", "timing"])
@pytest.mark.parametrize("form", ["dead_read", "dead_chain"])
def test_a_producer_read_only_through_a_dead_read_reads_equal_every_way_the_mapping_builds_it(form, family,
                                                                                            semantics):
    """Through the mapping and with the relays declared off: equal element by element, no relay, the producers the
    dead read leaves unread removed, no core left unread."""
    for seed in SEEDS:
        for how in ("mapping", "off"):
            read = _build(form, family, semantics, seed, how)
            assert read.off() == 0, (form, family, semantics, seed, how,
                                     float((read.walk - read.deployed).abs().max()))
            assert read.relays == 0
            assert read.window == {"relay_cores_inserted": 0, "dead_reads_off": read.window["dead_reads_off"],
                                   "unread_cores_removed": REMOVED[form]}, read.window
            assert read.cores_mapped - cw.neural_cores(read.ir) == REMOVED[form]
            assert _unread(read.ir) == []


@pytest.mark.parametrize("form", ["dead_read", "dead_chain", "dead_read_join"])
@pytest.mark.parametrize("how", ["mapping", "off"])
def test_the_per_hop_build_holds_no_stage_without_outputs(form, how):
    """The windowed LIF semantics re-times per hop: every level stage holds an output (the review's dead read failed
    here with ``mapping has no output_sources``), and the form reads equal (the per-hop units hold no window)."""
    for seed in SEEDS:
        read = cw.count(form, seed, how, semantics="windowed_lif")
        assert _stages_without_outputs(read) == [], (form, how, seed)
        assert any(getattr(s, "retimed_level_stages", None) for s in read.executor.hybrid_mapping.stages)
        assert read.off() == 0, (form, how, seed, float((read.walk - read.deployed).abs().max()))


@pytest.mark.parametrize("family,semantics", [("count", "streamed_lif"), ("timing", None)],
                         ids=["count_streamed", "timing"])
def test_the_dead_read_beside_a_live_join_is_refused_only_by_name_with_the_relays_declared_off(family, semantics):
    """C reads A, B and Q with Q's columns at zero: the live unequal-depth join remains after Q goes, so the relays
    make it equal through the mapping and the window rule refuses it by name with the relays declared off."""
    for seed in SEEDS:
        read = _build("dead_read_join", family, semantics, seed, "mapping")
        assert read.off() == 0 and read.relays >= 1, (family, seed)
        assert read.window["unread_cores_removed"] == REMOVED["dead_read_join"]
    for seed in SEEDS[:2]:
        with pytest.raises(UnbalancedFanInError, match="declared off"):
            _build("dead_read_join", family, semantics, seed, "off")


@pytest.mark.parametrize("how", ["mapping", "bare"])
@pytest.mark.parametrize("form", ["dead_read", "dead_chain", "dead_read_join"])
def test_ir_pruning_leaves_no_core_unread_under_per_hop_retiming(form, how):
    """The class's other site: IR pruning also turns reads off (its zero-row cascade) and removes the cores its own
    liveness finds no consumer for. Through the mapping (after the window pass) and with no window pass at all, the
    pruned per-hop build holds no core without a reader and no level stage without outputs, and reads equal."""
    for seed in SEEDS[:3]:
        read = cw.count(form, seed, how, semantics="windowed_lif", prune=True)
        assert _unread(read.ir) == [] and _stages_without_outputs(read) == [], (form, how, seed)
        assert read.off() == 0, (form, how, seed, float((read.walk - read.deployed).abs().max()))
