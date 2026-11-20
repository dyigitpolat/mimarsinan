"""[U-TGAUGE5] The window in a per-cycle cascade (the fourth verify's M-1 and the design review's J-1): every live
in-segment edge holds a latency gap of one after the alignment, through the mapping, in both walk families.

A per-cycle cascade steps each core from its latency and fills its axons from each producer core's buffer of the
previous cycle, so a reader at a latency gap ``g`` reads its producer's relative cycle ``k + g - 1``. The latency
alignment raises a producer read at two depths to one below its deepest reader, so its shallower reader reads it a
cycle late (gap 0), one count step low. The forms are the verify's (its smallest case and its minimal forms, written
as torch code) and the review's dead read (a zero axon row from a deeper producer, which the relay pass and the
latency engine read differently). Each is built three ways (``cascade_windows``):

- through the mapping (Soft Core Mapping's own window calls, the relays armed as the recipe of every per-cycle cascade
  arms them): the walk equals the deployed read element by element (U-TGAUGE4's tree read the timing family a step
  low: the relays were armed for the LIF family only, and the dead read moved the alignment in both families);
- with the relays declared off: refused by name at mapping ("declared off");
- built outside the mapping: refused by name at the per-cycle executor ("no mapping armed").
"""

from __future__ import annotations

import pytest

import unit.spiking.cascade_windows as cw

from mimarsinan.mapping.latency.depth_balancing import UnbalancedFanInError

#: Seeds per form (the smallest form's seed 0 draws a network whose output is zero on every probe input, which the
#: converter refuses by name: its seeds start at one).
SEEDS = {"smallest": range(1, 7)}
GAP_FORMS = ["smallest", "fanout_entry", "fanout_inner", "single_inner", "dead_read_join"]
FAMILIES = ["timing", "count"]


def _seeds(form):
    return SEEDS.get(form, range(0, 6))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("form", GAP_FORMS)
def test_through_the_mapping_every_live_edge_reads_its_producer_s_cycle(form, family):
    """The relays armed by the recipe of every per-cycle cascade: the walk equals the deployed read element by element on
    every seed, and the mapping inserted relay cores."""
    for seed in _seeds(form):
        read = cw.FAMILIES[family](form, seed, "mapping")
        assert read.off() == 0, (form, family, seed, float((read.walk - read.deployed).abs().max()))
        assert read.relays >= 1, (form, family, seed)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("form", GAP_FORMS)
def test_relays_declared_off_refuse_a_gapped_cascade_at_mapping(form, family):
    for seed in _seeds(form)[:2]:
        with pytest.raises(UnbalancedFanInError, match="declared off"):
            cw.FAMILIES[family](form, seed, "off")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("form", GAP_FORMS)
def test_a_gapped_cascade_built_outside_the_mapping_is_refused_at_the_executor(form, family):
    for seed in _seeds(form)[:2]:
        with pytest.raises(UnbalancedFanInError, match="no mapping armed"):
            cw.FAMILIES[family](form, seed, "bare")


@pytest.mark.parametrize("family", FAMILIES)
def test_a_dead_in_segment_read_leaves_no_gap(family):
    """J-1: C reads A live and a deeper producer Q through an all-zero axon row. The relay pass reads live edges only
    and inserts none; the latency engine read the dead read too and raised A above B's window (B read A at gap -1). The
    mapping turns the dead read off before the relays, so both read one edge set: equal through the mapping, equal with
    the relays declared off (no live gap remains). [U-LIVE] Every scheduling rule reads the one liveness predicate, so
    the dead read raises nothing where no mapping ran either: built bare, equal (it was refused by name)."""
    for seed in range(0, 6):
        for how in ("mapping", "off", "bare"):
            read = cw.FAMILIES[family]("dead_read", seed, how)
            assert read.off() == 0, (family, seed, how, float((read.walk - read.deployed).abs().max()))
            assert read.relays == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_a_single_reader_entry_core_reads_equal_every_way(family):
    """The alignment alone raises an entry core read once across a gap (U-TGAUGE4's S-4 shape): no gap remains, so the
    form reads equal through the mapping, with the relays off and built outside the mapping."""
    for seed in range(0, 6):
        for how in cw.HOWS:
            read = cw.FAMILIES[family]("single_entry", seed, how)
            assert read.off() == 0, (family, seed, how)


@pytest.mark.parametrize("family", FAMILIES)
def test_ir_pruning_after_the_relays_leaves_no_gap(family):
    """Soft Core Mapping prunes the IR after the relays (zero detection and its cascade). Removing reads from a graph
    whose every in-segment edge spans one level cannot open a gap: a core keeps its depth while one read remains, and
    the alignment raises a core that lost every read back under its readers. Two of the shallow producer's channels
    are dead, so pruning removes them, the relays that copy them and their readers' axons: equal, no refusal."""
    for seed in range(0, 4):
        whole = cw.FAMILIES[family]("fanout_inner", seed, "mapping")
        pruned = cw.FAMILIES[family]("fanout_inner", seed, "mapping", prune=True, kill={2: [0, 1]})
        assert pruned.off() == 0, (family, seed, float((pruned.walk - pruned.deployed).abs().max()))
        neurons = lambda ir: sum(int(n.core_matrix.shape[1]) for n in ir.get_neural_cores()  # noqa: E731
                                 if getattr(n, "core_matrix", None) is not None)
        assert neurons(pruned.ir) < neurons(whole.ir), (family, seed)
