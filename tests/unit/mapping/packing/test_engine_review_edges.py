"""U1 planner scaling, review round 1: the packing engine's edges a reviewer found, against the reference engine.

- A fusion that needs no core (a 0-axon softcore whose fresh-core check fails) takes the whole type group, as the
  reference's ``bucket[-0:]`` does.
- A list of unused cores that holds one core object twice is refused by name before any placement (the reference
  double-books such a core).
- The core-state index keys on exactly what the canonical predicate, the bankless best-fit cost and the split into a
  used core read of a core: ``canonical.CANONICAL_CORE_READS``, recorded attribute by attribute here.
"""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal.toolchain_refusal import is_toolchain_refusal
from mimarsinan.mapping.packing.canonical import (
    CANONICAL_CORE_READS,
    _remaining_capacity,
    canonical_is_mapping_possible,
    canonical_split_softcore,
)
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores
from mimarsinan.mapping.packing.greedy.pool import CoreListedTwice
from mimarsinan.mapping.packing.greedy.split import _try_split_into_used
from mimarsinan.mapping.packing.greedy.used_index import core_state
from mimarsinan.mapping.packing.placement_cost import placement_cost

from unit.mapping.packing import reference_engine as ref
from unit.mapping.packing.test_engine_equivalence import Hard, Soft, _place, _split


def _both(make, **kw):
    """(trace, error, used labels) on the engine (canonical index on) and on the reference engine."""
    out = []
    for engine, extra in ((greedy_pack_softcores, {"canonical_feasibility": True}),
                          (ref.reference_greedy_pack_softcores, {})):
        softs, used, unused = make()
        trace: list = []
        try:
            engine(softcores=softs, used_hardcores=used, unused_hardcores=unused,
                   is_mapping_possible=canonical_is_mapping_possible, place=_place(trace), **kw, **extra)
            raised = None
        except RuntimeError as exc:
            raised = str(exc)
        out.append((trace, raised, [h.label for h in used]))
    return out


class TestAFusionThatNeedsNoCore:
    def test_it_takes_the_whole_type_group(self):
        """Unused [h0, h1, h2] of class 0, a softcore (0 axons, 10 neurons) of class 1: no fresh core takes it, and
        ceil(0 / 32) = 0 cores fuse; the reference fuses ``bucket[-0:]``, the whole group (the fused core holds no
        class, as a fresh layout core does not)."""
        def make():
            return [Soft(0, 10, "s", tg=1)], [], [Hard(32, 32, f"h{i}", tg=0) for i in range(3)]

        def fuse(hards):
            return Hard(sum(h.get_input_count() for h in hards), hards[0].get_output_count(),
                        "+".join(h.label for h in hards))
        new, old = _both(make, fuse_hardcores=fuse)
        assert new == old
        assert old[1] is None and old[2] == ["h0+h1+h2"]


class TestAPoolListingOneCoreTwice:
    def test_it_is_refused_by_name_before_any_placement(self):
        """Unused [X, Y, X]: the reference's split into an unused core takes Y and leaves used [X, Y] while X is
        still listed; an engine reading a group's head by list position takes X again. Either way one core is
        opened twice, so the list is refused by name."""
        x, y = Hard(32, 32, "X"), Hard(32, 32, "Y")
        unused, used, trace = [x, y, x], [], []
        with pytest.raises(CoreListedTwice, match="listed 2 times") as raised:
            greedy_pack_softcores(softcores=[Soft(8, 40, "B"), Soft(8, 8, "A")], used_hardcores=used,
                                  unused_hardcores=unused, is_mapping_possible=canonical_is_mapping_possible,
                                  place=_place(trace), split_softcore=_split, split_threshold=0.9,
                                  pick_softcore=lambda remaining: remaining[-1])
        assert is_toolchain_refusal(raised.value) and not isinstance(raised.value, RuntimeError)
        assert trace == [] and used == [] and unused == [x, y, x]

    def test_distinct_cores_are_admitted(self):
        unused = [Hard(32, 32, f"h{i}") for i in range(3)]
        greedy_pack_softcores(softcores=[Soft(8, 8, "a")], used_hardcores=[], unused_hardcores=unused,
                              is_mapping_possible=canonical_is_mapping_possible, place=_place([]))
        assert len(unused) == 2


class RecordingCore:
    """A hard core that records the name of every attribute read of it (``getattr`` with a default included)."""

    def __init__(self, axons, neurons, **state):
        values = {"axons_per_core": axons, "neurons_per_core": neurons, **state,
                  "get_input_count": lambda: axons, "get_output_count": lambda: neurons}
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "reads", set())

    def __getattribute__(self, name):
        if name in ("_values", "reads", "__class__", "__dict__"):
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "reads").add(name)
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(name)


STATES = [dict(), dict(residency_class_id=1), dict(residency_class_id=0, latency=2),
          dict(residency_class_id=1, latency_tag=1, available_axons=40, available_neurons=8),
          dict(available_axons=64, available_neurons=0), dict(latency=1, available_axons=3, available_neurons=60)]
SOFTS = [Soft(8, 8, "a"), Soft(8, 8, "b", tg=1), Soft(8, 40, "c", tg=1, latency=1), Soft(70, 4, "d", latency=2),
         Soft(2, 90, "e", tg=0)]


class TestTheCoreStateIsWhatThePredicateReads:
    """[Review round 1] The index's state is defined once, beside the predicate; an attribute the predicate, the
    bankless best-fit cost or the split into a used core reads outside it would make two cores of one state answer
    a softcore differently."""

    ALLOWED = set(CANONICAL_CORE_READS) | {"get_input_count", "get_output_count"}

    def test_the_predicate_and_the_bankless_cost_read_only_the_set(self):
        for state in STATES:
            for soft in SOFTS:
                core = RecordingCore(64, 64, **state)
                if canonical_is_mapping_possible(soft, core):
                    placement_cost(soft, core, remaining_capacity=_remaining_capacity(soft, core))
                assert core.reads <= self.ALLOWED, (state, soft.label, core.reads - self.ALLOWED)

    def test_the_split_into_a_used_core_reads_only_the_set(self):
        split = lambda soft, n: canonical_split_softcore(  # noqa: E731
            soft, n, make_fragments=lambda *, softcore, first_neurons, remaining_neurons: (
                Soft(softcore.get_input_count(), first_neurons, "f1"),
                Soft(softcore.get_input_count(), remaining_neurons, "f2")))
        for state in STATES:
            for soft in SOFTS:
                core = RecordingCore(64, 64, **state)
                _try_split_into_used(soft, [core], split, 0.0, lambda *args: None, [soft])
                assert core.reads <= self.ALLOWED, (state, soft.label, core.reads - self.ALLOWED)

    def test_the_index_keys_on_the_whole_set(self):
        core = RecordingCore(64, 64, residency_class_id=0)
        core_state(core)
        assert core.reads == self.ALLOWED

    def test_a_bank_affine_softcore_reads_the_banks_too(self):
        """Why the engine scans every core for such a softcore (``scan_order(every_core=True)``)."""
        core = RecordingCore(64, 64)
        placement_cost(Soft(8, 8, "b", bank=7), core, remaining_capacity=0)
        assert core.reads - self.ALLOWED == {"resident_bank_ids"}
