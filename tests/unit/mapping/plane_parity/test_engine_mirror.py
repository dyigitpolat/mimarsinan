"""[W04] The layout packer is the hard-core packer on shapes: for the same soft cores it takes the same decisions.

The hard-core packer (``HardCoreMapping.map``) runs the greedy engine on the IR's soft cores: a wide core stays whole
and is fused onto ceil(a / A) cores only under declared coalescing, a core of a coalescing group is never neuron-split,
a bank-backed core is bank-affine and every core it lands on records its bank. Before W04 the layout packer pre-split
every wide spec into independent single-core fragments, wired fusion whatever the declaration, and its specs carried
neither the bank under the name the engine reads (``weight_bank_id``) nor the coalescing group, so the shape-only
plane (and the builder's own pass planner, which validates passes with it) answered for another packing.

The runtime stand-ins are the engine-equivalence suite's (``Soft``, ``Hard``, ``_place``, ``_fuse``, ``_split``).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreInstance, LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import canonical_is_mapping_possible
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal, refusal_name_of_message
from mimarsinan.mapping.packing.placement_cost import resident_bank_ids
from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore

from unit.mapping.packing.test_engine_equivalence import Hard, Soft, _fuse, _place, _split


def _soft(spec: LayoutSoftCoreSpec, label: str) -> Soft:
    soft = Soft(spec.input_count, spec.output_count, label, tg=spec.residency_class_id, latency=spec.latency_tag,
                bank=spec.bank_id)
    if spec.coalescing_group_id is not None:
        soft.coalescing_group_id = spec.coalescing_group_id
    return soft


def _runtime(specs, a, n, count, *, coalescing, splitting):
    """The hard-core packer's decisions on the specs' cores: the used cores' shapes in order, or the refusal."""
    hards = [Hard(a, n, f"h{i}") for i in range(count)]
    used, trace = [], []
    try:
        greedy_pack_softcores(softcores=[_soft(s, f"s{i}") for i, s in enumerate(specs)], used_hardcores=used,
                              unused_hardcores=hards, is_mapping_possible=canonical_is_mapping_possible,
                              place=_place(trace), fuse_hardcores=_fuse if coalescing else None,
                              split_softcore=_split if splitting else None)
    except RuntimeError as exc:
        return type(exc).__name__
    return [(h.axons_per_core, h.neurons_per_core, h.axons_per_core - h.available_axons,
             h.neurons_per_core - h.available_neurons) for h in used]


def _layout(specs, a, n, count, *, coalescing, splitting):
    pack = pack_layout(softcores=specs, core_types=[LayoutHardCoreType(a, n, count)],
                       allow_neuron_splitting=splitting, allow_coalescing=coalescing)
    if not pack.feasible:
        return refusal_name_of_message(pack.error or "")
    return [(s.axons_per_core, s.neurons_per_core, s.used_axons, s.used_neurons) for s in pack.used_core_snapshots]


def _random_specs(rng: random.Random, a: int, n: int):
    specs = []
    for i in range(rng.randint(4, 14)):
        wide = rng.random() < 0.3
        axons = rng.randint(a + 1, 3 * a) if wide else rng.randint(1, a)
        neurons = rng.randint(1, 2 * n)
        bank = rng.choice([None, 0, 1, 2]) if not wide or rng.random() < 0.5 else None
        group = i if wide and bank is None and neurons <= n and rng.random() < 0.5 else None
        specs.append(LayoutSoftCoreSpec(input_count=axons, output_count=neurons, residency_class_id=rng.choice([0, 1]),
                                        latency_tag=0, bank_id=bank, coalescing_group_id=group, name=f"s{i}"))
    return specs


@pytest.mark.parametrize("seed", range(40))
def test_the_layout_packer_takes_the_hard_core_packers_decisions(seed):
    rng = random.Random(seed)
    a, n = rng.choice([(8, 8), (16, 8), (8, 16)])
    specs = _random_specs(rng, a, n)
    count = rng.choice([4, 8, 16, 64])
    for coalescing in (True, False):
        for splitting in (True, False):
            want = _runtime(specs, a, n, count, coalescing=coalescing, splitting=splitting)
            got = _layout(specs, a, n, count, coalescing=coalescing, splitting=splitting)
            assert got == want, (coalescing, splitting)


def test_a_runtime_fragment_keeps_its_parents_residency_class():
    """Before W04 ``build_real_fragment`` dropped the class: a remainder fragment read as an ungrouped core and shared
    no core with its own layer (the canonical predicate's unique fallback), while a layout fragment kept the class."""
    parent = SoftCore(np.ones((3, 5)), [SpikeSource(0, i, True, False) for i in range(3)], id=11,
                      residency_class_id=7, name="p")
    parent.latency = 2
    first, rest = make_real_fragments(softcore=parent, first_neurons=2, remaining_neurons=3, group_id=0)
    assert (first.residency_class_id, rest.residency_class_id) == (7, 7)
    assert (first.latency, rest.latency) == (2, 2)


class TestTheSpecCarriesWhatThePackerReads:
    def test_a_bank_backed_spec_is_bank_affine(self):
        spec = LayoutSoftCoreSpec(input_count=4, output_count=4, bank_id=3)
        assert spec.weight_bank_id == 3
        assert LayoutSoftCoreSpec(input_count=4, output_count=4).weight_bank_id is None

    def test_a_layout_core_records_the_banks_it_hosts(self):
        core = LayoutHardCoreInstance(axons_per_core=8, neurons_per_core=8)
        core.add_softcore(LayoutSoftCoreSpec(input_count=2, output_count=2, bank_id=5))
        assert resident_bank_ids(core) == frozenset({5})

    def test_a_wide_spec_is_fused_whole_under_coalescing(self):
        pack = pack_layout(softcores=[LayoutSoftCoreSpec(input_count=20, output_count=6)],
                           core_types=[LayoutHardCoreType(8, 8, 8)], allow_coalescing=True)
        assert [(s.axons_per_core, s.used_axons, s.used_neurons) for s in pack.used_core_snapshots] == [(24, 20, 6)]

    def test_a_wide_spec_is_refused_without_coalescing(self):
        pack = pack_layout(softcores=[LayoutSoftCoreSpec(input_count=20, output_count=6)],
                           core_types=[LayoutHardCoreType(8, 8, 8)], allow_coalescing=False)
        assert not pack.feasible and refusal_name_of_message(pack.error or "") == "CoreCapacityRefusal"

    def test_a_coalescing_group_core_is_not_neuron_split(self):
        grouped = LayoutSoftCoreSpec(input_count=6, output_count=12, coalescing_group_id=0)
        pack = pack_layout(softcores=[grouped], core_types=[LayoutHardCoreType(8, 8, 4)], allow_neuron_splitting=True)
        assert not pack.feasible
        free = LayoutSoftCoreSpec(input_count=6, output_count=12)
        assert pack_layout(softcores=[free], core_types=[LayoutHardCoreType(8, 8, 4)],
                           allow_neuron_splitting=True).feasible


def test_the_capacity_refusal_is_one_named_refusal():
    assert issubclass(CoreCapacityRefusal, ToolchainRefusal) and issubclass(CoreCapacityRefusal, RuntimeError)
    with pytest.raises(CoreCapacityRefusal):
        greedy_pack_softcores(softcores=[Soft(9, 9, "big")], used_hardcores=[], unused_hardcores=[Hard(8, 8, "h")],
                              is_mapping_possible=canonical_is_mapping_possible, place=_place([]))
    assert refusal_name_of_message("[proven_infeasible] CoreCapacityRefusal: no core") == "CoreCapacityRefusal"
    assert refusal_name_of_message("CoreCapacityRefusal: no core") == "CoreCapacityRefusal"
    assert refusal_name_of_message("HW bin-packing infeasible") is None
