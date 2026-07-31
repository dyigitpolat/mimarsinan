"""The declared threshold policy must change the MAPPING, not just a label.

A target whose hardware core holds a single threshold register can only co-locate softcores that
agree on their threshold. A target with per-neuron thresholds has no such limit, so the same
softcores pack together -- fewer cores, higher utilization. If both policies produced the same
mapping the axis would be decoration; these pin that they do not.

Packing is exercised through the real HardCore/greedy path, not a model of it.
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore
from mimarsinan.mapping.platform.core_residency import (
    Granularity,
    constrained_names,
    default_residency_policy,
    residency_key,
    resolve_residency_policy,
)

AXONS, NEURONS = 16, 16
SOFT_AXONS, SOFT_NEURONS = 4, 4          # block-diagonal: four of these fill a 16x16 core
N_SOFTCORES = 4
DISTINCT_THRESHOLDS = (1.0, 0.5, 2.0, 0.25)


def _softcore(idx: int, threshold: float) -> SoftCore:
    sc = SoftCore(
        core_matrix=np.zeros((SOFT_AXONS, SOFT_NEURONS), dtype=np.float64),
        axon_sources=[None] * SOFT_AXONS,
        id=idx,
    )
    sc.threshold = threshold
    sc.activation_scale = torch.tensor(1.0)
    sc.parameter_scale = torch.tensor(1.0)
    sc.input_activation_scale = torch.tensor(1.0)
    return sc


def _policy(override: dict) -> dict:
    """A spiking target with `threshold` declared at the given grain."""
    p = default_residency_policy(value_domain=False)
    p.update(override)
    return p


PER_CORE_THRESHOLD = {"threshold": Granularity.PER_CORE}
PER_NEURON_THRESHOLD = {"threshold": Granularity.PER_NEURON}


def _constrained_names(policy: dict) -> frozenset[str]:
    """Which per-core values this target constrains, from the declared granularity."""
    return constrained_names(policy)


def _pack(softcores, policy) -> list[HardCore]:
    """Greedy first-fit honouring the residency class: a softcore joins a core it is compatible
    with, else it opens a new one. `add_softcore` is the real one, so an illegal merge raises."""
    names = _constrained_names(policy)
    cores: list[HardCore] = []
    for sc in softcores:
        key = residency_key(sc, constrained=names)
        for hc in cores:
            fits = (hc.available_axons >= sc.get_input_count()
                    and hc.available_neurons >= sc.get_output_count())
            if fits and getattr(hc, "_residency_key", key) == key:
                hc.add_softcore(sc)
                break
        else:
            hc = HardCore(AXONS, NEURONS)
            hc.constrained_properties = names
            hc._residency_key = key
            hc.add_softcore(sc)
            cores.append(hc)
    return cores


def _utilization(cores) -> float:
    """Fraction of allocated crossbar area actually carrying weights (block-diagonal placement)."""
    used = sum((c.axons_per_core - c.available_axons) * (c.neurons_per_core - c.available_neurons)
               - _idle_block_area(c) for c in cores)
    return used / sum(c.axons_per_core * c.neurons_per_core for c in cores)


def _idle_block_area(core) -> int:
    """Off-diagonal area a block-diagonal placement leaves empty."""
    used_ax = core.axons_per_core - core.available_axons
    used_ne = core.neurons_per_core - core.available_neurons
    n = used_ax // SOFT_AXONS
    return used_ax * used_ne - n * SOFT_AXONS * SOFT_NEURONS


@pytest.fixture
def softcores():
    return [_softcore(i, t) for i, t in enumerate(DISTINCT_THRESHOLDS)]


class TestPerNeuronThresholdsPackBetter:
    def test_hardcore_wide_thresholds_cannot_co_locate_distinct_thresholds(self, softcores):
        cores = _pack(softcores, _policy(PER_CORE_THRESHOLD))
        assert len(cores) == N_SOFTCORES, "one core per distinct threshold"
        for hc in cores:
            assert hc.available_neurons == NEURONS - SOFT_NEURONS

    def test_per_neuron_thresholds_co_locate_them(self, softcores):
        cores = _pack(softcores, _policy(PER_NEURON_THRESHOLD))
        assert len(cores) == 1, "nothing forbids sharing, so they share"

    def test_the_two_policies_give_different_mappings(self, softcores):
        wide = _pack([_softcore(i, t) for i, t in enumerate(DISTINCT_THRESHOLDS)],
                     _policy(PER_CORE_THRESHOLD))
        per_neuron = _pack(softcores, _policy(PER_NEURON_THRESHOLD))
        assert len(per_neuron) < len(wide)

    def test_per_neuron_utilization_is_visibly_higher(self, softcores):
        wide = _pack([_softcore(i, t) for i, t in enumerate(DISTINCT_THRESHOLDS)],
                     _policy(PER_CORE_THRESHOLD))
        per_neuron = _pack(softcores, _policy(PER_NEURON_THRESHOLD))
        u_wide, u_per_neuron = _utilization(wide), _utilization(per_neuron)
        assert u_per_neuron > u_wide
        assert u_per_neuron / u_wide == pytest.approx(N_SOFTCORES, rel=1e-9)


class TestTheRelaxationIsNarrow:
    def test_relaxing_thresholds_does_not_merge_cores_differing_elsewhere(self):
        """Per-neuron thresholds relax the threshold ONLY; another mismatch still separates.

        `activation_scale` and not `parameter_scale`: on a spiking target the weight scale is
        folded into the threshold, so `parameter_scale` is declared ABSENT and is vacuous there.
        """
        a, b = _softcore(0, 1.0), _softcore(1, 0.5)
        b.activation_scale = torch.tensor(8.0)
        cores = _pack([a, b], _policy(PER_NEURON_THRESHOLD))
        assert len(cores) == 2

    def test_an_illegal_merge_would_raise_rather_than_corrupt(self):
        """The guard underneath: even if a key were too coarse, add_softcore refuses."""
        from mimarsinan.mapping.platform.core_residency import CoreResidencyViolation

        hc = HardCore(AXONS, NEURONS)
        hc.add_softcore(_softcore(0, 1.0))
        with pytest.raises(CoreResidencyViolation, match="threshold"):
            hc.add_softcore(_softcore(1, 0.5))


class TestThePolicyComesFromTheDeclaration:
    def test_a_target_declaring_per_neuron_thresholds_gets_the_better_mapping(self, softcores):
        policy = resolve_residency_policy(
            {"core_value_granularity": {"threshold": "per_neuron"}}, value_domain=False
        )
        assert len(_pack(softcores, policy)) == 1

    def test_the_default_target_keeps_the_constraint(self, softcores):
        policy = resolve_residency_policy({}, value_domain=False)
        assert len(_pack(softcores, policy)) == N_SOFTCORES


class TestRelaxingAConstraintStoresTheValues:
    """Relaxing means the hardware stores one per neuron -- not that we stopped checking.

    Without this the packer would merge cores under a per-neuron declaration and then represent
    only the first core's threshold, which is precisely the silent corruption the guard removed.
    """

    def test_each_neuron_range_keeps_its_own_threshold(self, softcores):
        from mimarsinan.mapping.platform.core_residency import PER_NEURON_ATTR

        [core] = _pack(softcores, _policy(PER_NEURON_THRESHOLD))
        column = getattr(core, PER_NEURON_ATTR)["threshold"]
        for i, expected in enumerate(DISTINCT_THRESHOLDS):
            lo, hi = i * SOFT_NEURONS, (i + 1) * SOFT_NEURONS
            assert column[lo:hi] == [expected] * SOFT_NEURONS, f"neurons {lo}:{hi}"

    def test_a_constrained_target_records_no_per_neuron_column(self, softcores):
        from mimarsinan.mapping.platform.core_residency import PER_NEURON_ATTR

        cores = _pack(softcores, _policy(PER_CORE_THRESHOLD))
        for core in cores:
            store = getattr(core, PER_NEURON_ATTR, None)
            assert store is None or "threshold" not in store

    def test_the_scalar_threshold_is_not_silently_set_under_relaxation(self, softcores):
        """A single scalar cannot represent four values; it must stay unset, not hold the first."""
        [core] = _pack(softcores, _policy(PER_NEURON_THRESHOLD))
        assert core.threshold is None
