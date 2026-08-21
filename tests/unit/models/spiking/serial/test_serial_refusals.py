"""Every cycle-atomic theorem the ODIN point denies refuses BY NAME.

A mechanism whose correctness argument assumes one compare per cycle, or a
lossless unbounded accumulator, is not "approximate" under the point — it is
wrong, and running it would publish a different physics as the deployed
number. Each of these is a typed error naming the axis that refused it.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.serial import (
    CycleAtomicRefusalError,
    MappingTransformRefusalError,
    SaturatingMembraneRefusalError,
    SerialMembraneInitError,
)

from .test_serial_executors import (
    PER_EVENT_LAW,
    PER_EVENT_UNBOUNDED,
    T,
    build_chain,
    build_flow,
)

_SATURATING_PER_CYCLE = SomaLaw(
    firing_mode="Novena", thresholding_mode="<=",
    firing_granularity="per_cycle",
    membrane_arithmetic="saturating_unsigned", membrane_bits=8,
)


def _sample():
    torch.manual_seed(3)
    return torch.rand(2, 8) * 0.9


def test_the_synchronized_count_executor_refuses_per_event():
    _, hybrid = build_chain()
    flow = build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True)
    flow.lif_execution_synchronized = True
    with pytest.raises(CycleAtomicRefusalError, match="synchronized count executor"):
        with torch.no_grad():
            flow(_sample())


def test_the_synchronized_count_executor_refuses_a_saturating_membrane():
    _, hybrid = build_chain()
    flow = build_flow(hybrid, law=_SATURATING_PER_CYCLE, packed=True)
    flow.lif_execution_synchronized = True
    with pytest.raises(SaturatingMembraneRefusalError, match="linear accumulator"):
        with torch.no_grad():
            flow(_sample())


def test_per_hop_retimed_level_stages_refuse_per_event():
    _, hybrid = build_chain(retimed_levels=True)
    assert any(
        getattr(stage, "retimed_level_stages", None) for stage in hybrid.stages
    ), "fixture must actually carry retimed level stages"
    with pytest.raises(CycleAtomicRefusalError, match="retimed level stages"):
        build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True)


def test_membrane_readout_decode_refuses_a_saturating_membrane():
    _, hybrid = build_chain()
    with pytest.raises(SaturatingMembraneRefusalError, match="Q_T"):
        SpikingHybridCoreFlow(
            (8,), hybrid, T,
            firing_mode="Novena", spike_mode="Uniform", thresholding_mode="<=",
            spiking_mode="lif", cycle_accurate_lif_forward=True,
            membrane_readout=True, soma_law=PER_EVENT_LAW,
        )


def test_neuron_splitting_is_refused_at_the_executor_seam():
    _, hybrid = build_chain()
    mutated = copy.deepcopy(hybrid)
    stage = next(s for s in mutated.stages if s.hard_core_mapping is not None)
    stage.hard_core_mapping.soft_core_placements_per_hard_core[0][0][
        "split_group_id"] = 7
    with pytest.raises(MappingTransformRefusalError, match="neuron splitting"):
        build_flow(mutated, law=PER_EVENT_UNBOUNDED, packed=True)


def test_core_coalescing_is_refused_at_the_executor_seam():
    _, hybrid = build_chain()
    mutated = copy.deepcopy(hybrid)
    stage = next(s for s in mutated.stages if s.hard_core_mapping is not None)
    placements = stage.hard_core_mapping.soft_core_placements_per_hard_core[0]
    placements.append(dict(placements[0]))
    with pytest.raises(MappingTransformRefusalError, match="coalescing"):
        build_flow(mutated, law=PER_EVENT_UNBOUNDED, packed=True)


def test_a_non_integral_window_start_membrane_is_refused():
    _, hybrid = build_chain()
    with pytest.raises(SerialMembraneInitError, match="V0\\*theta"):
        build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True,
                   membrane_init=0.25)


def test_the_default_point_admits_every_one_of_these():
    """The refusals are POINT-keyed: nothing above fires at the default law."""
    _, hybrid = build_chain()
    _, retimed = build_chain(retimed_levels=True)
    build_flow(retimed, law=SomaLaw.resolve({}), packed=True)
    flow = build_flow(hybrid, law=SomaLaw.resolve({}), packed=True,
                      membrane_init=0.25)
    flow.lif_execution_synchronized = True
    with torch.no_grad():
        flow(_sample())
    SpikingHybridCoreFlow(
        (8,), hybrid, T,
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
        membrane_readout=True,
    )
