"""Tests for LavaLoihiRunner behavior config wiring."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi.runner import LavaLoihiRunner
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping


def _minimal_behavior(**overrides) -> NeuralBehaviorConfig:
    base = dict(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    base.update(overrides)
    return NeuralBehaviorConfig(**base)


def test_runner_rejects_non_lif_behavior():
    behavior = _minimal_behavior(spiking_mode="ttfs", firing_mode="TTFS")
    with pytest.raises(ValueError, match="spiking_mode"):
        LavaLoihiRunner(
            mapping=HybridHardCoreMapping(stages=[], output_sources=[]),
            simulation_length=4,
            behavior=behavior,
        )


def test_runner_novena_sets_zero_reset_flag():
    behavior = _minimal_behavior(firing_mode="Novena")
    runner = LavaLoihiRunner(
        mapping=HybridHardCoreMapping(stages=[], output_sources=[]),
        simulation_length=4,
        behavior=behavior,
    )
    assert runner._behavior.lava_zero_reset() is True
    assert runner._firing_strategy.mode.value == "Novena"
