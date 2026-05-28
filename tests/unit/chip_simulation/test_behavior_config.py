"""NeuralBehaviorConfig — simulator-facing activation semantics."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.firing_strategy import FiringMode


def test_from_deployment_config_defaults():
    cfg = NeuralBehaviorConfig.from_deployment_config({})
    assert cfg.spiking_mode == "lif"
    assert cfg.firing_mode == "Default"
    assert cfg.thresholding_mode == "<="
    assert cfg.spike_generation_mode == "Uniform"


def test_for_lava_rejects_non_lif():
    with pytest.raises(ValueError, match="spiking_mode"):
        NeuralBehaviorConfig.for_lava({"spiking_mode": "ttfs"})


def test_for_lava_accepts_lif():
    behavior = NeuralBehaviorConfig.for_lava(
        {"spiking_mode": "lif", "firing_mode": "Novena", "thresholding_mode": "<"}
    )
    assert behavior.spiking_mode == "lif"
    assert behavior.firing_mode == "Novena"


def test_nevresim_reset_policy():
    default = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<",
        spike_generation_mode="Uniform",
    )
    novena = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Novena",
        thresholding_mode="<",
        spike_generation_mode="Uniform",
    )
    assert default.nevresim_reset_policy() == "SubtractiveReset"
    assert novena.nevresim_reset_policy() == "ZeroReset"


def test_nevresim_compare_policy():
    strict = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<",
        spike_generation_mode="Uniform",
    )
    inclusive = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    assert strict.nevresim_compare_policy() == "StrictCompare"
    assert inclusive.nevresim_compare_policy() == "InclusiveCompare"


def test_lava_zero_reset():
    default = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    novena = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Novena",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    assert default.lava_zero_reset() is False
    assert novena.lava_zero_reset() is True


def test_sanafe_reset_mode():
    cfg = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Novena",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    assert cfg.sanafe_reset_mode() == "hard"


def test_firing_strategy_wraps_factory():
    cfg = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Novena",
        thresholding_mode="<",
        spike_generation_mode="Uniform",
    )
    assert cfg.firing_strategy().mode == FiringMode.NOVENA


def test_encode_segment_input_uniform():
    cfg = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )
    rates = np.full((1, 2), 0.5, dtype=np.float32)
    out = cfg.encode_segment_input(rates, T=8)
    assert out.shape == (1, 2, 8)
    assert out.sum() == 8.0


def test_encode_segment_input_rejects_ttfs():
    cfg = NeuralBehaviorConfig(
        spiking_mode="ttfs",
        firing_mode="TTFS",
        thresholding_mode="<=",
        spike_generation_mode="TTFS",
    )
    with pytest.raises(ValueError, match="TTFS"):
        cfg.encode_segment_input(np.zeros((1, 1), dtype=np.float32), T=4)
