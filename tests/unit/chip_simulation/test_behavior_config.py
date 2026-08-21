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


# [ODIN P3, D2] the reset law has ONE resolver now
# (chip_simulation.nevresim_policy_types). Both LIF cells are the historical
# answer on both former call sites. TTFS is the cell the two resolvers
# DISAGREED on — inert (a TTFS neuron never reads a LIF reset policy), and the
# consolidation keeps the codegen answer, the one that reaches emitted C++.
# Every other value now RAISES instead of silently becoming somebody's physics.
NEVRESIM_RESET_BY_FIRING_MODE = {
    "Default": "SubtractiveReset",
    "Novena": "ZeroReset",
    "TTFS": "SubtractiveReset",
}
NEVRESIM_RESET_UNKNOWN_FIRING_MODES = ("", "Bogus")

NEVRESIM_COMPARE_BY_THRESHOLDING_MODE = {
    "<": "StrictCompare",
    "<=": "InclusiveCompare",
}
NEVRESIM_COMPARE_UNKNOWN_THRESHOLDING_MODES = ("", "<==", "le")

LAVA_ZERO_RESET_BY_FIRING_MODE = {
    "Default": False,
    "Novena": True,
    "TTFS": False,
    "": False,
    "Bogus": False,
}


def _behavior_with_firing_mode(firing_mode: str) -> NeuralBehaviorConfig:
    return NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode=firing_mode,
        thresholding_mode="<=",
        spike_generation_mode="Uniform",
    )


def test_nevresim_reset_policy_full_firing_mode_table():
    for firing_mode, expected in NEVRESIM_RESET_BY_FIRING_MODE.items():
        got = _behavior_with_firing_mode(firing_mode).nevresim_reset_policy()
        assert got == expected, firing_mode


def test_an_unknown_firing_mode_has_no_reset_policy():
    """The D2 defect, pinned shut: an unrecognized firing mode used to resolve
    to ZeroReset here and SubtractiveReset in codegen. Now it resolves to
    nothing at all."""
    from mimarsinan.chip_simulation.nevresim_policy_types import (
        NevresimPolicyTypeError,
    )

    for firing_mode in NEVRESIM_RESET_UNKNOWN_FIRING_MODES:
        with pytest.raises(NevresimPolicyTypeError, match="firing_mode"):
            _behavior_with_firing_mode(firing_mode).nevresim_reset_policy()


def test_nevresim_compare_policy_full_thresholding_mode_table():
    for mode, expected in NEVRESIM_COMPARE_BY_THRESHOLDING_MODE.items():
        behavior = NeuralBehaviorConfig(
            spiking_mode="lif", firing_mode="Novena", thresholding_mode=mode,
            spike_generation_mode="Uniform",
        )
        assert behavior.nevresim_compare_policy() == expected, mode


def test_an_unknown_thresholding_mode_has_no_compare_policy():
    from mimarsinan.chip_simulation.nevresim_policy_types import (
        NevresimPolicyTypeError,
    )

    for mode in NEVRESIM_COMPARE_UNKNOWN_THRESHOLDING_MODES:
        behavior = NeuralBehaviorConfig(
            spiking_mode="lif", firing_mode="Novena", thresholding_mode=mode,
            spike_generation_mode="Uniform",
        )
        with pytest.raises(NevresimPolicyTypeError, match="thresholding_mode"):
            behavior.nevresim_compare_policy()


def test_lava_zero_reset_full_firing_mode_table():
    for firing_mode, expected in LAVA_ZERO_RESET_BY_FIRING_MODE.items():
        got = _behavior_with_firing_mode(firing_mode).lava_zero_reset()
        assert got is expected, firing_mode


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
