"""FiringStrategy factory and LIF reset semantics."""

import pytest
import torch

from mimarsinan.chip_simulation.firing_strategy import FiringMode, FiringStrategyFactory
from mimarsinan.models.lif_kernels import lif_fire_and_reset


def test_novena_zero_resets_membrane():
    memb = torch.tensor([2.0, 0.5])
    th = torch.tensor(1.0)
    lif_fire_and_reset(memb, th, thresholding_mode="<=", firing_mode="Novena")
    assert memb[0].item() == 0.0
    assert memb[1].item() == 0.5


def test_default_subtractive_reset():
    memb = torch.tensor([2.0])
    th = torch.tensor(1.0)
    lif_fire_and_reset(memb, th, thresholding_mode="<=", firing_mode="Default")
    assert memb[0].item() == pytest.approx(1.0)


def test_factory_ttfs_validation():
    with pytest.raises(ValueError):
        FiringStrategyFactory.from_config(
            {"spiking_mode": "ttfs", "firing_mode": "Default", "thresholding_mode": "<="}
        )


def test_factory_lif_modes():
    s = FiringStrategyFactory.from_config(
        {"spiking_mode": "lif", "firing_mode": "Novena", "thresholding_mode": "<"}
    )
    assert s.mode == FiringMode.NOVENA
    assert s.training_lif_v_reset() == 0.0
    assert s.sanafe_reset_mode() == "hard"
