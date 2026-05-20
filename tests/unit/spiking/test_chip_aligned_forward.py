"""``ChipAlignedForward`` wrapper: symmetric install/uninstall + chip-aligned forward."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.chip_aligned_forward import (
    ChipAlignedForward,
    install_chip_aligned_forward,
    uninstall_chip_aligned_forward,
)
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _tiny_lif_setup(T: int = 4):
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    lif1 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    lif2 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )

    class _Flow(nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessor = nn.Identity()
            self.p1 = p1
            self.p2 = p2
            self._repr = repr_

        def forward(self, x):
            return self._repr(x)

    model = _Flow()
    model.eval()
    cfg = {
        "input_shape": (8,),
        "simulation_steps": T,
        "firing_mode": "Default",
        "spike_generation_mode": "Uniform",
        "thresholding_mode": "<=",
        "spiking_mode": "lif",
        "cycle_accurate_lif_forward": True,
        "device": "cpu",
    }
    return model, hybrid, cfg


def test_chip_aligned_forward_requires_refresh() -> None:
    model, _, cfg = _tiny_lif_setup()
    wrapper = ChipAlignedForward(model, cfg)
    with pytest.raises(RuntimeError, match="refresh"):
        wrapper(torch.rand(2, 8))


def test_install_uninstall_symmetric() -> None:
    model, hybrid, cfg = _tiny_lif_setup()
    assert "forward" not in model.__dict__
    install_chip_aligned_forward(model, cfg, hybrid)
    assert "forward" in model.__dict__
    uninstall_chip_aligned_forward(model)
    assert "forward" not in model.__dict__
    # idempotent uninstall — no AttributeError
    uninstall_chip_aligned_forward(model)


def test_install_blocks_double_patch() -> None:
    model, hybrid, cfg = _tiny_lif_setup()
    install_chip_aligned_forward(model, cfg, hybrid)
    try:
        with pytest.raises(AssertionError, match="already patched"):
            install_chip_aligned_forward(model, cfg, hybrid)
    finally:
        uninstall_chip_aligned_forward(model)


def test_chip_aligned_forward_runs_eval_path_by_default() -> None:
    """When model.eval(), forward should produce bit-exact SCM output."""
    model, hybrid, cfg = _tiny_lif_setup()
    install_chip_aligned_forward(model, cfg, hybrid)
    try:
        model.eval()
        x = torch.rand(2, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()
    finally:
        uninstall_chip_aligned_forward(model)


def test_chip_aligned_forward_training_yields_gradients() -> None:
    """When model.train(), backward must reach the encoding Perceptron weights."""
    model, hybrid, cfg = _tiny_lif_setup()
    install_chip_aligned_forward(model, cfg, hybrid)
    try:
        model.train()
        x = torch.rand(2, 8)
        out = model(x)
        out.sum().backward()
        assert model.p1.layer.weight.grad is not None
        assert torch.isfinite(model.p1.layer.weight.grad).all()
        assert model.p1.layer.weight.grad.abs().sum().item() > 0
    finally:
        uninstall_chip_aligned_forward(model)


def test_chip_aligned_forward_eval_matches_unpatched_scm() -> None:
    """ChipAlignedForward in eval == raw SCM forward (bit-identical)."""
    from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow

    model, hybrid, cfg = _tiny_lif_setup()
    # Reference: build a flow directly and run it.
    flow_ref = SpikingHybridCoreFlow(
        cfg["input_shape"], hybrid, simulation_length=cfg["simulation_steps"],
        firing_mode=cfg["firing_mode"], spike_mode=cfg["spike_generation_mode"],
        thresholding_mode=cfg["thresholding_mode"], spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    ).eval()
    x = torch.rand(2, 8)
    with torch.no_grad():
        ref_out = flow_ref(x)

    install_chip_aligned_forward(model, cfg, hybrid)
    try:
        model.eval()
        with torch.no_grad():
            wrapped_out = model(x)
    finally:
        uninstall_chip_aligned_forward(model)
    torch.testing.assert_close(ref_out, wrapped_out, atol=1e-6, rtol=0.0)
