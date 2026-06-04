"""bias_mode is config-resolved, validated on both nodes, and a no-op for dynamics.

Mode A (on_chip) and mode B (param_encoded) deliver the same cumulative membrane
``bias·(t_local+1)``, so the differentiable nodes must not branch their forward on
bias_mode. ``resolve_bias_mode`` is the single config→mode resolver shared by the
tuners and the mapping step.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.nn.activations.bias_mode import (
    bias_mode_from_hardware_bias,
    validate_bias_mode,
)
from mimarsinan.models.nn.activations.lif import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode


def test_validate_and_map_helpers():
    assert validate_bias_mode("on_chip") == "on_chip"
    assert validate_bias_mode("param_encoded") == "param_encoded"
    with pytest.raises(ValueError):
        validate_bias_mode("nonsense")
    assert bias_mode_from_hardware_bias(True) == "on_chip"
    assert bias_mode_from_hardware_bias(False) == "param_encoded"


@pytest.mark.parametrize("node_cls", [LIFActivation, TTFSActivation])
def test_nodes_accept_and_validate_bias_mode(node_cls):
    node = node_cls(T=4, activation_scale=1.0, bias_mode="param_encoded")
    assert node.bias_mode == "param_encoded"
    with pytest.raises(ValueError):
        node_cls(T=4, activation_scale=1.0, bias_mode="bogus")


def test_resolve_bias_mode_from_config():
    assert resolve_bias_mode({}) == "on_chip"  # has_bias defaults True
    assert resolve_bias_mode({"platform_constraints": {"has_bias": True}}) == "on_chip"
    assert (
        resolve_bias_mode({"platform_constraints": {"has_bias": False}})
        == "param_encoded"
    )
    # any core lacking has_bias -> mode B
    cfg = {"cores": [{"max_axons": 8, "max_neurons": 8, "has_bias": True},
                     {"max_axons": 8, "max_neurons": 8, "has_bias": False}]}
    assert resolve_bias_mode(cfg) == "param_encoded"


def test_lif_bias_mode_is_a_noop():
    """LIF integrates the post-bias pre-activation; the two modes coincide exactly."""
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=torch.float64)
    out = {}
    for mode in ("on_chip", "param_encoded"):
        lif = LIFActivation(T=6, activation_scale=1.0, bias_mode=mode).double()
        from spikingjelly.activation_based import functional

        functional.reset_net(lif.if_node)
        out[mode] = lif(x.clone())
    torch.testing.assert_close(out["on_chip"], out["param_encoded"], atol=0.0, rtol=0.0)
