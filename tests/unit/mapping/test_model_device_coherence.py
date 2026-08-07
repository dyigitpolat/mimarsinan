"""A model lives on ONE device, and every writer into it has to honour that.

Two independent writers used to break it, and the on-chip validity gate was
where the wreckage surfaced — as an ``addmm`` device mismatch a hundred frames
away from either cause:

* ``Perceptron.set_*_activation_scale`` built its tensor with a bare
  ``torch.tensor(float)``, so writing a scale into a CUDA model silently
  RELOCATED that registered Parameter to CPU;
* the on-chip probe read the flow's device off the FIRST parameter it found and
  probed with it, so a half-migrated flow produced a confusing kernel error
  instead of naming the split.

``meta`` stands in for "some other device" so both are observable without a GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.device_placement import (
    MixedDevicePlacementError,
    single_device_of,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

META = torch.device("meta")
CPU = torch.device("cpu")


def _perceptron():
    return Perceptron(4, 3)


@pytest.mark.parametrize(
    "setter", ["set_activation_scale", "set_input_activation_scale"]
)
def test_scale_setters_do_not_relocate_the_parameter(setter):
    """A float write must land where the model already lives."""
    perceptron = _perceptron().to(META)

    getattr(perceptron, setter)(0.5)

    assert single_device_of(perceptron) == META


@pytest.mark.parametrize(
    "setter", ["set_activation_scale", "set_input_activation_scale"]
)
def test_scale_setters_still_write_the_value(setter):
    """Device preservation must not cost the write itself."""
    perceptron = _perceptron()

    getattr(perceptron, setter)(0.25)

    name = setter.removeprefix("set_")
    assert float(getattr(perceptron, name)) == pytest.approx(0.25)


def test_scale_setters_accept_a_tensor_from_another_device():
    perceptron = _perceptron().to(META)

    perceptron.set_activation_scale(torch.tensor(0.5))

    assert single_device_of(perceptron) == META


def test_single_device_of_returns_the_one_device():
    assert single_device_of(nn.Linear(3, 2)) == CPU
    assert single_device_of(nn.Linear(3, 2).to(META)) == META


def test_single_device_of_defaults_to_cpu_when_there_is_no_state():
    assert single_device_of(nn.ReLU()) == CPU


def test_single_device_of_names_the_split_instead_of_picking_one():
    """The gate must FAIL LOUD on a half-migrated model, not guess."""
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
    model[1].to(META)

    with pytest.raises(MixedDevicePlacementError) as excinfo:
        single_device_of(model, what="flow")

    message = str(excinfo.value)
    assert "flow" in message
    assert "cpu" in message and "meta" in message
    assert "1.weight" in message


def test_onchip_gate_names_a_half_migrated_flow():
    """The reported defect, end to end.

    A host ComputeOp module left on another device than the rest of the flow
    used to reach the on-chip MAC probe, which read the device off the first
    parameter and then died inside the host op's kernel — pointing at the gate
    instead of at the corruption. The gate must name the split.
    """
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
    from mimarsinan.mapping.verification.onchip_fraction import (
        estimate_onchip_fraction,
    )
    from mimarsinan.models.deep_mlp import DeepMLP
    from mimarsinan.torch_mapping.converter import convert_torch_model

    input_shape = (1, 8, 8)
    flow = convert_torch_model(
        DeepMLP(input_shape=input_shape, num_classes=4, depth=2, width=8),
        input_shape,
        4,
    )
    flow.get_mapper_repr()._ensure_exec_graph()
    host_ops = [
        node
        for node in flow.get_mapper_repr()._exec_order
        if isinstance(node, ComputeOpMapper)
    ]
    assert host_ops, "the vehicle must carry a host ComputeOp to strand"
    host_ops[-1].module.to(META)

    with pytest.raises(MixedDevicePlacementError) as excinfo:
        estimate_onchip_fraction(flow, input_shape, 4, metric="macs")

    assert "meta" in str(excinfo.value)
