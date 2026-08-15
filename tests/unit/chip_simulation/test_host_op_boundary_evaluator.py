"""The host-op evaluator is part of the deployment contract, not the backend.

Root cause of the ``seg_input: 1/256 differ, Σ exp=86 act=87`` SANA-FE parity
failure (MNIST simple_mlp, subsumed encoding layer, seed 3): a hosted ComputeOp
ends in a staircase activation of step ``theta/T``. Element 5's pre-activation
sat 1.8e-8 from a step edge, so the census/HCM reference (which evaluates host
ops on the pipeline device) and the SANA-FE runner (which evaluated them on the
cpu) landed on ADJACENT steps — and one step is exactly one spike once the
segment boundary transcodes value to counts. The nevresim runner already pinned
the evaluator (``host_compute_device``) and passed parity in the very same run;
SANA-FE and Lava were the two backends that did not.

The fix is structural: the evaluator rides on ``SpikingDeploymentContract``, and
``execute_compute_op_numpy`` REFUSES to default it, so a backend cannot silently
re-derive a host op somewhere else.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_numpy,
    assemble_segment_input_torch,
    execute_compute_op_numpy,
)
from mimarsinan.chip_simulation.recording._spike_encoding import (
    encode_segment_input as encode_segment_input_numpy,
)
from mimarsinan.chip_simulation.sanafe import runner as runner_mod
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.spiking.segment_boundary import (
    normalize_boundary_slices_numpy,
    normalize_boundary_slices_torch,
)
from mimarsinan.spiking.spike_trains import uniform_spike_train


def _declared_behavior(spiking_mode="lif", firing_mode="Default"):
    """The semantics these fixtures were written against — the executor's former
    silent defaults, now STATED at the fixture (the hardening's point)."""
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig

    return NeuralBehaviorConfig(
        spiking_mode=spiking_mode, firing_mode=firing_mode,
        thresholding_mode="<=", spike_generation_mode="Uniform")



# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _compute_only_mapping(op_id=42):
    op = SimpleNamespace(id=op_id)
    stage = SimpleNamespace(
        kind="compute", name="host_op", hard_core_mapping=None,
        compute_op=op, input_map=[], output_map=[],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    return SimpleNamespace(
        stages=[stage],
        get_neural_segments=lambda: [],
        get_compute_ops=lambda: [op],
        output_sources=np.array([], dtype=object),
        node_activation_scales={},
        node_input_activation_scales={},
    )


def _contract(device):
    return SpikingDeploymentContract.from_pipeline_config({
        "simulation_steps": 4,
        "spiking_mode": "lif",
        "firing_mode": "Default",
        "thresholding_mode": "<",
        "spike_generation_mode": "Uniform",
        "device": device,
    })


def _seg_io_slice(node_id, offset, size):
    return SimpleNamespace(node_id=node_id, offset=offset, size=size)


def _op(module, size, src_node):
    sources = np.array(
        [IRSource(node_id=src_node, index=i) for i in range(size)], dtype=object,
    ).reshape(1, size)
    return ComputeOp(
        id=1, name="host_op", op_type=type(module).__name__,
        input_sources=sources, params={"module": module},
        input_shape=(size,), output_shape=(size,),
    )


# ---------------------------------------------------------------------------
# the contract owns the evaluator
# ---------------------------------------------------------------------------


def test_contract_carries_the_pipeline_device_as_the_host_op_evaluator():
    assert _contract("cuda:0").host_compute_device == "cuda:0"
    assert _contract("cpu").host_compute_device == "cpu"


def test_contract_without_a_device_key_leaves_the_evaluator_unset():
    contract = SpikingDeploymentContract.from_pipeline_config({
        "simulation_steps": 4,
        "spiking_mode": "lif",
        "firing_mode": "Default",
        "thresholding_mode": "<",
        "spike_generation_mode": "Uniform",
    })
    assert contract.host_compute_device is None


# ---------------------------------------------------------------------------
# every backend that re-derives a host op asks the contract where
# ---------------------------------------------------------------------------


def test_sanafe_runner_evaluates_host_ops_on_the_contract_device(monkeypatch):
    seen = {}

    def fake_compute(op, original_input, state_buffer, *, device,
                     in_scale, out_scale, dtype=np.float32):
        seen["device"] = device
        return np.asarray([[3.0]], dtype=dtype)

    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy", fake_compute)
    runner = SanafeRunner(
        mapping=_compute_only_mapping(), simulation_length=4,
        contract=_contract("cuda:1"),
    )
    runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)
    assert seen["device"] == "cuda:1", (
        "the runner must re-derive host ops where the census reference did"
    )


def test_sanafe_runner_without_a_contract_takes_the_evaluator_explicitly(monkeypatch):
    seen = {}

    def fake_compute(op, original_input, state_buffer, *, device,
                     in_scale, out_scale, dtype=np.float32):
        seen["device"] = device
        return np.asarray([[3.0]], dtype=dtype)

    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy", fake_compute)
    runner = SanafeRunner(
        mapping=_compute_only_mapping(), simulation_length=4,
        host_compute_device="cuda:1",
    behavior=_declared_behavior())
    runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)
    assert seen["device"] == "cuda:1"


def test_execute_compute_op_numpy_refuses_to_default_the_evaluator():
    """No caller may leave the tie-deciding device to a default."""
    module = nn.Linear(2, 2)
    op = _op(module, 2, src_node=7)
    buf = {7: np.zeros((1, 2), dtype=np.float64)}
    with pytest.raises(TypeError, match="device"):
        execute_compute_op_numpy(op, np.zeros((1, 2)), buf)  # type: ignore[call-arg]


class _HomeReportingLinear(nn.Linear):
    """Records where it is asked to move, without actually moving."""

    def __init__(self):
        super().__init__(2, 2)
        self.to_targets: list = []

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.to_targets.append(args[0] if args else kwargs.get("device"))
        return self


def test_hosted_module_is_handed_back_to_its_home_device(monkeypatch):
    """The census flow evaluates the same module again for the next sample, so
    a borrowed module must be returned where it was found — not parked on cpu."""
    module = _HomeReportingLinear()
    op = _op(module, 2, src_node=7)
    home = torch.device("cuda:3")
    monkeypatch.setattr(
        "mimarsinan.chip_simulation.hybrid_run.host_compute._module_home_device",
        lambda _m: home,
    )
    buf = {7: np.zeros((1, 2), dtype=np.float64)}
    execute_compute_op_numpy(op, np.zeros((1, 2)), buf, device=None)
    assert module.to_targets[-1] == home


def test_module_home_device_reports_none_when_there_is_nothing_to_restore():
    from mimarsinan.chip_simulation.hybrid_run.host_compute import (
        _module_home_device,
    )

    assert _module_home_device(nn.Identity()) is None
    assert _module_home_device(nn.Linear(2, 2)) == next(
        nn.Linear(2, 2).parameters()
    ).device


# ---------------------------------------------------------------------------
# the split-neuron boundary itself: the numpy and torch twins are one transcode
# ---------------------------------------------------------------------------


def _split_producer_input_map():
    """A consumer reassembling a producer that was TILED across two cores.

    This is the reported vehicle's hop1: a 128-neuron perceptron that exceeded
    ``max_neurons`` and landed as nodes 1 and 2 of 64 neurons each.
    """
    return [_seg_io_slice(1, 0, 4), _seg_io_slice(2, 4, 4)]


def test_split_producer_reassembles_identically_in_both_twins():
    """The suspected seam that is NOT the defect, pinned so it stays that way."""
    rng = np.random.default_rng(0)
    frag_a = rng.random((1, 4))
    frag_b = rng.random((1, 4))
    input_map = _split_producer_input_map()
    divisors = {1: 1.8360226154327393, 2: 1.8360226154327393}

    buf_np = {1: frag_a, 2: frag_b}
    seg_np = assemble_segment_input_numpy(
        input_map, buf_np, num_samples=1, dtype=np.float64,
    )
    seg_np = normalize_boundary_slices_numpy(input_map, seg_np, divisors)

    buf_t = {1: torch.tensor(frag_a, dtype=torch.float64),
             2: torch.tensor(frag_b, dtype=torch.float64)}
    seg_t = assemble_segment_input_torch(
        input_map, buf_t, 1, torch.device("cpu"), torch.float64,
    )
    seg_t = normalize_boundary_slices_torch(input_map, seg_t, divisors)

    np.testing.assert_array_equal(seg_np, seg_t.numpy())


@pytest.mark.parametrize("T", [4, 8, 16])
def test_split_producer_boundary_counts_are_identical_in_both_twins(T):
    """Same rates through both encoders ⇒ the SAME spike counts, bit for bit.

    Rates are placed ON the wire grid and one ULP either side of the comb's
    half-integer ties — where a quantization-aware-trained network actually
    lands, and where an f32 landing used to add a spike on the numpy side.
    """
    grid = [k / (2.0 * T) for k in range(2 * T + 1)]
    rates = np.array([[
        v for k in grid
        for v in (k, np.nextafter(k, 0.0), np.nextafter(k, 1.0))
    ]], dtype=np.float64)
    rates = np.clip(rates, 0.0, 1.0)

    ref = uniform_spike_train(
        torch.tensor(rates, dtype=torch.float64), T,
    ).sum(dim=0)[0].to(torch.int64).numpy()
    actual = encode_segment_input_numpy(rates, T, "Uniform")[0].sum(axis=1)

    np.testing.assert_array_equal(ref, actual.astype(np.int64))
