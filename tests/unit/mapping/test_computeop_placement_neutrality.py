"""A ComputeOp BORROWS ``params['module']`` from the live model.

The deployed executor places that module on the buffer device on purpose; the
pure analyses (liveness transfer, constant lattice, the zero-preservation
certificate) probe the same op with fabricated tensors and must leave the model
exactly where it was. Before the placement-neutral seam existed they did not,
which left the model half-migrated: on a CUDA run the CPU probe dragged the
host classifier to CPU while every other parameter stayed on cuda:0, and the
next consumer to touch the flow died inside ``addmm`` with a device mismatch.

``meta`` stands in for "some other device" so the split is observable without a
GPU: relocating a real module onto it destroys the parameter data, which is the
same corruption class a stray ``.to(cpu)`` inflicts on a CUDA model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir.types import ComputeOp
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.pruning.certificate.zero_preserving import (
    is_zero_preserving_host_op,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_transfer import (
    derive_constant_outputs,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import (
    derive_liveness_transfer,
)

META = torch.device("meta")
CPU = torch.device("cpu")


def _op(module, n_in=3, n_out=2, name="classifier", op_type="linear"):
    return ComputeOp(
        id=1,
        name=name,
        input_sources=np.array(
            [IRSource(node_id=-2, index=i) for i in range(n_in)], dtype=object
        ),
        op_type=op_type,
        params={"module": module, "input_shape": (n_in,), "module_kwargs": {}},
        input_shape=(n_in,),
        output_shape=(n_out,),
    )


def _placement(module):
    return {name: p.device for name, p in module.named_parameters()}


def test_analysis_probe_leaves_the_borrowed_module_where_it_was():
    """The exact defect: probing on a foreign device relocated the model."""
    linear = nn.Linear(3, 2)
    weights_before = linear.weight.detach().clone()
    op = _op(linear)

    op.probe_on_gathered(torch.zeros(1, 3, device=META))

    assert _placement(linear) == {"weight": CPU, "bias": CPU}
    torch.testing.assert_close(linear.weight.detach(), weights_before)


def test_analysis_probe_returns_the_result_on_the_probe_device():
    """Placement neutrality must not cost the caller its result."""
    op = _op(nn.Linear(3, 2))
    out = op.probe_on_gathered(torch.zeros(1, 3, device=META))
    assert out.device.type == "meta"
    assert tuple(out.shape) == (1, 2)


def test_analysis_probe_restores_buffers_too():
    """Buffers are model state exactly like parameters."""
    norm = nn.BatchNorm1d(3).eval()
    op = _op(norm, n_in=3)

    op.probe_on_gathered(torch.zeros(1, 3, device=META))

    assert all(b.device == CPU for b in norm.buffers())


def test_deployed_execution_still_places_the_module_on_the_buffer_device():
    """The deployment seam OWNS placement — that behaviour is unchanged."""
    linear = nn.Linear(3, 2)
    op = _op(linear)

    op.execute_on_gathered(torch.zeros(1, 3, device=META))

    assert _placement(linear) == {"weight": META, "bias": META}


@pytest.mark.parametrize(
    "analysis",
    [
        pytest.param(is_zero_preserving_host_op, id="zero_preserving"),
        pytest.param(derive_liveness_transfer, id="liveness_transfer"),
        pytest.param(
            lambda op: derive_constant_outputs(
                op, derive_liveness_transfer(op), [0.0, 0.0, 0.0]
            ),
            id="constant_lattice",
        ),
    ],
)
def test_analyses_request_the_placement_neutral_seam(analysis):
    """Every pure analysis probes through ``probe_on_gathered``.

    A future analysis reaching for the deployment seam re-opens the defect, so
    the contract is pinned on the call, not only on the outcome.
    """
    op = _op(nn.ReLU(), n_out=3, op_type="relu")
    seams: list[str] = []
    deployed, probe = op.execute_on_gathered, op.probe_on_gathered

    def _spy(tag, fn):
        def _call(*args, **kwargs):
            seams.append(tag)
            return fn(*args, **kwargs)

        return _call

    op.execute_on_gathered = _spy("deployed", deployed)
    op.probe_on_gathered = _spy("probe", probe)

    analysis(op)

    assert seams, "the analysis never executed the op's seam"
    assert "deployed" not in seams, (
        f"analysis reached for the deployment seam: {seams}"
    )
