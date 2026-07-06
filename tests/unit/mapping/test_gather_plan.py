"""Golden equivalence: precomputed gather plans replicate the per-source walk bit-exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir.gather_plan import (
    gather_inputs_reference,
    gather_plan_for,
)
from mimarsinan.mapping.ir.types import ComputeOp, IRSource


def _sources(rng, n, *, n_input=16, buffer_shapes=None):
    buffer_shapes = buffer_shapes or {}
    kinds = ["off", "input", "on"] + (["buf"] if buffer_shapes else [])
    out = []
    for _ in range(n):
        kind = rng.choice(kinds)
        if kind == "off":
            out.append(IRSource(-1, 0))
        elif kind == "input":
            out.append(IRSource(-2, int(rng.integers(0, n_input))))
        elif kind == "on":
            out.append(IRSource(-3, 0))
        else:
            node_id = int(rng.choice(list(buffer_shapes)))
            out.append(IRSource(node_id, int(rng.integers(0, buffer_shapes[node_id]))))
    return np.array(out, dtype=object)


def _node(sources):
    return ComputeOp(
        id=99,
        name="op",
        input_sources=sources.reshape(-1, 1),
        op_type="identity",
        params={"module": torch.nn.Identity()},
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plan_matches_reference_walk(seed, dtype):
    rng = np.random.default_rng(seed)
    buffer_shapes = {3: 12, 7: 5, 11: 30}
    node = _node(_sources(rng, 64, n_input=16, buffer_shapes=buffer_shapes))
    batch = 4
    x = torch.tensor(rng.standard_normal((batch, 16)), dtype=dtype)
    buffers = {
        nid: torch.tensor(rng.standard_normal((batch, w)), dtype=dtype)
        for nid, w in buffer_shapes.items()
    }
    ref = gather_inputs_reference(node, x, buffers)
    got = node.gather_inputs(x, buffers)
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


def test_plan_all_off_sources():
    node = _node(np.array([IRSource(-1, 0)] * 5, dtype=object))
    x = torch.ones(2, 3)
    ref = gather_inputs_reference(node, x, {})
    got = node.gather_inputs(x, {})
    assert torch.equal(got, ref)
    assert got.abs().sum() == 0


def test_plan_input_only_and_always_on():
    sources = np.array(
        [IRSource(-2, 2), IRSource(-3, 0), IRSource(-2, 0)], dtype=object
    )
    node = _node(sources)
    x = torch.tensor([[10.0, 20.0, 30.0]])
    ref = gather_inputs_reference(node, x, {})
    got = node.gather_inputs(x, {})
    assert torch.equal(got, ref)
    assert got.tolist() == [[30.0, 1.0, 10.0]]


def test_plan_is_cached_per_node():
    rng = np.random.default_rng(0)
    node = _node(_sources(rng, 8, n_input=4))
    assert gather_plan_for(node) is gather_plan_for(node)


def test_plan_referenced_buffer_ids():
    rng = np.random.default_rng(1)
    buffer_shapes = {5: 6, 9: 4}
    node = _node(_sources(rng, 40, n_input=4, buffer_shapes=buffer_shapes))
    plan = gather_plan_for(node)
    expected = {
        int(s.node_id)
        for s in node.input_sources.flatten()
        if not (s.is_off() or s.is_input() or s.is_always_on())
    }
    assert set(plan.referenced_node_ids) == expected


def test_float64_input_downcast_matches_reference():
    # The reference walk writes into a default-dtype zeros tensor (float32),
    # silently downcasting float64 inputs; the plan must reproduce that.
    sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)
    node = _node(sources)
    x = torch.tensor([[1.00000000001, 2.0]], dtype=torch.float64)
    ref = gather_inputs_reference(node, x, {})
    got = node.gather_inputs(x, {})
    assert got.dtype == ref.dtype == torch.float32
    assert torch.equal(got, ref)
