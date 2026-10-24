"""N2 design §3.2 (host-op determinism): the host ops between operand segments, run in fp64 by the executor's own
seam (``value_flow.execute_host_op``) on 8-bit grid values, are bitwise invariant to batch composition and thread
count, so the int64 golden's replay in the executor's batch equals a replay sample by sample. Shapes of every
token count a certificate ran at: ViT-Lite-7/4 (65), ViT-B/16 at 224 (197) and at 96 (37), ViT-B/32 at 224 (50).
The operand certificate's batch-of-one edge checks the same per program it certifies.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.value_run.value_flow import execute_host_op
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

#: (tokens, width, heads): ViT-Lite-7/4 on CIFAR (65 = 64 patches + the class token), ViT-B/16 at 224 and at 96,
#: ViT-B/32 at 224 (the token count of ViT-B/16 at 112 too).
SHAPES = {"vit_lite_7_4": (65, 256, 4), "vit_b_16_224": (197, 768, 12), "vit_b_16_96": (37, 768, 12),
          "vit_b_32_224": (50, 768, 12)}
BATCHES = {"vit_lite_7_4": (1, 7, 64), "vit_b_16_224": (1, 7, 16), "vit_b_16_96": (1, 7, 16),
           "vit_b_32_224": (1, 7, 16)}
STEP = 0.0173


def _op(module, shapes):
    count = sum(int(np.prod(s)) for s in shapes)
    params = {"module": module}
    if len(shapes) == 1:
        params["input_shape"] = tuple(shapes[0])
    else:
        params["input_shapes"] = [tuple(s) for s in shapes]
    return ComputeOp(id=7, name="host", input_sources=np.array([IRSource(-2, i) for i in range(count)], dtype=object),
                     op_type=type(module).__name__, params=params,
                     input_shape=tuple(shapes[0]) if len(shapes) == 1 else None, output_shape=tuple(shapes[0]))


def _ops(key):
    tokens, width, heads = SHAPES[key]
    scores = (heads, tokens, tokens)
    norm = nn.LayerNorm(width)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width, generator=torch.Generator().manual_seed(1)))
        norm.bias.copy_(torch.randn(width, generator=torch.Generator().manual_seed(2)))

    def relu_follower(x):
        return F.relu(x) / tokens

    return {
        "softmax": _op(ComputeAdapter(torch.Tensor.softmax, extra_args=(-1,)), [scores]),
        "relu_follower": _op(ComputeAdapter(relu_follower), [scores]),
        "layernorm": _op(norm, [(tokens, width)]),
        "gelu": _op(ComputeAdapter(F.gelu), [(tokens, 2 * width)]),
        "residual_add": _op(ComputeAdapter(operator.add), [(tokens, width), (tokens, width)]),
    }


def _grid_values(op, batch, seed=0):
    count = op.input_sources.size
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-127, 128, (batch, count), generator=generator).to(torch.float64) * STEP


def _run(op, x):
    with torch.no_grad():
        return execute_host_op(op, x, {-2: x}, dtype=torch.float64, twins={})


@pytest.mark.parametrize("key", sorted(SHAPES))
def test_host_ops_are_bitwise_invariant_to_batch_composition_and_threads(key):
    previous = torch.get_num_threads()
    try:
        for name, op in _ops(key).items():
            batches = BATCHES[key]
            x = _grid_values(op, max(batches))
            torch.set_num_threads(1)
            alone = torch.cat([_run(op, x[i:i + 1]) for i in range(x.shape[0])])
            for batch in batches:
                for start in range(0, x.shape[0], batch):
                    chunk = x[start:start + batch]
                    assert torch.equal(_run(op, chunk), alone[start:start + batch]), (key, name, batch)
            torch.set_num_threads(8)
            assert torch.equal(_run(op, x), alone), (key, name, "8 threads")
    finally:
        torch.set_num_threads(previous)
