"""Values of a program with an attention or recurrent leaf (declared change of review round 4, scope of round 5, the
whole program from round 7; restated in review round 14).

The fake kernel of a torch attention leaf (a batch-first ``nn.MultiheadAttention``, ``nn.TransformerEncoderLayer``)
requests ``Eq(B, 7)`` at the symbolic hint. Rounds 4 to 13 left every value of such a program without a proof, so its
static shape chains kept the w0-base host ops.

[restated in review round 14: no fold rests on torch's guards now. The leaf's declared batch rule (the output has the
query's shape, ``symbolic/batch_domain/kernels.py``) computes its shape in the affine batch domain, so the chain after
it folds again, as on the round-3 tree: exact in the converted model at every batch, its identity value program exact,
and a ``torch.cat`` axis computed from the rank folded and exact.]
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE, BatchLayouts
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, native, quietly


def _leaf(kind: str) -> nn.Module:
    if kind == "mha":
        return nn.MultiheadAttention(4, 1, batch_first=True)
    return nn.TransformerEncoderLayer(4, 1, 8, dropout=0.0, batch_first=True)


class AfterLeaf(nn.Module):
    """``h`` = an attention leaf on (B, 2, 4) tokens, then a shape chain of ``h``."""

    FORMS = {
        "scale": lambda h: (h * h.shape[-1] ** -0.5).flatten(1),
        "layer_norm": lambda h: F.layer_norm(h, h.shape[-1:]).flatten(1),
        "softmax_dim": lambda h: F.softmax(h, dim=h.dim() - 1).flatten(1),
        "size_scale": lambda h: torch.cat([h, h], dim=-1).flatten(1) * h.size(-1),
        "cat_dim": lambda h: torch.cat([h, h], h.dim() - 1).flatten(1),
    }

    def __init__(self, kind: str, how: str):
        super().__init__()
        self.kind, self.how = kind, how
        self.fc = nn.Linear(8, 8)
        self.leaf = _leaf(kind)

    def forward(self, x):
        y = self.fc(x).reshape(x.size(0), 2, 4)
        h = self.leaf(y, y, y, need_weights=False)[0] if self.kind == "mha" else self.leaf(y)
        return self.FORMS[self.how](h)


LEAVES = ["mha", "encoder"]
# (leaf, form) -> the literals the chain folds (as on the round-3 tree; rounds 4 to 13 kept the w0-base host ops)
CHAIN_LITERALS = {
    ("mha", "scale"): {"getitem_1": ("getitem", "4"), "pow_1": ("pow", "0.5")},
    ("mha", "layer_norm"): {"getitem_1": ("getitem", "torch.Size([4])")},
    ("mha", "softmax_dim"): {"dim": ("dim", "3"), "sub": ("sub", "2")},
    ("mha", "size_scale"): {"size_1": ("size", "4")},
    ("mha", "cat_dim"): {"dim": ("dim", "3"), "sub": ("sub", "2")},
    ("encoder", "scale"): {"getitem": ("getitem", "4"), "pow_1": ("pow", "0.5")},
    ("encoder", "layer_norm"): {"getitem": ("getitem", "torch.Size([4])")},
    ("encoder", "softmax_dim"): {"dim": ("dim", "3"), "sub": ("sub", "2")},
    ("encoder", "size_scale"): {"size_1": ("size", "4")},
    ("encoder", "cat_dim"): {"dim": ("dim", "3"), "sub": ("sub", "2")},
}
GUARD_BATCHES = (1, 2, 3, 6, 7, 8, 13)


def test_every_value_of_a_program_with_an_attention_leaf_is_proved():
    """The leaf's rule states its output: every value before and after it is proved. [restated in review round 14:
    rounds 7 to 13 proved none]"""
    for kind in LEAVES:
        torch.manual_seed(0)
        gm = trace_model(AfterLeaf(kind, "scale").double().eval(), (8,))
        layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 8), kind
        assert layouts.layout(nodes["leaf"] if kind == "encoder" else nodes["getitem"]) == (BATCH_SIZE, 2, 4), kind
        assert layouts.opaque() is None, kind


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("leaf,how", sorted(CHAIN_LITERALS))
def test_a_shape_chain_after_the_leaf_folds_and_is_exact(leaf, how, contract):
    flow = assert_converted_exact(native(lambda: AfterLeaf(leaf, how)), (8,), contract, batches=GUARD_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == CHAIN_LITERALS[(leaf, how)] and flow.shape_opaque is None


@pytest.mark.parametrize("leaf", LEAVES)
@pytest.mark.parametrize("how", sorted(AfterLeaf.FORMS))
def test_a_shape_chain_after_the_leaf_passes_stage_2(leaf, how):
    """Round 3 certified these at stage 2 (a literal); rounds 4 to 13 refused the run-time shape value by name."""
    flow = quietly(convert_torch_model, native(lambda: AfterLeaf(leaf, how)), (8,), 10, packaging=PACKAGINGS["mvm"],
                   value_program_stage=True)
    assert flow.conversion_certificate.certified
