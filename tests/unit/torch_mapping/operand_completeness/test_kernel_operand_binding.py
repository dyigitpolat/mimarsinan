"""A declared kernel rule reads its call bound to the kernel's signature, and a stated extent is cross-checked at two
batches at least (review round 16).

The round-15 reviewer's witnesses (``review_r15/scripts/r15_fold.py``, ``r15_fold2.py``, ``r15_rules.py``): the rules
read a call's tensor operands in the order the call passed them, keywords in their order, so ``mha(key=kv, query=q,
value=kv)``, ``torch.matmul(other=W, input=y)``, ``lstm(hx=(h0, c0), input=q)`` and ``TransformerDecoderLayer(memory=m,
tgt=q)`` stated the other operand's shape, and ``nn.Transformer`` (rule ``SAME``) stated ``src``'s. The misstated
constant 7 (a ``(7, 1, 8)`` parameter) agreed with torch's run at the hint batch 7, the only cross-check, and a scalar
equal to 1 at batches 1, 6 and 7 and to 2 elsewhere then sized a factory whose extent folded: certified, MVM stage 2
passing, and wrong by a factor of 2 at 60 of 64 batches.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Opaque, affine
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.acceptance import agrees
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernel_rules import einsum, matmul
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernels import rule_of, same
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Call, Shape
from mimarsinan.torch_mapping.value_parity import relative_error

from .harness import CONTRACTS, EXACT, assert_converted_exact, convert, native

ALL_BATCHES = tuple(range(1, 65))


def _pick(n):
    """1 at batches 1, 6 and 7 (the probe's and the hint), 2 elsewhere."""
    return 1 + ((n - 1) * (n - 6) * (n - 7) != 0)


def _pick_mod(n):
    return 1 + (n % 6 >= 2)


class _Fc(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.fc = nn.Linear(8, width)


class MHAKeyFirstFold(_Fc):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(8, 2)
        self.kv = nn.Parameter(torch.randn(7, 1, 8))

    def forward(self, x):
        y = self.fc(x)
        out = self.mha(key=self.kv, query=y.unsqueeze(1), value=self.kv)[0]
        z = torch.zeros(_pick(out.shape[0]), dtype=y.dtype)
        return y * z.shape[0]


class MHAKeyFirstMod(MHAKeyFirstFold):
    def forward(self, x):
        y = self.fc(x)
        out = self.mha(key=self.kv, query=y.unsqueeze(1), value=self.kv)[0]
        return y * torch.zeros(_pick_mod(out.shape[0]), dtype=y.dtype).shape[0]


class TransformerFold(_Fc):
    def __init__(self):
        super().__init__()
        self.tr = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16,
                                 dropout=0.0)
        self.src = nn.Parameter(torch.randn(7, 1, 8))

    def forward(self, x):
        y = self.fc(x)
        out = self.tr(self.src, y.unsqueeze(1))
        z = torch.zeros(_pick(out.shape[0]), dtype=y.dtype)
        return y * z.shape[0]


class DecoderMemoryFirstFold(_Fc):
    def __init__(self):
        super().__init__()
        self.dec = nn.TransformerDecoderLayer(8, 2, 16, dropout=0.0)
        self.mem = nn.Parameter(torch.randn(7, 1, 8))

    def forward(self, x):
        y = self.fc(x)
        out = self.dec(memory=self.mem, tgt=y.unsqueeze(1))
        z = torch.zeros(_pick(out.shape[0]), dtype=y.dtype)
        return y * z.shape[0]


class LSTMStateFirstFold(_Fc):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 8, num_layers=7)
        self.h0 = nn.Parameter(torch.randn(7, 1, 8))
        self.c0 = nn.Parameter(torch.randn(7, 1, 8))

    def forward(self, x):
        y = self.fc(x)
        out = self.lstm(hx=(self.h0, self.c0), input=y.unsqueeze(1))[0]
        z = torch.zeros(_pick(out.shape[0]), dtype=y.dtype)
        return y * z.shape[0]


class MatmulKwFold(_Fc):
    def __init__(self):
        super().__init__(7)
        self.w = nn.Parameter(torch.randn(7, 7))

    def forward(self, x):
        y = self.fc(x)
        out = torch.matmul(other=self.w, input=y)
        z = torch.zeros(_pick(out.shape[0]), dtype=y.dtype)
        return y * z.shape[0]


WITNESSES = [MHAKeyFirstFold, MHAKeyFirstMod, MatmulKwFold, TransformerFold, DecoderMemoryFirstFold,
             LSTMStateFirstFold]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", WITNESSES, ids=lambda f: f.__name__)
def test_a_keyword_operand_order_is_exact_at_every_batch(factory, contract):
    """Each witness is exact at batches 1 to 64 under both contracts, and ``z.shape[0]`` is not folded."""
    model = native(factory)
    flow = convert(model, (8,), contract)
    assert not [name for name, (operator, _) in flow.shape_literals.items() if operator.startswith("getitem")], \
        flow.shape_literals
    for batch in ALL_BATCHES:
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            err = relative_error(flow(x.clone()), model(x.clone())).relative_error
        assert err <= EXACT, (batch, err)


class TransformerSrcParam(_Fc):
    """nn.Transformer(src, tgt) returns tgt's shape, (B, 1, 8) here, not src's (7, 1, 8)."""

    def __init__(self):
        super().__init__()
        self.tr = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16,
                                 dropout=0.0)
        self.src = nn.Parameter(torch.randn(7, 1, 8))

    def forward(self, x):
        return self.tr(self.src, self.fc(x).unsqueeze(1)).squeeze(1)


class GLUBatchView(_Fc):
    def __init__(self):
        super().__init__(6)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        return self.glu(self.fc(x)).view(-1, 3)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [TransformerSrcParam, GLUBatchView], ids=lambda f: f.__name__)
def test_a_module_whose_output_is_not_its_first_operands_shape_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=(1, 2, 3, 5, 7, 13))


Q, KV = Shape((affine(1, 0), 1, 8)), Shape((7, 1, 8))


def test_a_call_is_bound_to_its_signature():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.binding import bound

    mha = nn.MultiheadAttention(8, 2)
    holds = lambda value: isinstance(value, Shape)  # noqa: E731
    assert bound("call_module", None, mha, (), {"key": KV, "query": Q, "value": KV}, holds) == ((Q, KV, KV), {})
    lstm = nn.LSTM(8, 8)
    assert bound("call_module", None, lstm, (), {"hx": (KV, KV), "input": Q}, holds) == ((Q, (KV, KV)), {})
    decoder = nn.TransformerDecoderLayer(8, 2, 16)
    assert bound("call_module", None, decoder, (), {"memory": KV, "tgt": Q}, holds) == ((Q, KV), {})
    assert bound("call_function", torch.matmul, None, (), {"other": KV, "input": Q}, holds) == ((Q, KV), {})
    assert bound("call_method", "addmm", None, (Q,), {"mat2": KV, "mat1": Q}, holds) == ((Q, Q, KV), {})
    # a parameter left out keeps every later one a keyword; a call passed positionally is unchanged
    assert bound("call_module", None, mha, (Q, KV, KV), {"need_weights": False}, holds) == (
        (Q, KV, KV), {"need_weights": False})
    assert bound("call_function", torch.matmul, None, (Q, KV), {}, holds) == ((Q, KV), {})
    # no signature binds: a tensor keyword operand makes the call unreadable, a number keeps it as passed
    assert bound("call_function", torch.matmul, None, (), {"other": KV, "first": Q}, holds) is None
    assert bound("call_function", torch.matmul, None, (Q,), {"first": 2}, holds) == ((Q,), {"first": 2})


def test_the_rules_read_the_bound_operands():
    mha = nn.MultiheadAttention(8, 2)
    name, rule = rule_of("call_module", "mha", mha)
    assert rule(Call(name, (Q, KV, KV), {}, ((7, 1, 8), (1, 7, 7)), mha))[0] == Q.dims
    assert matmul(Call("matmul", (Shape((BATCH, 7)), Shape((7, 5))))) == [(BATCH, 5)]
    transformer = nn.Transformer(8, 2, 1, 1, 16)
    name, rule = rule_of("call_module", "tr", transformer)
    assert rule(Call(name, (KV, Q), {}, ((7, 1, 8),), transformer)) == [Q.dims]
    assert rule_of("call_module", "dec", nn.TransformerDecoderLayer(8, 2, 16))[1] is same


def test_glu_halves_its_axis():
    glu = nn.GLU(dim=-1)
    name, rule = rule_of("call_module", "glu", glu)
    assert rule(Call(name, (Shape((BATCH, 6)),), {}, ((7, 3),), glu)) == [(BATCH, 3)]
    halved = rule(Call(name, (Shape((7, BATCH)),), {}, ((7, 3),), glu))[0][1]
    assert isinstance(halved, Opaque)  # B / 2 is not affine in B
    assert rule(Call(name, (Shape((BATCH, affine(2, 4))),), {}, ((7, 9),), glu)) == [(BATCH, affine(1, 2))]
    name, rule = rule_of("call_function", F.glu, None)
    assert rule(Call(name, (Shape((BATCH, 8)), 1))) == [(BATCH, 4)]


def test_an_einsum_letter_broadcasts():
    found = einsum(Call("einsum", ("bi,bi->b", Shape((BATCH, 6)), Shape((7, 6)))))
    assert isinstance(found[0][0], Opaque)  # torch gives 7 at B = 1, raises elsewhere
    assert einsum(Call("einsum", ("bi,bi->b", Shape((BATCH, 6)), Shape((1, 6))))) == [(BATCH,)]


def test_a_stated_extent_is_cross_checked_at_two_batches():
    assert not agrees(7, {7: 7})  # one batch: a constant where torch's extent is B would pass
    assert not agrees(7, {1: 1, 7: 7})
    assert agrees(7, {1: 7, 2: 7, 7: 7})
    assert agrees(BATCH, {1: 1, 2: 2, 3: 3, 7: 7})
    assert agrees(Opaque("a sample"), {7: 7})


_SAME_CASES = {  # class name: (constructor arguments, input shape); default ((), (3, 4))
    "Threshold": ((0.1, 0.0), (3, 4)), "Softmax": ((1,), (3, 4)), "Softmin": ((1,), (3, 4)),
    "LogSoftmax": ((1,), (3, 4)), "Softmax2d": ((), (3, 4, 5, 5)), "Dropout1d": ((), (3, 4, 5)),
    "Dropout2d": ((), (3, 4, 5, 5)), "Dropout3d": ((), (3, 4, 5, 5, 5)), "GroupNorm": ((2, 4), (3, 4)),
    "LayerNorm": ((5,), (3, 4, 5)), "RMSNorm": ((5,), (3, 4, 5)), "LocalResponseNorm": ((2,), (3, 4, 5)),
    "CrossMapLRN2d": ((2,), (3, 4, 5, 5)), "BatchNorm1d": ((4,), (3, 4, 5)), "BatchNorm2d": ((4,), (3, 4, 5, 5)),
    "BatchNorm3d": ((4,), (3, 4, 5, 5, 5)), "InstanceNorm1d": ((4,), (3, 4, 5)), "InstanceNorm2d": ((4,), (3, 4, 5, 5)),
    "InstanceNorm3d": ((4,), (3, 4, 5, 5, 5)), "LazyBatchNorm1d": ((), (3, 4, 5)),
    "LazyBatchNorm2d": ((), (3, 4, 5, 5)),
    "LazyBatchNorm3d": ((), (3, 4, 5, 5, 5)), "LazyInstanceNorm1d": ((), (3, 4, 5)),
    "LazyInstanceNorm2d": ((), (3, 4, 5, 5)), "LazyInstanceNorm3d": ((), (3, 4, 5, 5, 5)),
    "SyncBatchNorm": ((4,), (3, 4)),
    "TransformerEncoderLayer": ((8, 2, 16), (5, 3, 8)), "TransformerDecoderLayer": ((8, 2, 16), (5, 3, 8)),
    "Transformer": ((8, 2, 1, 1, 16), (5, 3, 8)), "MultiheadAttention": ((8, 2), (5, 3, 8)),
}


def _same_ruled_classes():
    import torch.nn.modules.activation as activation
    import torch.nn.modules.batchnorm as batchnorm
    import torch.nn.modules.dropout as dropout
    import torch.nn.modules.instancenorm as instancenorm
    import torch.nn.modules.normalization as normalization
    import torch.nn.modules.transformer as transformer
    for where in (activation, dropout, normalization, batchnorm, instancenorm, transformer):
        for name, kind in sorted(vars(where).items()):
            if isinstance(kind, type) and issubclass(kind, nn.Module) and kind.__module__ == where.__name__ \
                    and not name.startswith("_"):
                yield kind


@pytest.mark.parametrize("kind", list(_same_ruled_classes()), ids=lambda kind: kind.__name__)
def test_every_module_of_the_rule_same_keeps_its_first_operands_shape(kind):
    """Every public class of the torch modules whose classes the rule ``SAME`` covers: its rule is ``SAME`` only where
    its output has its first operand's shape (``nn.Transformer``, ``nn.GLU`` and attention have their own)."""
    stacks = {nn.TransformerEncoder: nn.TransformerEncoderLayer, nn.TransformerDecoder: nn.TransformerDecoderLayer}
    if kind in stacks:
        arguments, shape = ((stacks[kind](8, 2, 16), 1), (5, 3, 8))
    else:
        arguments, shape = _SAME_CASES.get(kind.__name__, ((), (3, 4)))
    module = kind(*arguments).eval()
    found = rule_of("call_module", "m", module)
    if kind in (nn.Transformer, nn.GLU, nn.MultiheadAttention):
        assert found is not None and found[1] is not same, kind
        return
    assert found is not None and found[1] is same, kind
    x = torch.randn(shape)
    with torch.no_grad():
        decoder = kind in (nn.TransformerDecoderLayer, nn.TransformerDecoder)
        out = module(x, torch.randn(4, 3, 8)) if decoder else module(x)
    assert tuple(out.shape) == shape, (kind, tuple(out.shape))
