"""The probe's certificate is bound to the returned program (review round 13; restated in review round 14).

Round 12's probe converted its fp64 deep copy of the model (``fp64_snapshot``) while ``convert_torch_model`` returned
the original's conversion, bound by the literals each folds only: a model whose public ``__deepcopy__`` returns another
program was certified with the copy's program (DeepcopySwitchNest: wrong at batches 6 to 11 by rel 0.33; the
reviewer's ``r12_deepcopy.py``). Round 13 compared the two traced programs node by node and refused such a model by
name (``WitnessProgramMismatchError``).

[restated in review round 14: the witness program is now derived from the returned conversion's own traced graph
(``witness.fp64_program``: FX's copy of the traced ``GraphModule``, its parameters and buffers cast to fp64), and the
native reference is the model's own call on those fp64 tensors (``witness.NativeFp64``), or the witness program
itself where it is the model's own call bit for bit in the model's dtype (``witness.traced_as_native``: an fp32 model
reading its own dtype, the round-13 reviewer's ``r13_dtype.py``); the user's root object is never copied. So a root
``__deepcopy__`` changes nothing: each DeepcopySwitch model is certified for its own program, exact at every batch.
The node-by-node comparison stays as the binding of the two programs, each FX tensor constant compared by value
(DeepcopySwitchConst, the round-13 reviewer's: its constant was compared by shape only).]
"""

from __future__ import annotations

import collections
import contextlib
import copy
import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.value_parity.verdict import WitnessProgramMismatchError

PACKAGINGS = {"spiking": SPIKING_PACKAGING, "mvm": MVM_PACKAGING}


class NestOffset(nn.Module):
    def __init__(self, k: int = -12):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.k = k

    def forward(self, x):
        y = self.fc(x)
        b = x.size(0) + self.k
        return y * torch.tile(y, (1, 2 + (b // -7) // -3 - b // 21)).shape[1]


class DeepcopySwitchNest(NestOffset):
    """Its deep copy is the same model with the offset 0: another program folding the same literals."""

    def __deepcopy__(self, memo):
        new = NestOffset.__new__(type(self))
        nn.Module.__init__(new)
        new.fc = copy.deepcopy(self.fc, memo)
        new.k = 0
        return new


class DeepcopySwitchScale(nn.Module):
    def __init__(self, s: float = 2.0):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.s = s

    def forward(self, x):
        return self.fc(x) * self.s

    def __deepcopy__(self, memo):
        new = DeepcopySwitchScale.__new__(type(self))
        nn.Module.__init__(new)
        new.fc = copy.deepcopy(self.fc, memo)
        new.s = 3.0
        return new


class DeepcopySwitchWeight(nn.Module):
    """The same program; the copy's Linear weight is another (0.25 everywhere, for the copy of a copy too)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return F.relu(self.fc(x))

    def __deepcopy__(self, memo):
        new = DeepcopySwitchWeight.__new__(type(self))
        nn.Module.__init__(new)
        new.fc = copy.deepcopy(self.fc, memo)
        with torch.no_grad():
            new.fc.weight.fill_(0.25)
        return new


class DeepcopySwitchStride(nn.Module):
    """The same graph; the copy's convolution has another stride (a module configuration, no FX literal)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x).flatten(1)

    def __deepcopy__(self, memo):
        new = DeepcopySwitchStride.__new__(type(self))
        nn.Module.__init__(new)
        new.conv = copy.deepcopy(self.conv, memo)
        new.conv.stride = (2, 2)
        return new


class Ordinary(nn.Module):
    """A faithful deep copy: convolution, BatchNorm (the probe draws its statistics), Linear, a plain tensor attribute."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 5)
        self.register_buffer("offset", torch.full((5,), 0.25))

    def forward(self, x):
        y = F.relu(self.bn(self.conv(x)))
        return self.fc(F.max_pool2d(y, 2).flatten(1)) + self.offset


def _model(factory):
    torch.manual_seed(0)
    return factory().double().eval()


def _fp32(factory):
    torch.manual_seed(0)
    return factory().eval()


def _convert(model, shape, contract, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_torch_model(model, shape, 10, packaging=PACKAGINGS[contract], **kwargs)


class DeepcopySwitchConst(nn.Module):
    """The round-13 reviewer's: the forward turns a list attribute into an FX tensor constant, and the copy's list is
    another one."""

    def __init__(self, vals=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.vals = list(vals)

    def forward(self, x):
        return self.fc(x) * torch.tensor(self.vals, dtype=torch.float64)

    def __deepcopy__(self, memo):
        new = DeepcopySwitchConst([0.5] * 6)
        new.fc = copy.deepcopy(self.fc, memo)
        return new


SWITCHES = {  # form -> (factory, input shape): refused by name on the round-13 tree, certified for the copy before
    "nest": (DeepcopySwitchNest, (8,)),
    "scale": (DeepcopySwitchScale, (8,)),
    "weight": (DeepcopySwitchWeight, (8,)),
    "stride": (DeepcopySwitchStride, (1, 4, 4)),
    "const": (DeepcopySwitchConst, (8,)),
}


@pytest.mark.parametrize("contract", sorted(PACKAGINGS))
@pytest.mark.parametrize("how", sorted(SWITCHES))
def test_a_root_whose_deep_copy_is_another_program_is_certified_for_its_own_program(how, contract):
    """The model object itself is converted (no harness copy, which would switch the program first): certified, and
    the returned conversion equals the model at batches 1 to 13."""
    factory, shape = SWITCHES[how]
    model = _model(factory)
    flow = _convert(model, shape, contract).eval()
    assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()
    for batch in range(1, 14):
        x = torch.randn(batch, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            got, want = flow(x), model(x)
        assert torch.allclose(got, want, rtol=0, atol=1e-12 * max(1.0, float(want.abs().max())))


def test_a_witness_holding_another_tensor_constant_is_another_program():
    """The binding compares each FX tensor constant by value: a witness program whose constant differs (as a copy of
    DeepcopySwitchConst's would) is another program, named at the node that reads it, and the tensor is listed among
    those the witness holds otherwise."""
    from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
    from mimarsinan.torch_mapping.value_parity.program_binding import (
        program_difference,
        program_record,
        state_differences,
    )
    from mimarsinan.torch_mapping.value_parity.witness import fp64_program

    traced = trace_model(_model(DeepcopySwitchConst), (8,))
    witness = fp64_program(traced, (8,))
    assert program_difference(program_record(traced), program_record(witness), state_differences(traced, witness)) \
        is None
    witness._tensor_constant0 = torch.full_like(witness._tensor_constant0, 0.5)
    altered = state_differences(traced, witness)
    found = program_difference(program_record(traced), program_record(witness), altered)
    assert "_tensor_constant0" in altered and found is not None and found[0] == "_tensor_constant0"


def test_a_permissive_conversion_of_a_switching_root_is_certified():
    """[restated in review round 14: not certified on the round-13 tree, naming ``mul``]"""
    flow = _convert(_model(DeepcopySwitchScale), (8,), "mvm", strict=False)
    assert flow.conversion_certificate.certified


@pytest.mark.parametrize("contract", sorted(PACKAGINGS))
@pytest.mark.parametrize("factory, shape", [(Ordinary, (1, 8, 8)), (NestOffset, (8,))], ids=["ordinary", "nest"])
def test_a_faithful_deep_copy_keeps_its_certified_conversion(factory, shape, contract):
    """The returned conversion of a model whose deep copy is faithful is certified and equals a conversion of the copy
    in every parameter; NestOffset (the integer rewrite of round 13) is exact at batches 1 to 13."""
    model = _model(factory)
    flow = _convert(model, shape, contract).eval()
    assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()
    for batch in range(1, 14):
        x = torch.randn(batch, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            got, want = flow(x), model(x)
        assert torch.allclose(got, want, rtol=0, atol=1e-12 * max(1.0, float(want.abs().max())))


class ShiftedPair(collections.namedtuple("Pair", "a b")):
    """A namedtuple whose constructor shifts its first field: copying an instance rebuilds it through ``__new__``, so
    a plain deep copy holds ``ShiftedPair(13, 0)`` for the model's ``ShiftedPair(7, 0)``."""

    def __new__(cls, a, b):
        return super().__new__(cls, a + 6, b)


class NamedTupleAttribute(nn.Module):
    """The round-12 reviewer's ``NamedTupleRoll``: certified on the round-12 tree for the copy's program (a shift of 13
    against the model's 7), exact on w0-base (converted itself). The probe's copy holds the model's own immutable
    attribute values (``witness.faithful_copy``), so the witness is the model's program."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.shift = ShiftedPair(1, 0)

    def forward(self, x):
        y = self.fc(x)
        return torch.roll(y, shifts=self.shift[:1], dims=(0,)) + y


@pytest.mark.parametrize("contract", sorted(PACKAGINGS))
def test_an_immutable_attribute_a_plain_deep_copy_rebuilds_otherwise_keeps_the_models_program(contract):
    """[restated in review round 14: the witness is the traced graph, where FX baked the shift in as the literal 7]"""
    model = _model(NamedTupleAttribute)
    assert copy.deepcopy(model).shift == (13, 0)
    flow = _convert(model, (8,), contract).eval()
    assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()
    for batch in range(1, 14):
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            assert torch.equal(flow(x), model(x))


class SpectralNormConv(nn.Module):
    """The round-12 reviewer's (``r12_attack.py`` group G4): a Conv2d under ``spectral_norm``, whose pre-hook sets the
    module's ``weight`` to a tensor computed with autograd in the traced run. Exact under both contracts on the round-13
    tree and on ``w0-base``; refused on the round-14 tree (``WitnessProgramError``: the witness program's deep copy
    refused a tensor that is not a graph leaf). The copy holds it detached (``witness.faithful_copy``)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.utils.spectral_norm(nn.Conv2d(1, 3, 3))
        self.fc = nn.Linear(3 * 4 * 4, 5)

    def forward(self, x):
        return self.fc(F.relu(self.conv(x)).flatten(1))


@pytest.mark.parametrize("contract", sorted(PACKAGINGS))
def test_a_computed_tensor_attribute_is_copied_into_the_witness_program(contract):
    model = _model(SpectralNormConv)
    flow = _convert(model, (1, 6, 6), contract).eval()
    assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()
    for batch in range(1, 14):
        x = torch.randn(batch, 1, 6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            got, want = flow(x), model(x)
        assert torch.allclose(got, want, rtol=0, atol=1e-12 * max(1.0, float(want.abs().max())))


# ------------------------------------------------------------------------- a dtype the model reads (review round 14)
class _ParamDtype(nn.Module):
    """The ``self.dtype`` idiom: the parameters' dtype, read where the program is traced."""

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class SoftmaxParamDtype(_ParamDtype):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return F.softmax(self.fc(x), -1, dtype=self.dtype)


class LayerNormFinfoEps(_ParamDtype):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return F.layer_norm(self.fc(x), (6,), eps=torch.finfo(self.dtype).eps)


class OnesLikeParamDtype(_ParamDtype):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return y + torch.ones_like(y, dtype=self.dtype)


DTYPE_IDIOMS = {"softmax": SoftmaxParamDtype, "layer_norm_eps": LayerNormFinfoEps, "ones_like": OnesLikeParamDtype}


@pytest.mark.parametrize("contract", sorted(PACKAGINGS))
@pytest.mark.parametrize("how", sorted(DTYPE_IDIOMS))
def test_an_fp32_model_reading_its_own_dtype_is_certified_for_its_traced_program(how, contract):
    """The round-13 reviewer's ``r13_dtype.py``: exact on w0-base in fp32, refused on the round-13 tree
    (``WitnessProgramMismatchError``: the fp64 deep copy traced the literal ``torch.float64``, the returned conversion
    ``torch.float32``). The witness program is the same traced graph in fp64 now, and it computes the model's own call
    bit for bit in fp32 (``witness.traced_as_native``), so it is the reference: certified, and the returned conversion
    equals the fp32 model at batches 1 to 64."""
    torch.manual_seed(0)
    model = DTYPE_IDIOMS[how]().eval()  # fp32
    flow = _convert(model, (8,), contract).eval()
    assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()
    for batch in range(1, 65):
        x = torch.randn(batch, 8, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            assert torch.equal(flow(x), model(x)), batch


class CountingCalls(nn.Module):
    """A model whose own call is not its traced program: each call scales by the number of calls so far, which the
    trace reads once (``self.calls`` is the literal 1)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.fc(x) * self.calls


def test_a_model_whose_own_call_is_not_its_traced_program_keeps_its_own_call_as_the_reference():
    """``traced_as_native`` is False (the model's second call computes another function), so the reference is the
    model's own call in fp64 and the probe refuses the conversion by name."""
    from mimarsinan.torch_mapping.value_parity.verdict import ConversionValueParityError

    for model in (_model(CountingCalls), _fp32(CountingCalls)):
        with pytest.raises(ConversionValueParityError):
            _convert(model, (8,), "spiking")


def test_the_witness_program_is_the_reference_only_where_it_is_the_models_call_bit_for_bit():
    from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
    from mimarsinan.torch_mapping.value_parity.witness import fp64_program, native_reference, traced_as_native

    torch.manual_seed(0)
    model = SoftmaxParamDtype().eval()
    program = fp64_program(trace_model(model, (8,)), (8,))
    inputs = [torch.randn(6, 8, dtype=torch.float64)]
    assert traced_as_native(model, program, inputs) and native_reference(model)(program, inputs) is program
    counting = _fp32(CountingCalls)
    counted = fp64_program(trace_model(counting, (8,)), (8,))
    assert not traced_as_native(counting, counted, inputs)
    assert native_reference(counting)(counted, inputs) is not counted
    fp64 = _model(SoftmaxParamDtype)  # an fp64 model: its own call is the reference, unchecked
    assert not traced_as_native(fp64, fp64_program(trace_model(fp64, (8,)), (8,)), inputs)


def test_the_reference_check_runs_on_the_first_seeded_input_alone_too():
    """``operator.is_(x.size(0), 1)`` is False in the trace and True natively at batch 1: the traced program is the
    model's call at batch 6 only, so the model's own call stays the reference and the batch-1 comparison refuses it
    (``TestBatchOne``)."""
    import operator

    from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
    from mimarsinan.torch_mapping.value_parity.witness import fp64_program, traced_as_native

    class IsBatchOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 6)

        def forward(self, x):
            y = self.fc(x)
            return y * operator.is_(x.size(0), 1) + y

    model = _fp32(IsBatchOne)
    program = fp64_program(trace_model(model, (8,)), (8,))
    x = torch.randn(6, 8, dtype=torch.float64)
    assert traced_as_native(model, program, [x]) and not traced_as_native(model, program, [x, x[:1]])
