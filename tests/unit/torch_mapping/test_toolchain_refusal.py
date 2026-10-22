"""One base type for every named refusal: a refusal by name is told apart from a loud crash by its type alone.

Every refusal the conversion edge raises inherits ``ToolchainRefusal`` and keeps its
builtin base, so existing ``except`` clauses keep working. The probes never wrap an
anonymous crash in a toolchain type: it reaches the caller as itself (loud), and a
census reads ``isinstance(exc, ToolchainRefusal)`` to classify it.
"""

from __future__ import annotations

import contextlib
import copy
import io

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal, is_toolchain_refusal, refusal_in_chain
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.support.shape_probe import HostValueBatchLayoutError, NonTensorHostValueError
from mimarsinan.pipelining.cache.load_store_strategies import CorruptCacheEntryError, UncertifiedCachedFlowError
from mimarsinan.torch_mapping.conversion_probe import ConversionProbeError, probe_forward
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.conv_mixin import UnsupportedConvPaddingError
from mimarsinan.torch_mapping.converter_handlers.refusals import (
    BatchAxisLayoutError,
    ConcatenationOperandError,
    GenericOpArgumentError,
    ModuleCallArgumentError,
    TupleOutputIndexError,
)
from mimarsinan.torch_mapping.encoding_layers import UnresolvedEncodingPlacementError
from mimarsinan.torch_mapping.mapper_graph_converter import UnsupportedCallMethodError
from mimarsinan.torch_mapping.mapper_graph_fx import GenericOpBatchAxisError
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityError
from mimarsinan.torch_mapping.torch_graph_tracer import TracingError
from mimarsinan.torch_mapping.value_parity import ConversionValueParityError, WitnessConversion, probe_conversion_values
from mimarsinan.torch_mapping.value_parity import probe as probe_module
from mimarsinan.torch_mapping.value_parity import value_program

# Every named refusal of the conversion edge, with the builtin base it kept.
REFUSALS = [
    (RepresentabilityError, Exception),
    (UnsupportedConvPaddingError, NotImplementedError),
    (GenericOpArgumentError, RuntimeError),
    (GenericOpBatchAxisError, RuntimeError),
    (ModuleCallArgumentError, RuntimeError),
    (TupleOutputIndexError, RuntimeError),
    (ConcatenationOperandError, RuntimeError),
    (BatchAxisLayoutError, RuntimeError),
    (UnsupportedCallMethodError, RuntimeError),
    (ConversionProbeError, RuntimeError),
    (ConversionValueParityError, RuntimeError),
    (UncertifiedCachedFlowError, Exception),
    (TracingError, Exception),
    (UnresolvedEncodingPlacementError, ValueError),
    (NonTensorHostValueError, TypeError),
    (HostValueBatchLayoutError, ValueError),
]


@pytest.mark.parametrize("refusal,builtin", REFUSALS, ids=[r.__name__ for r, _ in REFUSALS])
def test_every_named_refusal_inherits_the_base_and_keeps_its_builtin_base(refusal, builtin):
    assert issubclass(refusal, ToolchainRefusal)
    assert issubclass(refusal, builtin)


def test_the_value_parity_refusal_is_a_conversion_probe_error():
    assert issubclass(ConversionValueParityError, ConversionProbeError)


class TestPredicate:
    def test_reads_the_type_never_the_module_name(self):
        class ForeignRefusal(ToolchainRefusal):
            pass

        anonymous = type("Anonymous", (RuntimeError,), {"__module__": "mimarsinan.torch_mapping.fake"})
        assert ForeignRefusal.__module__ == __name__
        assert is_toolchain_refusal(ForeignRefusal("named"))
        assert not is_toolchain_refusal(anonymous("crash inside a mimarsinan module"))

    def test_a_toolchain_error_that_is_no_refusal_is_not_one(self):
        corrupt = CorruptCacheEntryError("model", "/run", "torch_model", EOFError("torn"))
        assert not is_toolchain_refusal(corrupt)
        assert not is_toolchain_refusal(AttributeError("anonymous"))

    def test_the_chain_carries_a_refusal_raised_inside_a_forward(self):
        refusal = GenericOpArgumentError("mul", "mul", "size", "call_method 'size'")
        try:
            try:
                raise refusal
            except GenericOpArgumentError as exc:
                raise RuntimeError("[ModelRepresentation] forward failed at node X") from exc
        except RuntimeError as wrapped:
            assert refusal_in_chain(wrapped) is refusal
        assert refusal_in_chain(ValueError("plain")) is None


# ----------------------------------------------------------------------------- the probes keep crashes anonymous
class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.r = nn.ReLU()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(self.r(self.l1(x)))


def _model() -> nn.Module:
    torch.manual_seed(0)
    return _MLP().double().eval()


def _quiet(call, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def _crashing_lowering(_flow):
    raise AttributeError("'NoneType' object has no attribute 'core_matrix' (deep inside IR lowering)")


class TestAnonymousCrashStaysAnonymous:
    def test_an_attribute_error_during_ir_lowering_is_not_a_refusal(self, monkeypatch):
        monkeypatch.setattr(value_program, "lower_identity_program", _crashing_lowering)
        with pytest.raises(AttributeError, match="core_matrix") as excinfo:
            _quiet(convert_torch_model, _model(), (12,), 6, packaging=MVM_PACKAGING, value_program_stage=True)
        assert type(excinfo.value) is AttributeError
        assert not is_toolchain_refusal(excinfo.value)

    def test_a_permissive_conversion_records_the_crash_uncertified(self, monkeypatch):
        monkeypatch.setattr(value_program, "lower_identity_program", _crashing_lowering)
        with pytest.warns(UserWarning, match="NOT certified"):
            flow = _quiet(convert_torch_model, _model(), (12,), 6, packaging=MVM_PACKAGING,
                          value_program_stage=True, strict=False)
        (_, lowering) = flow.conversion_certificate.stages
        assert not lowering.ok and "AttributeError" in (lowering.reason or "")

    def test_an_anonymous_forward_crash_of_the_converted_model_propagates_as_itself(self):
        class _Crashes(nn.Module):
            def forward(self, x):
                raise KeyError("missing value")

        with pytest.raises(KeyError, match="missing value"):
            probe_conversion_values(_model(), lambda m: WitnessConversion(_Crashes(), {}), (12,))

    def test_a_refusal_raised_inside_the_forward_stays_named(self):
        class _Refuses(nn.Module):
            def forward(self, x):
                raise GenericOpArgumentError("mul", "mul", "p", "get_attr 'p'")

        with pytest.raises(ConversionValueParityError, match="GenericOpArgumentError") as excinfo:
            probe_conversion_values(_model(), lambda m: WitnessConversion(_Refuses(), {}), (12,))
        assert is_toolchain_refusal(excinfo.value)


class _AlwaysRaisingFlow(nn.Module):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    def forward(self, x):
        try:
            raise self.error
        except Exception as exc:
            raise RuntimeError(
                "[ModelRepresentation] forward failed at node ComputeOpMapper(name='bad')"
            ) from exc


class TestStructuralProbe:
    def test_an_anonymous_crash_propagates_as_itself(self):
        with pytest.raises(RuntimeError, match="name='bad'") as excinfo:
            probe_forward(_AlwaysRaisingFlow(ZeroDivisionError("x")), (4,), "cpu", strict=True)
        assert type(excinfo.value) is RuntimeError
        assert not is_toolchain_refusal(excinfo.value)

    def test_a_refusal_raised_inside_the_forward_propagates_by_name(self):
        refusal = GenericOpArgumentError("mul", "mul", "size", "call_method 'size'")
        with pytest.raises(GenericOpArgumentError) as excinfo:
            probe_forward(_AlwaysRaisingFlow(refusal), (4,), "cpu", strict=True)
        assert excinfo.value is refusal

    def test_the_corpus_oracle_reads_the_base(self):
        assert probe_module.is_toolchain_refusal is is_toolchain_refusal


@pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
def test_a_refusal_raised_while_converting_reaches_the_caller_by_name(packaging):
    class _BatchCat(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(12, 6)

        def forward(self, x):
            y = self.fc(x)
            return torch.cat([y, y])

    torch.manual_seed(0)
    with pytest.raises(ConcatenationOperandError) as excinfo:
        _quiet(convert_torch_model, copy.deepcopy(_BatchCat().double().eval()), (12,), 6, packaging=packaging)
    assert is_toolchain_refusal(excinfo.value)
