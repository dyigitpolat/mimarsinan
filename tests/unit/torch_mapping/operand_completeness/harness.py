"""Exactness harness: the converted model and the MVM identity value program against the native fp64 model."""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Callable, Iterable

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import PermuteMapper, ReshapeMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model, trace_and_convert
from mimarsinan.torch_mapping.value_parity import relative_error
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program

EXACT = 1e-12
PACKAGINGS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}
CONTRACTS = [pytest.param(name, id=name) for name in sorted(PACKAGINGS)]
BATCHES = (1, 2, 3, 5)
INPUT_KINDS = ("normal", "wide", "uniform", "negative", "zero")


def native(factory: Callable[[], nn.Module], seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return factory().double().eval()


def quietly(call, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def convert(model: nn.Module, shape, contract: str) -> nn.Module:
    """The probed conversion (``convert_torch_model``) of a copy of ``model``."""
    return quietly(convert_torch_model, copy.deepcopy(model), tuple(shape), 10,
                   packaging=PACKAGINGS[contract]).eval()


def convert_unprobed(model: nn.Module, shape, contract: str) -> nn.Module:
    return quietly(trace_and_convert, copy.deepcopy(model), tuple(shape), "cpu",
                   packaging=PACKAGINGS[contract]).flow.eval()


def inputs(shape, batch: int, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + 17 * batch + seed)
    full = (batch, *tuple(shape))
    if kind == "normal":
        return torch.randn(full, generator=g, dtype=torch.float64)
    if kind == "wide":
        return 5.0 * torch.randn(full, generator=g, dtype=torch.float64)
    if kind == "uniform":
        return torch.rand(full, generator=g, dtype=torch.float64)
    if kind == "negative":
        return -torch.rand(full, generator=g, dtype=torch.float64) - 0.1
    if kind == "zero":
        return torch.zeros(full, dtype=torch.float64)
    raise ValueError(kind)


def worst_error(candidate: nn.Module, model: nn.Module, shape, *, batches: Iterable[int] = BATCHES,
                kinds: Iterable[str] = INPUT_KINDS, flat_candidate: bool = False) -> float:
    worst = 0.0
    for batch in batches:
        for kind in kinds:
            x = inputs(shape, batch, kind)
            with torch.no_grad():
                want = model(x)
                want = want[0] if isinstance(want, (tuple, list)) else want
                got = candidate(x)
            if flat_candidate:
                got = got.reshape(want.shape)
            worst = max(worst, relative_error(got, want).relative_error)
    return worst


def assert_converted_exact(model: nn.Module, shape, contract: str, **kwargs) -> nn.Module:
    flow = convert(model, shape, contract)
    err = worst_error(flow, model, shape, **kwargs)
    assert err <= EXACT, f"{contract}: converted relative error {err:.3e} > {EXACT:g}"
    return flow


def assert_value_program_exact(model: nn.Module, shape, *, batches=(2, 3), **kwargs) -> None:
    flow = convert(model, shape, "mvm")
    program = identity_value_program(flow)
    err = worst_error(program, model, shape, batches=batches, flat_candidate=True, **kwargs)
    assert err <= EXACT, f"value program relative error {err:.3e} > {EXACT:g}"


def assert_refused_by_name(call, refusal_types, *names: str) -> BaseException:
    """``call`` raises one of the declared refusal types, and its message names every one of ``names``."""
    with pytest.raises(refusal_types) as excinfo:
        call()
    message = str(excinfo.value)
    for name in names:
        assert name in message, f"refusal does not name {name!r}: {message}"
    return excinfo.value


IMPURE_BATCHES = (1, 2, 3, 6, 7, 9, 13)


def assert_impure_exact(model: nn.Module, shape, contract: str, node: str, operator: str, *,
                        batches: Iterable[int] = IMPURE_BATCHES, kinds: Iterable[str] = ("normal", "uniform")):
    """Review round 9: an impure program (``symbolic/purity.py``) converts exactly at every batch with no static shape
    proof: no literal, no per-sample layout mapper (every layout is the host op of its native method), and the flow and
    the certificate name its first impure node (``node``, ``operator``). Returns the flow."""
    flow = assert_converted_exact(model, shape, contract, batches=batches, kinds=kinds)
    impurity = flow.shape_impurity
    assert impurity is not None and (impurity.node, impurity.operator) == (node, operator), impurity
    assert flow.shape_literals == {} and flow.shape_proof == impurity.statement()
    assert flow.conversion_certificate.shape_proof == impurity.statement()
    layouts = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, (ReshapeMapper, PermuteMapper))]
    assert not layouts, layouts
    return flow


def assert_impure_stage_2_refused(model: nn.Module, shape, node: str, operator: str) -> str:
    """Stage 2 of an impure program is refused by name (``ImpureShapeProgramError``), naming its first impure node."""
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, copy.deepcopy(model), tuple(shape), 10, packaging=MVM_PACKAGING,
                value_program_stage=True)
    message = str(excinfo.value)
    assert "ImpureShapeProgramError" in message and f"{operator} (FX node {node!r}): " in message, message
    return message
