"""A chunk is rewritten to ``narrow``s only along an axis whose extent is proved static (G1 review round 1).

The batch can sit on an axis other than 0: ``torch.stack([y, 2 * y], 0)`` puts it on axis 1, and so do ``movedim``,
``swapaxes`` and ``t``. A chunk there takes ``ceil(B / n)`` samples natively. Rewritten with the batch-1 trace's piece
size, it would read a constant one sample at every batch: exact for every B up to n (``ceil(B / 6) = 1``), so the
probe's batches 1 and 6 pass, and wrong from B = n + 1. Such a chunk stays as traced and refuses by name; the witnesses
are checked at batches 7 and 13, beyond the chunk count, under both contracts and at stage 2. A chunk along an axis the
batch domain proves static (the stacked pair's feature axis) is still rewritten and exact at those batches.

An impure program has no static extent (``symbolic/purity.py``): Wave 0's ``ImpureChunk`` witness widens the chunked
axis above batch 8 in a call outside the declared operator set, so its ``torch.chunk`` keeps its native host op and is
exact at every batch (rewritten with the batch-1 trace's widths, it was wrong from batch 9).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.value_parity import relative_error
from operand_completeness.harness import EXACT, PACKAGINGS, native, quietly

F = 8
BEYOND = (7, 13)
"""Batches beyond the chunk count 6: ``ceil(B / 6)`` is 2 and 3 there, 1 at every batch the probe samples."""


class _Pair(nn.Module):
    def __init__(self, read):
        super().__init__()
        self.a, self.b = nn.Linear(F, F), nn.Linear(F, 4)
        self.read = read

    def forward(self, x):
        y = self.a(x)
        return self.b(y + self.read(y))


def _stacked_method(y):
    return torch.stack([y, 2 * y], 0).chunk(6, dim=1)[0].mean(1).mean(0)  # (2, B, F): batch on axis 1


def _stacked_function(y):
    return torch.chunk(torch.stack([y, 2 * y], 0), 6, 1)[0].mean(1).mean(0)


def _moved(y):
    return y.movedim(0, 1).chunk(6, 1)[0].mean(1)  # (F, B): batch on axis 1


def _swapped(y):
    return y.swapaxes(0, 1).chunk(6, 1)[0].mean(1)


def _stacked_static_axis(y):
    halves = torch.stack([y, 2 * y], 0).chunk(2, dim=2)  # the feature axis F: static
    return torch.cat([(halves[1] - halves[0]).sum(0), halves[0].mean(0)], dim=1)


BATCH_AXIS_CHUNKS = {
    "stack_dim0_chunk_dim1_method": _stacked_method,
    "stack_dim0_chunk_dim1_function": _stacked_function,
    "movedim_chunk": _moved,
    "swapaxes_chunk": _swapped,
}

STAGES = [pytest.param("spiking", False, id="spiking"), pytest.param("mvm", False, id="mvm_stage_1"),
          pytest.param("mvm", True, id="mvm_stage_2")]


def _outcome(model: nn.Module, contract: str, stage_2: bool):
    try:
        return quietly(convert_torch_model, copy.deepcopy(model), (F,), 10, packaging=PACKAGINGS[contract],
                       value_program_stage=stage_2).eval()
    except ToolchainRefusal as refusal:
        return refusal


def _errors(flow: nn.Module, model: nn.Module, batches=BEYOND) -> dict:
    found = {}
    for batch in batches:
        x = torch.randn(batch, F, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            found[batch] = relative_error(flow(x), model(x)).relative_error
    return found


@pytest.mark.parametrize("key", sorted(BATCH_AXIS_CHUNKS))
@pytest.mark.parametrize("contract, stage_2", STAGES)
def test_a_chunk_along_a_batch_carrying_axis_refuses_by_name(key, contract, stage_2):
    model = native(lambda: _Pair(BATCH_AXIS_CHUNKS[key]))
    outcome = _outcome(model, contract, stage_2)
    if not isinstance(outcome, ToolchainRefusal):
        errors = _errors(outcome, model)
        pytest.fail(f"converted (certified {outcome.conversion_certificate.certified}); relative error at batches "
                    f"{BEYOND}: {errors}")


@pytest.mark.parametrize("contract", ["spiking", "mvm"])
def test_a_chunk_along_a_proved_static_axis_is_still_rewritten(contract):
    """Stage 1 only: the stacked pair is a host value without the batch on axis 0, which stage 2 refuses by name
    (``HostValueBatchLayoutError``) whatever reads it."""
    model = native(lambda: _Pair(_stacked_static_axis))
    flow = _outcome(model, contract, False)
    assert not isinstance(flow, ToolchainRefusal), flow
    assert flow.conversion_certificate.certified
    errors = _errors(flow, model, (1, 2, 3, *BEYOND))
    assert max(errors.values()) <= EXACT, errors
    narrows = [m for m in flow.get_mapper_repr().execution_order()
               if isinstance(m, ComputeOpMapper) and getattr(m.module, "fn", None) is torch.narrow]
    assert len(narrows) == 3  # one per getitem: halves[0] is read twice


@torch.fx.wrap
def wider_above_8(t):
    """Outside the declared operator set: (B, 6) up to batch 8, (B, 8) above (Wave 0 review round 9)."""
    return torch.cat([t, t[:, :2]], 1) if t.shape[0] > 8 else t


class ImpureChunk(nn.Module):
    def __init__(self, method: bool):
        super().__init__()
        self.fc, self.method = nn.Linear(F, 6), method

    def forward(self, x):
        z = wider_above_8(self.fc(x))
        a, b = z.chunk(2, 1) if self.method else torch.chunk(z, 2, 1)
        return a * b


@pytest.mark.parametrize("method", [False, True], ids=["function", "method"])
@pytest.mark.parametrize("contract", ["spiking", "mvm"])
def test_a_chunk_of_an_impure_program_is_never_rewritten_with_traced_widths(method, contract):
    model = native(lambda: ImpureChunk(method))
    flow = _outcome(model, contract, False)
    if isinstance(flow, ToolchainRefusal):
        assert method, f"the function form converts natively, as on the frozen Wave 0 tree: {flow!r}"
        return
    errors = _errors(flow, model, (1, 2, 3, 9, 13))
    assert max(errors.values()) <= EXACT, errors
