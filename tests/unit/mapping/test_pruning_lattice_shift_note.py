"""[U-LIVE] The pruning lattice's note (U-RSHIFT's design review, m-5): the IR pruning lattice reads no input shift.

Constant folding (``mapping/pruning/graph/constant_folding.py``) and the host operations' liveness transfers
(``mapping/pruning/liveness_transfer/transfer_registry.py``) never read ``node_output_shifts``: a core row reading a
shifted host value whose raw value the lattice proved zero would be killed while the program delivers the shift. The
review found it unreachable, and these tests pin both reasons:

- every host operation with a zero-preserving transfer (identity, ReLU, LeakyReLU, GELU, dropout, the index moves and
  the average and max pools) maps a dead core's output (zero) to zero and a core's nonnegative output to a nonnegative
  value, so no signed (shifted) operation carries a zero from a dead core;
- nonzero constants fold only in the value domain, where no input shift is installed, never in a spiking mode.

A change that breaks either premise is red here and must revisit the lattice with the shift.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.spiking_semantics import ALL_SPIKING_MODES
from mimarsinan.mapping.pruning.liveness_transfer.constant_policy import domain_admits_nonzero_constants
from mimarsinan.mapping.pruning.liveness_transfer.transfer_region import REGION_REDUCE_HOST_OP_TYPES
from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import _ELEMENTWISE_HOST_OP_TYPES

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def _module(kind: type) -> nn.Module:
    if kind in (nn.AvgPool1d, nn.MaxPool1d, nn.AvgPool2d, nn.MaxPool2d):
        return kind(kernel_size=2, stride=1, padding=1) if "Max" not in kind.__name__ else kind(kernel_size=2)
    return kind()


def _input(kind: type, values: torch.Tensor) -> torch.Tensor:
    if kind in (nn.AvgPool2d, nn.MaxPool2d, nn.Dropout2d):
        return values.reshape(1, 2, 3, 4)
    if kind in (nn.AvgPool1d, nn.MaxPool1d):
        return values.reshape(1, 4, 6)
    return values.reshape(2, 12)


@pytest.mark.parametrize("kind", sorted(_ELEMENTWISE_HOST_OP_TYPES | REGION_REDUCE_HOST_OP_TYPES,
                                        key=lambda k: k.__name__))
def test_a_zero_preserving_transfer_carries_no_signed_value_from_a_core(kind):
    module = _module(kind).eval()
    zero = torch.zeros(24)
    with torch.no_grad():
        assert torch.count_nonzero(module(_input(kind, zero))) == 0
        nonnegative = torch.rand(24, generator=torch.Generator().manual_seed(7)) * 3.0
        assert bool((module(_input(kind, nonnegative)) >= 0).all())


def test_nonzero_constants_fold_in_the_value_domain_only():
    from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE

    assert not any(domain_admits_nonzero_constants(mode) for mode in ALL_SPIKING_MODES)
    assert domain_admits_nonzero_constants(INERT_SPIKING_MODE)


def test_the_lattice_and_the_transfers_read_no_input_shift():
    """The premise the two tests above make safe: the lattice reads no shift (a reader of one revisits this note)."""
    for rel in ("mapping/pruning/graph/constant_folding.py",
                "mapping/pruning/liveness_transfer/transfer_registry.py"):
        names = {n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", None) or getattr(n, "value", None)
                 for n in ast.walk(ast.parse((SRC / rel).read_text()))
                 if isinstance(n, (ast.Attribute, ast.Name, ast.Constant))}
        assert "node_output_shifts" not in names, rel
