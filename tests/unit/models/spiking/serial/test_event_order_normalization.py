"""§7 row 8: a sweep-interleaved event stream is NORMALIZED or refused.

Adjacency is count-changing, so the wire may never carry a raw arrival order.
Producers fold a stream into per-slot counts through the ``event_order`` SSOT
and consumers drain ascending with each slot's multiplicity adjacent; the
kernel's input is that count vector, and anything that is not one is refused.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.platform.event_order import (
    canonical_slot_order,
    drain_events,
    normalize_event_counts,
)
from mimarsinan.models.spiking.serial import (
    SerialFoldUnsupportedError,
    lif_serial_fold,
    require_event_counts,
)

_LAW = SomaLaw(
    firing_mode="Novena", thresholding_mode="<=",
    firing_granularity="per_event", membrane_arithmetic="unbounded",
    membrane_bits=0,
)
# The §2.3 witness: theta=5, w=[+3,-3]; adjacent fires once, interleaved zero.
_WEIGHT = torch.tensor([[3.0, -3.0]], dtype=torch.float64)
_THETA = torch.tensor(5.0, dtype=torch.float64)
_SWEEP_INTERLEAVED = [(0, 1), (1, 1), (0, 1)]


def _fold(counts):
    return lif_serial_fold(
        torch.zeros(1, 1, dtype=torch.float64), _WEIGHT,
        torch.tensor([counts], dtype=torch.float64), _THETA, soma_law=_LAW,
    ).tolist()


def test_the_interleaved_stream_normalizes_to_canonical_adjacency():
    counts = normalize_event_counts(_SWEEP_INTERLEAVED, 2)
    assert counts == [2, 1]
    assert drain_events(counts) == [(0, 2), (1, 1)]
    assert _fold(counts) == [[1.0]]


def test_folding_the_raw_arrival_order_would_be_a_different_computation():
    """Non-vacuity: normalization is not cosmetic. Expanding the interleaved
    stream slot-by-slot as it arrived gives ZERO spikes, not one."""
    arrival_weight = torch.tensor([[3.0, -3.0, 3.0]], dtype=torch.float64)
    arrival = lif_serial_fold(
        torch.zeros(1, 1, dtype=torch.float64), arrival_weight,
        torch.ones(1, 3, dtype=torch.float64), _THETA, soma_law=_LAW,
    )
    assert arrival.tolist() == [[0.0]]
    assert _fold(normalize_event_counts(_SWEEP_INTERLEAVED, 2)) == [[1.0]]


def test_the_kernel_input_is_a_count_vector_and_anything_else_refuses():
    require_event_counts(torch.tensor([[2.0, 1.0]], dtype=torch.float64))
    with pytest.raises(SerialFoldUnsupportedError, match="RATE"):
        require_event_counts(torch.tensor([[0.5]], dtype=torch.float64))
    with pytest.raises(SerialFoldUnsupportedError, match="NON-NEGATIVE"):
        require_event_counts(torch.tensor([[-1.0]], dtype=torch.float64))


def test_the_normalizer_refuses_a_stream_outside_the_slot_order():
    with pytest.raises(ValueError, match="outside"):
        normalize_event_counts([(5, 1)], 2)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_event_counts([(0, -1)], 2)


def test_the_fold_walks_exactly_the_canonical_slot_order():
    """The kernel and the SSOT agree on WHICH order 'ascending' means."""
    assert list(canonical_slot_order(int(_WEIGHT.shape[-1]))) == [0, 1]
    reversed_weight = _WEIGHT.flip(-1)
    reversed_counts = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    flipped = lif_serial_fold(
        torch.zeros(1, 1, dtype=torch.float64), reversed_weight,
        reversed_counts, _THETA, soma_law=_LAW,
    )
    assert flipped.tolist() == [[0.0]]
