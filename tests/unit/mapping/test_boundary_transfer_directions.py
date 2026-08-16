"""[E3] A pass boundary is crossed TWICE, and the two crossings differ.

The census charged one bucket: ``carried_bytes``, sized under the run's
transfer discipline. That number is the BUFFER requirement — what the host
holds between the producing and the consuming pass — and it was also being
used as the transfer multiplicand, which is a different question with a
different answer.

What actually crosses, read off the runner rather than assumed:

* chip -> host: the boundary window's EMISSIONS. Both disciplines read the
  same spike trace off the chip; the COLLAPSE reduction to window counts is
  done host-side afterwards (``_compute_seg_output_spike_count`` sums the
  trace). So the discipline changes what is BUFFERED, never what came off
  the chip.
* host -> chip: the re-emitted train. The chip's input interface takes spike
  trains — the runner always injects an encoded raster
  (``set_input_spike_trains``) — so this direction is dense under both
  disciplines too.

Under COLLAPSE that makes the transfer strictly larger than the buffer, by
exactly the ratio the two representations differ by (T bits vs log2(T+1)).
Owner decision: the directions are charged separately, with their own
constants and multipliers.
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.support.schedule.pass_carry import (
    boundary_transfer_bytes,
    carried_wire_bytes,
    pass_carry_census,
)
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM


class _Slice:
    def __init__(self, node_id: int, size: int):
        self.node_id = node_id
        self.size = size


class _Stage:
    """One neural pass of a segment, duck-typed as the census reads it."""

    kind = "neural"

    def __init__(self, segment: int, outputs=(), inputs=()):
        self.schedule_segment_index = segment
        self.output_map = [_Slice(n, w) for n, w in outputs]
        self.input_map = [_Slice(n, w) for n, w in inputs]


def _two_pass_program(width: int = 8):
    """Pass 0 produces one wire; pass 1 of the SAME segment consumes it."""
    return [_Stage(0, outputs=[(1, width)]), _Stage(0, inputs=[(1, width)])]


class TestOneWireCrossingOnce:
    @pytest.mark.parametrize("transfer", [VERBATIM, COLLAPSE])
    def test_a_crossing_is_dense_under_either_discipline(self, transfer):
        """The chip emits per cycle and consumes per cycle; the discipline is a
        HOST-side choice about what to keep in between."""
        del transfer
        assert boundary_transfer_bytes(width=8, timesteps=16) == 16

    def test_collapse_buffers_less_than_it_transfers(self):
        """The saving COLLAPSE buys is in the buffer, not on the wire — the
        distinction the single-bucket census could not express."""
        buffered = carried_wire_bytes(8, 16, COLLAPSE)
        assert buffered < boundary_transfer_bytes(width=8, timesteps=16)

    def test_verbatim_buffers_exactly_what_it_transfers(self):
        assert carried_wire_bytes(8, 16, VERBATIM) == boundary_transfer_bytes(
            width=8, timesteps=16,
        )


class TestTheCensusChargesBothDirections:
    @pytest.mark.parametrize("transfer", [VERBATIM, COLLAPSE])
    def test_each_carried_wire_leaves_and_re_enters(self, transfer):
        census = pass_carry_census(_two_pass_program(), timesteps=16,
                                   transfer=transfer)
        one_way = boundary_transfer_bytes(width=8, timesteps=16)
        assert census["boundary_out_bytes"] == one_way
        assert census["boundary_in_bytes"] == one_way

    def test_the_buffer_figures_keep_their_meaning(self):
        """``carried_bytes``/``peak_live_bytes`` size the BUFFER (the declared
        capacity gate reads them); splitting the transfer must not move them."""
        collapsed = pass_carry_census(_two_pass_program(), 16, COLLAPSE)
        verbatim = pass_carry_census(_two_pass_program(), 16, VERBATIM)
        assert collapsed["carried_bytes"] == carried_wire_bytes(8, 16, COLLAPSE)
        assert verbatim["carried_bytes"] == carried_wire_bytes(8, 16, VERBATIM)
        assert collapsed["peak_live_bytes"] < verbatim["peak_live_bytes"]

    def test_a_program_that_carries_nothing_transfers_nothing(self):
        census = pass_carry_census([_Stage(0, outputs=[(1, 8)])], 16, VERBATIM)
        assert census["boundary_out_bytes"] == 0
        assert census["boundary_in_bytes"] == 0
