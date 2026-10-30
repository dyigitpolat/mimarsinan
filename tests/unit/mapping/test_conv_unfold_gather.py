"""The NF event-serial twin's conv slot table (``mappers/conv_unfold.py``), and its seam with the IR lowering.

[U-H port] On the lineage this file was ``test_conv_patch_gather.py`` and the unfold was the IR mapping's gather
too. On the release head the IR lowering is ``conv_helpers.map_shared_weight_conv`` over
``conv_helpers.gather_conv_patches`` (the D9 gather, pinned by the release's own ``test_conv_patch_gather.py``), so
the twin's table is pinned EQUAL to that gather cell for cell (``TestTheTwinTableIsTheIRGather``): a shared weight
bank's columns and a core's axon slots stay the same thing, with one lowering of record.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mimarsinan.mapping.mappers.conv_helpers import conv_output_size, gather_conv_patches
from mimarsinan.mapping.mappers.conv_unfold import conv_patch_gather


def _gather2d(**kwargs):
    return conv_patch_gather(**kwargs)


class TestTheTwinTableIsTheIRGather:
    @pytest.mark.parametrize("c_in,grid,kernel,stride,padding,dilation", [
        (3, (7, 5), (3, 2), (2, 1), (1, 0), (1, 2)),
        (2, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, (6, 9), (2, 3), (2, 2), (0, 2), (2, 1)),
        (4, (5, 5), (1, 1), (1, 1), (0, 0), (1, 1)),
    ])
    def test_conv2d_padded_tables_are_equal_cell_for_cell(self, c_in, grid, kernel, stride, padding, dilation):
        gather = _gather2d(in_channels=c_in, source_grid=grid, kernel=kernel, stride=stride,
                           padding=padding, dilation=dilation)
        padded = np.arange(c_in * (grid[0] + 2 * padding[0]) * (grid[1] + 2 * padding[1])).reshape(
            c_in, grid[0] + 2 * padding[0], grid[1] + 2 * padding[1])
        out = conv_output_size(grid, kernel, stride, padding, dilation)
        assert tuple(gather.out_grid) == tuple(out)
        assert np.array_equal(gather.gather(padded), gather_conv_patches(padded, kernel, stride, dilation, out))

    @pytest.mark.parametrize("c_in,length,kernel,stride,padding,dilation", [
        (2, 4, 2, 1, 0, 1), (3, 9, 3, 2, 1, 2), (1, 7, 1, 1, 0, 1), (4, 12, 5, 3, 2, 1),
    ])
    def test_conv1d_padded_tables_are_equal_cell_for_cell(self, c_in, length, kernel, stride, padding, dilation):
        gather = conv_patch_gather(in_channels=c_in, source_grid=(length,), kernel=kernel, stride=stride,
                                   padding=padding, dilation=dilation)
        padded = np.arange(c_in * (length + 2 * padding)).reshape(c_in, length + 2 * padding)
        out = conv_output_size((length,), (kernel,), (stride,), (padding,), (dilation,))
        assert np.array_equal(gather.gather(padded),
                              gather_conv_patches(padded, (kernel,), (stride,), (dilation,), out))


class TestThePaddedTable:
    def test_it_reproduces_the_historical_conv2d_index_arrays(self):
        """The shipped bundles were derived with these exact broadcast arrays;
        the SSOT must produce them byte-for-byte or every conv bundle moves."""
        c_in, h_in, w_in = 3, 7, 5
        k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w = 3, 2, 2, 1, 1, 0, 1, 2
        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
        h_base = np.arange(h_out) * s_h
        w_base = np.arange(w_out) * s_w
        kh_off = np.arange(k_h) * d_h
        kw_off = np.arange(k_w) * d_w
        h_idx = h_base[:, None, None, None, None] + kh_off[None, None, None, :, None]
        w_idx = w_base[None, :, None, None, None] + kw_off[None, None, None, None, :]
        c_idx = np.arange(c_in)[None, None, :, None, None]
        h_b, w_b, c_b = np.broadcast_arrays(h_idx, w_idx, c_idx)

        gather = _gather2d(
            in_channels=c_in, source_grid=(h_in, w_in), kernel=(k_h, k_w),
            stride=(s_h, s_w), padding=(p_h, p_w), dilation=(d_h, d_w),
        )

        assert gather.out_grid == (h_out, w_out)
        assert np.array_equal(gather.patch_index[0], c_b)
        assert np.array_equal(gather.patch_index[1], h_b)
        assert np.array_equal(gather.patch_index[2], w_b)

    def test_gather_flattens_position_major_over_the_padded_grid(self):
        gather = _gather2d(
            in_channels=2, source_grid=(4, 4), kernel=3, stride=1, padding=1,
            dilation=1,
        )
        padded = np.arange(2 * 6 * 6).reshape(2, 6, 6)
        table = gather.gather(padded)
        assert table.shape == (gather.n_positions, gather.patch_size) == (16, 18)
        # Position (1, 2) of the padded grid, channel-major then taps.
        expected = [
            padded[c, 1 + kh, 2 + kw]
            for c in range(2) for kh in range(3) for kw in range(3)
        ]
        assert list(table[1 * 4 + 2]) == expected


class TestTheUnpaddedTableTheTwinGathersWith:
    @pytest.mark.parametrize("padding", [0, 1, 2])
    def test_it_selects_exactly_what_torch_unfold_selects(self, padding):
        """The twin's table must pick the same input cells ``F.unfold`` does —
        that is the deployment's receptive field, stated independently."""
        c_in, h_in, w_in, kernel, stride, dilation = 3, 6, 5, 2, 2, 2
        gather = _gather2d(
            in_channels=c_in, source_grid=(h_in, w_in), kernel=kernel,
            stride=stride, padding=padding, dilation=dilation,
        )
        x = torch.arange(1, c_in * h_in * w_in + 1, dtype=torch.float64).reshape(
            1, c_in, h_in, w_in
        )
        unfolded = F.unfold(
            x, kernel_size=kernel, stride=stride, padding=padding,
            dilation=dilation,
        )[0].T  # (positions, slots)

        flat = torch.cat([x.reshape(1, -1), x.new_zeros(1, 1)], dim=1)
        index = torch.as_tensor(gather.slot_source_index, dtype=torch.long)
        table = flat[:, index][0]

        assert table.shape == unfolded.shape
        assert torch.equal(table, unfolded)

    def test_padding_slots_are_marked_off_and_read_as_zero_events(self):
        gather = _gather2d(
            in_channels=1, source_grid=(3, 3), kernel=3, stride=1, padding=1,
            dilation=1,
        )
        index = gather.slot_source_index
        assert index.shape == (9, 9)
        # The corner position sees five padded taps out of nine.
        assert int((index[0] < 0).sum()) == 5
        assert int((index[4] < 0).sum()) == 0, "the centre position is fully inside"

    def test_the_two_tables_describe_the_same_cells(self):
        """The padded table and the unpadded table are ONE unfold: strip the
        padding offset from the first and the second must fall out."""
        gather = _gather2d(
            in_channels=2, source_grid=(5, 4), kernel=3, stride=2, padding=1,
            dilation=1,
        )
        c_b, h_b, w_b = gather.patch_index
        h_u = h_b - gather.padding[0]
        w_u = w_b - gather.padding[1]
        inside = (
            (h_u >= 0) & (h_u < 5) & (w_u >= 0) & (w_u < 4)
        )
        flat = (c_b * 5 + np.clip(h_u, 0, 4)) * 4 + np.clip(w_u, 0, 3)
        expected = np.where(inside, flat, -1).reshape(
            gather.n_positions, gather.patch_size
        )
        assert np.array_equal(gather.slot_source_index, expected)


class TestTheOneDimensionalUnfold:
    def test_the_slot_order_is_channel_major_then_taps(self):
        """The historical conv1d flatten broadcast to ``(C, positions, taps)``
        and reshaped that to ``(positions, C*taps)`` — a scramble whenever
        ``positions != C``. The SSOT is position-major by construction."""
        c_in, l_in, k, s, d = 2, 4, 2, 1, 1
        gather = conv_patch_gather(
            in_channels=c_in, source_grid=(l_in,), kernel=k, stride=s,
            padding=0, dilation=d,
        )
        source = np.arange(c_in * l_in).reshape(c_in, l_in)
        table = gather.gather(source)
        assert table.shape == (3, 4)
        for pos in range(3):
            assert list(table[pos]) == [
                source[c, pos * s + t * d] for c in range(c_in) for t in range(k)
            ]

    def test_it_selects_exactly_what_torch_unfold_selects(self):
        c_in, l_in, k, s, p, d = 3, 9, 3, 2, 1, 2
        gather = conv_patch_gather(
            in_channels=c_in, source_grid=(l_in,), kernel=k, stride=s,
            padding=p, dilation=d,
        )
        x = torch.arange(1, c_in * l_in + 1, dtype=torch.float64).reshape(1, c_in, l_in)
        unfolded = F.unfold(
            x.unsqueeze(-1), kernel_size=(k, 1), stride=(s, 1), padding=(p, 0),
            dilation=(d, 1),
        )[0].T
        flat = torch.cat([x.reshape(1, -1), x.new_zeros(1, 1)], dim=1)
        index = torch.as_tensor(gather.slot_source_index, dtype=torch.long)
        assert torch.equal(flat[:, index][0], unfolded)


def test_a_geometry_with_no_output_positions_refuses():
    with pytest.raises(ValueError, match="no output positions"):
        conv_patch_gather(
            in_channels=1, source_grid=(3, 3), kernel=5, stride=1, padding=0,
            dilation=1,
        )
