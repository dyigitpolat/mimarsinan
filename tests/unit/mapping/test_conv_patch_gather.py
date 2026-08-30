"""ONE source of unfold truth: the conv mappers' patch gather.

The IR mapping gathers ``IRSource`` patches with it and the NF event-serial
twin gathers upstream event multiplicities with it. Both tables therefore have
to be the SAME table, and both have to agree with the order the perceptron's
own weight columns are in (channel-major, then kernel taps) — that agreement is
what makes a shared weight bank's columns and a core's axon slots the same
thing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mimarsinan.mapping.mappers.conv_unfold import conv_patch_gather


def _gather2d(**kwargs):
    return conv_patch_gather(**kwargs)


class TestThePaddedTableTheIRMappingGathersWith:
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
