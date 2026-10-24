"""WI-5a: declared boundary grids (offset, rounding, code range) and the one entry-code rule every reader shares."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.activations.integer_law import OffGridInputError, grid_codes
from mimarsinan.models.nn.activations.value_quantizer import (
    ENTRY_ROUNDINGS,
    BoundaryGrid,
    BoundaryGridDeclarationError,
    DeclaredBoundaryGrid,
    ValueGridQuantizer,
    boundary_grid_of,
    quantize_to_value_grid,
)


def _values(n=4001, seed=0):
    return torch.linspace(-4.0, 4.0, n, dtype=torch.float64) + 1e-4 * torch.randn(
        n, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


class TestTheBaseGridKeepsItsRule:
    def test_snap_is_the_boundary_ssot_byte_for_byte(self):
        grid = BoundaryGrid(scale=2.5, bits=5)
        x = _values()
        assert torch.equal(grid.snap(x), quantize_to_value_grid(x, grid.scale, grid.bits))
        x32 = x.float()
        assert torch.equal(grid.snap(x32), quantize_to_value_grid(x32, grid.scale, grid.bits))

    def test_entry_codes_are_the_snap_codes_and_the_offset_is_zero(self):
        grid = BoundaryGrid(scale=2.5, bits=5)
        x = _values().numpy()
        codes = grid.entry_codes(x)
        assert codes.dtype == np.int64 and grid.entry_offset() == 0.0
        snapped = grid.snap(torch.as_tensor(x)).numpy()
        assert np.array_equal(codes * grid.step, snapped)

    def test_the_base_grid_pickles_and_reprs_as_before(self):
        grid = BoundaryGrid(scale=1.25, bits=8)
        assert set(vars(grid)) == {"scale", "bits"}
        assert repr(grid) == "BoundaryGrid(scale=1.25, bits=8)"
        assert pickle.loads(pickle.dumps(grid)) == grid


class TestDeclaredGrids:
    def test_the_rounding_vocabulary(self):
        assert ENTRY_ROUNDINGS == ("nearest", "floor")

    def test_an_offset_floor_grid_is_the_harness_input_encoder(self):
        # The harness's input lane: k = clamp(floor(15 (x + beta) / theta), 0, 15), value k theta / 15 - beta.
        theta, beta = 2.759164571762085, 1.7548962831497192
        grid = DeclaredBoundaryGrid(scale=theta, bits=5, offset=beta, rounding="floor", code_min=0, code_max=15)
        x = _values()
        k = torch.clamp(torch.floor((x + beta) / (theta / 15)), 0, 15)
        assert torch.equal(grid.snap(x), k * (theta / 15) - beta)
        assert np.array_equal(grid.entry_codes(x.numpy()), k.numpy().astype(np.int64))
        assert grid.code_range == (0, 15) and grid.levels == 15

    def test_nearest_rounds_half_away_on_the_shifted_quotient(self):
        grid = DeclaredBoundaryGrid(scale=1.5, bits=3, offset=0.25)  # step 0.5, codes -3..3
        x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 9.0, -9.0], dtype=torch.float64)
        # (x + 0.25) / 0.5 = -0.5, 0.5, 1.5, 2.5, 18.5, -17.5 -> -1, 1, 2, 3, 3, -3 (half away from zero)
        assert grid.entry_codes(x.numpy()).tolist() == [-1, 1, 2, 3, 3, -3]
        assert torch.equal(grid.snap(x), torch.tensor([-1, 1, 2, 3, 3, -3], dtype=torch.float64) * 0.5 - 0.25)

    def test_default_fields_equal_the_symmetric_grid(self):
        grid = DeclaredBoundaryGrid(scale=2.5, bits=5)
        x = _values()
        assert torch.equal(grid.snap(x), BoundaryGrid(scale=2.5, bits=5).snap(x))
        assert grid.code_range == (-15, 15)

    @pytest.mark.parametrize("kwargs,named", [
        (dict(rounding="half_even"), "rounding"),
        (dict(code_min=-16), "code_min"),
        (dict(code_max=16), "code_max"),
        (dict(code_min=3, code_max=3), "code_min"),
        (dict(offset=float("nan")), "offset"),
    ])
    def test_a_grid_the_dac_cannot_hold_refuses_by_name(self, kwargs, named):
        with pytest.raises(BoundaryGridDeclarationError, match=named) as info:
            DeclaredBoundaryGrid(scale=2.5, bits=5, **kwargs)
        assert isinstance(info.value, ToolchainRefusal)

    def test_grids_with_different_declarations_never_share_a_core(self):
        a = DeclaredBoundaryGrid(scale=2.5, bits=5, offset=0.5)
        assert a != DeclaredBoundaryGrid(scale=2.5, bits=5, offset=0.25)
        assert a != BoundaryGrid(scale=2.5, bits=5)
        assert a == DeclaredBoundaryGrid(scale=2.5, bits=5, offset=0.5)


class TestTheQuantizerCarriesTheDeclaration:
    def test_declare_installs_scale_and_grid(self):
        quantizer = ValueGridQuantizer(5)
        grid = DeclaredBoundaryGrid(scale=3.0, bits=5, offset=1.0, rounding="floor", code_min=0, code_max=15)
        quantizer.declare(grid)
        assert float(quantizer.scale) == 3.0 and quantizer.grid == grid
        assert boundary_grid_of(quantizer) == grid
        x = _values()
        assert torch.equal(quantizer(x), grid.snap(x))

    def test_a_calibration_after_a_declaration_drops_it(self):
        quantizer = ValueGridQuantizer(5)
        quantizer.declare(DeclaredBoundaryGrid(scale=3.0, bits=5, offset=1.0))
        quantizer.calibrate(2.0)
        assert quantizer.grid == BoundaryGrid(scale=2.0, bits=5)

    def test_a_grid_of_other_bits_refuses(self):
        with pytest.raises(BoundaryGridDeclarationError, match="bits"):
            ValueGridQuantizer(5).declare(DeclaredBoundaryGrid(scale=3.0, bits=8))

    def test_an_undeclared_quantizer_keeps_its_instance_state(self):
        quantizer = ValueGridQuantizer(5, 2.0)
        assert "_declared" not in vars(quantizer)


class TestEntryCodesWithAnOffset:
    def test_the_law_reads_shifted_codes_within_the_dtype_rounding(self):
        grid = DeclaredBoundaryGrid(scale=2.759164571762085, bits=5, offset=1.7548962831497192,
                                    rounding="floor", code_min=0, code_max=15)
        x = _values()
        step = torch.tensor(grid.step, dtype=torch.float64)
        offset = torch.tensor(grid.offset, dtype=torch.float64)
        codes = grid_codes(grid.snap(x), step, where="t", offset=offset)
        assert torch.equal(codes, torch.as_tensor(grid.entry_codes(x.numpy()), dtype=torch.float64))
        x32 = x.float()
        own = torch.clamp(torch.floor((x32 + torch.tensor(grid.offset, dtype=torch.float32))
                                      / torch.tensor(grid.step, dtype=torch.float32)), 0, 15)
        assert torch.equal(grid_codes(grid.snap(x32), step, where="t", offset=offset), own.double())

    def test_without_the_offset_an_offset_value_is_off_grid(self):
        grid = DeclaredBoundaryGrid(scale=3.0, bits=5, offset=0.07, rounding="floor", code_min=0, code_max=15)
        snapped = grid.snap(_values())
        with pytest.raises(OffGridInputError):
            grid_codes(snapped, torch.tensor(grid.step, dtype=torch.float64), where="t")

    def test_no_offset_is_the_base_rule(self):
        x = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
        step = torch.tensor(0.5, dtype=torch.float64)
        assert torch.equal(grid_codes(x, step, where="t"), grid_codes(x, step, where="t", offset=None))


class TestALaneChipReadsUnsignedGrids:
    """WI-5e: on a lane chip's DAC an unsigned code range is the unsigned lane grid of its bits (2^b codes,
    ``step = scale / (2^b - 1)``), so a declared entry grid and a lane's readout range mean one thing."""

    def test_an_unsigned_entry_grid_is_the_lane_grid_of_its_bits(self):
        from mimarsinan.models.nn.activations.lane_quantization import quantize_to_lane_grid, unsigned_levels

        theta, beta = 2.759164571762085, 1.7548962831497192
        grid = DeclaredBoundaryGrid(scale=theta, bits=4, offset=beta, rounding="floor", code_min=0, code_max=15,
                                    unsigned_dac=True)
        assert grid.step == theta / unsigned_levels(4) == theta / 15
        x = _values()
        k = torch.clamp(torch.floor((x + beta) / (theta / 15)), 0, 15)  # the harness's input lane at 4 bits
        assert torch.equal(grid.snap(x), k * (theta / 15) - beta)
        assert np.array_equal(grid.entry_codes(x.numpy()), k.numpy().astype(np.int64))
        nearest = DeclaredBoundaryGrid(scale=theta, bits=4, offset=beta, code_min=0, code_max=15, unsigned_dac=True)
        assert torch.equal(nearest.snap(x), quantize_to_lane_grid(x, theta, 4, beta))

    def test_a_signed_range_on_a_lane_dac_keeps_the_symmetric_step(self):
        grid = DeclaredBoundaryGrid(scale=2.1, bits=4, offset=0.5, code_min=-7, code_max=7, unsigned_dac=True)
        assert grid.step == 2.1 / 7

    def test_without_a_lane_dac_the_step_is_the_symmetric_grid_s(self):
        grid = DeclaredBoundaryGrid(scale=2.1, bits=4, offset=0.5, code_min=0, code_max=7)
        assert grid.step == 2.1 / 7
