"""[U-PFOLD r2, the merge rule with U-BRANGE2's stage 1e] The row drive's reader takes the positions form (m-3).

The drive site's rule (``drive_rule.least_error_drive_and_rows``) sizes a perceptron's always-on row from its effective
weight and bias. Over a fold per position the one reader in its per-input form refuses by name, so the drive site reads
the reader in both forms (``PerceptronTransformer(positions=True)``): every weight row and every position's bias, the
positions flattened into rows, which is what the program's cores carry; a perceptron of one fold per input reads as
before, byte for byte.

No converter-built form with a fold per position reaches this reader with the projection's weight bits: on the spiking
wire the stamp finds no armed reader grid, and on the integer value path the token-mixing form is refused by the
value path's own reader rule first (UF/report_r2.md). The test is therefore on the reader, with each reader's input
grid given. Red on the composition without the rule: the drive site refuses the per-position reader by name.
"""

from __future__ import annotations

import pytest
import torch

import unit.spiking.position_fold_forms as pf

import mimarsinan.mapping.support.bias_drive as bias_drive
from mimarsinan.mapping.support.drive_rule import DriveGrid, RowBudget, least_error_drive_and_rows
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import bias_offsets

GRID = DriveGrid(step=0.0625, offset=0.0, code_max=255)
BITS = 4


def _program(form: str, offsets: bool):
    _model, flow, shape = pf.converted(form, 0)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, 0, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2
    if offsets:  # the last position's bias of neuron 0 well above every other's: the offsets move the rule's input
        from mimarsinan.transformations.perceptron.position_bias import add_bias_corrections

        corrections = torch.zeros(int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[0]))
        corrections[-1, 0] = -3.0
        add_bias_corrections(reader, corrections, lambda f: PerceptronTransformer().write_shared_bias(reader, f))
        assert bias_offsets(reader) is not None
    return flow, reader


def _every_reader_reads(monkeypatch, grid: DriveGrid) -> None:
    monkeypatch.setattr(bias_drive, "reader_input_grids",
                        lambda model_repr, absorbed=frozenset(), why=None: {
                            id(n): grid for n in bias_drive._perceptron_nodes(model_repr)})


@pytest.mark.parametrize("offsets", (False, True))
@pytest.mark.parametrize("form", ("token_mixing", "token_mixing_3", "conv_seam"))
def test_the_drive_site_reads_every_weight_row_and_every_position_s_bias(form, offsets, monkeypatch):
    flow, reader = _program(form, offsets)
    _every_reader_reads(monkeypatch, GRID)
    both = PerceptronTransformer(positions=True)
    want = least_error_drive_and_rows(both.get_effective_weight(reader), both.get_effective_bias(reader), GRID,
                                      2 ** (BITS - 1) - 1, fractional=False, budget=RowBudget())
    bias_drive.stamp_bias_row_drives(flow, weight_bits=BITS)
    assert bias_row_drive(reader) == want[0]
    if offsets:  # the fixture: one position's bias apart moves the rule's choice from the shared bias's
        shared = least_error_drive_and_rows(both.get_effective_weight(reader),
                                            PerceptronTransformer().shared_effective_bias(reader), GRID,
                                            2 ** (BITS - 1) - 1, fractional=False, budget=RowBudget())
        assert shared[0] != want[0]


def test_a_reader_of_one_fold_per_input_reads_its_one_pair_as_before(monkeypatch):
    """Every producer (one fold per input) takes the per-input reader's pair: the drive is the one it took before."""
    flow, reader = _program("token_mixing", False)
    _every_reader_reads(monkeypatch, GRID)
    bias_drive.stamp_bias_row_drives(flow, weight_bits=BITS)
    one = PerceptronTransformer()
    for p in flow.get_perceptrons():
        if p is reader or getattr(p, "is_encoding_layer", False):
            continue
        want = least_error_drive_and_rows(one.get_effective_weight(p), one.get_effective_bias(p), GRID,
                                          2 ** (BITS - 1) - 1, fractional=False, budget=RowBudget())
        assert bias_row_drive(p) == want[0]
