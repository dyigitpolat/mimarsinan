"""[U-PFOLD] The fold per core position (the fourth verify's M-2; ruling (b) of 'U-TGAUGE4's Verify'; the design's Part 2).

A deployed core folds each input at the gauge of the producer of the element it reads. Where one fold per input of the
perceptron holds every element so, the stamp keeps that form and its bytes. Where it does not (a token-mixing Linear
over a join of producers at different gauges, a convolution over a seam of two producers), the program maps the
reading perceptron to one core per position, so the stamp holds one row per position, and every reader takes it per
position: the mappers (a linear's positions of equal rows share one bank, a convolution holds one bank set per
distinct patch fold), both walks and the deployed program. The fold's named refusal stays only where positions share
one core, which no program as built does.

Each test was red on U-TGAUGE4's tree by the fold's refusal ("differ within an input channel") or, for the
convolution seam, unnamed or silently off (the stamp read a per-element vector as a linear's rows).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import unit.spiking.position_fold_forms as pf

from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

SEEDS = (0, 1, 2)
SEMANTICS = ("streamed_lif", "windowed_lif")


def _stamped(form: str, seed: int):
    _model, flow, shape = pf.converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, seed, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    assert perceptrons[-1].per_input_scales.dim() == 2, "the reader's fold is held per position"
    return flow, perceptrons


@pytest.mark.parametrize("form", pf.LINEAR_FORMS + pf.CONV_FORMS + ("shared_rows",))
def test_the_stamp_holds_one_row_per_position_where_one_fold_per_input_cannot(form):
    """The reader's stamp is ``(positions, in_features)``: a linear one row per leading position, a convolution one per
    output position (its patch's fold). On U-TGAUGE4's tree the stamp refused by name."""
    flow, perceptrons = _stamped(form, 0)
    reader = perceptrons[-1]
    stamp = reader.per_input_scales
    assert stamp is not None and stamp.dim() == 2
    assert int(stamp.shape[1]) == int(reader.layer.weight.shape[1])
    assert bool((stamp != stamp[:1]).any()), "the positions hold different folds"


@pytest.mark.parametrize("form", ("token_mixing", "tree481", "conv_seam"))
@pytest.mark.parametrize("family", ("timing", "count"))
def test_each_core_folds_each_input_at_its_producer_s_gauge(form, family):
    """The m-5 check on the deployed program: each reading core's weight on each axon over its raw weight is the gauge
    of the element that axon reads (its source's producer gauge, read at its source index)."""
    if family == "timing":
        read = pf.timing_read(form, 0)
        ir, flow = read["ir"], read["flow"]
    else:
        program = pf.count_program(form, 0)
        ir, flow = program.ir, program.flow
    errors = pf.fold_errors(ir, flow)
    assert errors["cores"] > 0
    assert errors["max_rel"] <= 1e-6, errors


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gauges", ("one", "drawn"))
@pytest.mark.parametrize("form", pf.LINEAR_FORMS + pf.CONV_FORMS + ("shared_rows",))
def test_the_timing_walk_reads_as_the_deployed_program(form, gauges, seed):
    """The timing walk (each core's input at its fold over its one input scale, element by element) against the
    cascaded identity program's read, element by element. The convolution seam crashed in the walk (a per-channel fold
    on a channel-first input read on the last axis)."""
    pf.assert_equal_reads(pf.timing_read(form, seed, gauges))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("gauges", ("one", "drawn"))
@pytest.mark.parametrize("form", pf.LINEAR_FORMS + pf.CONV_FORMS + ("shared_rows", "mixed_token_mixing", "tree1301"))
def test_the_count_walk_reads_as_the_deployed_program(form, gauges, semantics, seed):
    """The count walk (both LIF semantics, inside the measurement plane) against the deployed flow's read. The
    convolution seam read 119 to 187 of 288 elements off silently with one gauge per producer; the MIXED gather's
    reader reads its spikes at its fold per position."""
    program = pf.count_program(form, seed, semantics, gauges)
    pf.assert_equal_reads(pf.count_read(program, semantics, seed))


def _banks(ir, perceptron_index) -> dict:
    cores = [c for c in ir.get_neural_cores()
             if getattr(c, "perceptron_index", None) == perceptron_index and not getattr(c, "_depth_relay", False)]
    banks: dict = {}
    for c in cores:
        banks.setdefault(getattr(c, "weight_bank_id", None), []).append(c)
    return banks


def _reader_index(flow, perceptron) -> int:
    return next(n.perceptron_index for n in flow.get_mapper_repr()._exec_order
                if getattr(n, "perceptron", None) is perceptron)


def test_positions_of_equal_folds_share_one_bank():
    """Correction 10: a linear's positions whose fold rows are equal share one weight bank and its shared cores; a
    distinct row has its own. Three tokens, the first and the last of one producer: two banks, three cores."""
    program = pf.count_program("shared_rows", 0)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    banks = _banks(program.ir, _reader_index(program.flow, reader))
    assert None not in banks, "every position is a shared core of its row's bank"
    assert sorted(len(cores) for cores in banks.values()) == [1, 2]


@pytest.mark.parametrize("form", ("conv_seam", "conv_seam_tall"))
def test_a_convolution_holds_one_bank_set_per_distinct_patch_fold(form):
    """A convolution's positions share a bank where their patches' folds are equal: one bank per distinct patch fold
    (the seam's rows), every output position a core; inside one tall producer the patches share one fold."""

    program = pf.count_program(form, 0)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    from mimarsinan.transformations.perceptron.position_fold import position_rows
    rows, index = position_rows(reader)
    banks = _banks(program.ir, _reader_index(program.flow, reader))
    assert len(banks) == int(rows.shape[0]) <= int(index.numel())  # a padding slot holds the fold 1.0
    if form == "conv_seam_tall":
        assert int(rows.shape[0]) < int(index.numel()), "the patches inside one producer share one fold"
    assert sum(len(cores) for cores in banks.values()) == int(index.numel())
    for cores in banks.values():  # each bank's cores: the positions of one patch fold
        assert len({tuple(rows[int(index[c.perceptron_output_column])].tolist()) for c in cores}) == 1


@pytest.mark.parametrize("form", ("token_mixing", "shared_rows", "conv_seam"))
def test_no_core_serves_positions_of_different_folds(form):
    """The fold's refusal stays only where positions share one core: on every program the mappers build, each bank's
    cores hold one fold (the m-5 check per bank), and a core set over positions of two folds refuses by name."""

    program = pf.count_program(form, 1)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    from mimarsinan.transformations.perceptron.position_fold import PositionFoldError, assert_one_fold_per_core
    for cores in _banks(program.ir, _reader_index(program.flow, reader)).values():
        mats = [np.asarray(c.get_core_matrix(program.ir), dtype=np.float64) for c in cores]
        assert all(np.array_equal(mats[0], m) for m in mats[1:])
    with pytest.raises(PositionFoldError, match="different folds would share one core"):
        assert_one_fold_per_core("reader", torch.tensor([[1.0, 2.0], [1.0, 3.0]]))


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_a_per_input_reader_refuses_a_per_position_fold_by_name(form):
    """Closure by construction: the per-input readers (``get_effective_weight``, ``input_scale``) refuse a perceptron
    whose fold is held per position by name, so a reader that does not take the one per-position reader cannot read
    it silently."""
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    _flow, perceptrons = _stamped(form, 0)
    from mimarsinan.transformations.perceptron.position_fold import PositionFoldError
    reader = perceptrons[-1]
    with pytest.raises(PositionFoldError, match="held per core position"):
        PerceptronTransformer().get_effective_weight(reader)
    with pytest.raises(PositionFoldError, match="held per core position"):
        PerceptronTransformer().input_scale(reader)
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    apply_negative_shift_bias(reader, 0.3)  # each position its own bias: no one effective bias holds them
    with pytest.raises(PositionFoldError, match="held per core position"):
        PerceptronTransformer().get_effective_bias(reader)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_one_fold_per_input_keeps_today_s_stamp(semantics):
    """Guard (green before and after): a join whose every input channel holds one gauge keeps the per-input stamp,
    ``(in_features,)``, and the program's counts are the walk's."""
    import unit.spiking.host_gather_forms as hg

    built = hg.build("dense_concat", 0, 1.3, semantics, wq=False)
    for p in built.perceptrons:
        stamp = getattr(p, "per_input_scales", None)
        assert stamp is None or stamp.dim() == 1
    mismatches, compared, _ = hg.compare(built, semantics, n=8)
    assert compared > 0 and sum(mismatches.values()) == 0


def test_a_boundary_slice_is_divided_by_its_own_elements_gauges():
    """The deployed boundary (both twins) divides a producer's slice by the gauges of the elements it holds, the
    producer's first ``size`` (the verify's tree 1301: a per-token table longer than the slice a level stage reads)."""
    from types import SimpleNamespace

    from mimarsinan.spiking.segment_boundary import normalize_boundary_slices_numpy, normalize_boundary_slices_torch

    piece = SimpleNamespace(node_id=3, offset=1, size=2)
    seg = np.arange(1.0, 9.0).reshape(2, 4)
    table = np.array([2.0, 4.0, 8.0])
    want = seg.copy()
    want[:, 1:3] /= table[:2]
    np.testing.assert_array_equal(normalize_boundary_slices_numpy([piece], seg, {3: table}), want)
    torch.testing.assert_close(normalize_boundary_slices_torch([piece], torch.as_tensor(seg), {3: table}),
                               torch.as_tensor(want))
