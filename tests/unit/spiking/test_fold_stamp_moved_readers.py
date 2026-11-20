"""[U-PFOLD r3] The three readers moved to the one reader's accessor read the values they read before, site by site.

Each site read the stamp directly in revision 2; it now reads ``position_fold.position_folds``. Each test computes the
site's value as revision 2 read it (the stamp itself, read here directly) beside the value the site now gives, on the
token-mixing form (a linear over two positions, one producer each) and on the convolution seam (patches over the seam
mix two producers), both built by the converter:

- the event-serial twin's per-cycle fold arms only where the mapper's unfold holds one core per position (the
  position count against the unfold's cores);
- the mappers' banks: each bank keyed by one position's fold row and bias row, and the assertion of one fold per core
  given each bank's rows;
- the saturation bound's declared input range: the boundary grid's range over each position's own fold row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import unit.spiking.position_fold_forms as pf

from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

FORMS = ("token_mixing", "conv_seam")


def _reader(form: str, seed: int = 0):
    _model, flow, shape = pf.converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, seed, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    assert int(torch.unique(reader.per_input_scales, dim=0).shape[0]) > 1, "the positions hold more than one fold"
    return reader


@pytest.mark.parametrize("form", FORMS)
def test_the_per_cycle_fold_arms_only_with_one_core_per_position(form):
    """``_arm_cycle_fold``: the twin's per-cycle fold arms where the unfold's cores equal the positions (revision 2's
    read: the stamp's row count), and keeps the twin's own arithmetic otherwise."""
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.models.nn.lif_kernels import measurement_plane
    import mimarsinan.spiking.segment_policy_lif_serial as serial

    reader = _reader(form)
    law = SomaLaw(firing_mode="Novena", thresholding_mode="<=", firing_granularity="per_cycle",
                  membrane_arithmetic="unbounded", membrane_bits=0)
    lif = LIFActivation(T=2, activation_scale=reader.activation_scale)
    n_slots = int(reader.layer.weight.shape[1])
    events = torch.zeros(2, 1, 3, 4)
    positions = int(reader.per_input_scales.shape[0])  # revision 2's read
    armed = {}
    for cores in (positions - 1, positions, positions + 1):
        node = SimpleNamespace(serial_slot_unfold=lambda input_shape, n_slots, k=cores: SimpleNamespace(
            n_slots=n_slots, source_size=int(torch.tensor(input_shape).prod()), n_cores=k))
        with measurement_plane():
            slot = serial._arm_cycle_fold(law, node, reader, lif, [events])
        armed[cores] = slot is not None and slot.unfold.n_cores == cores and slot.perceptron is reader
    assert armed == {positions - 1: False, positions: True, positions + 1: False}


@pytest.mark.parametrize("form", FORMS)
def test_the_banks_key_and_assert_each_position_s_own_fold_row(form, monkeypatch):
    """``position_banks``: one bank per distinct (fold row, bias row) in order of first position, and the assertion of
    one fold per core given each bank's fold rows, as revision 2 read them from the stamp."""
    import mimarsinan.mapping.mappers.position_cores as cores
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
    from mimarsinan.transformations.perceptron.position_fold import effective_bias_rows, effective_weight_rows

    reader = _reader(form)
    asserted: list = []
    real = cores.assert_one_fold_per_core

    def recording(name, folds):
        asserted.append(folds.detach().clone())
        real(name, folds)

    monkeypatch.setattr(cores, "assert_one_fold_per_core", recording)
    weights, biases, groups = cores.position_banks(reader)
    stamp = reader.per_input_scales  # revision 2's read
    held = effective_bias_rows(reader, PerceptronTransformer().shared_effective_bias(reader))
    keys: dict = {}
    expected: list = []
    for p in range(int(stamp.shape[0])):
        key = stamp[p].detach().cpu().numpy().tobytes() + held[p].detach().cpu().numpy().tobytes()
        if key not in keys:
            keys[key] = len(expected)
            expected.append([])
        expected[keys[key]].append(p)
    assert groups == expected and len(expected) > 1
    assert len(asserted) == len(expected)
    for folds, group in zip(asserted, expected):
        assert torch.equal(folds, stamp[group]) and folds.dtype == stamp.dtype
    rows, index = effective_weight_rows(reader)
    assert torch.equal(weights, rows) and torch.equal(biases, held)
    assert int(index.numel()) == int(stamp.shape[0])


@pytest.mark.parametrize("form", FORMS)
def test_the_declared_input_range_divides_by_each_position_s_fold_row(form):
    """``declared_input_range``: the boundary grid's range, hulled with zero, over each position's own fold row
    (revision 2's divisor: the stamp), ``(positions, in_features)``, on the weight's device."""
    from mimarsinan.models.nn.activations.value_quantizer import ValueGridQuantizer, wire_output_grid
    from mimarsinan.transformations.perceptron.bias_saturation import declared_input_range

    reader = _reader(form)
    quantizer = ValueGridQuantizer(5)
    quantizer.calibrate(2.0)
    reader.append_input_wire_op(quantizer)
    grid = wire_output_grid(reader.input_activation)
    assert grid is not None
    low, high = grid.value_range
    stamp = reader.per_input_scales  # revision 2's read
    a, b = min(low, 0.0) / stamp, max(high, 0.0) / stamp
    got = declared_input_range(reader)
    assert got is not None
    assert torch.equal(got.low, torch.minimum(a, b)) and torch.equal(got.high, torch.maximum(a, b))
    assert tuple(got.low.shape) == tuple(stamp.shape) and got.low.device == reader.layer.weight.device
    assert int(torch.unique(got.high, dim=0).shape[0]) > 1, "the positions' ranges differ"
