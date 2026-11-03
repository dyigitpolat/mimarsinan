"""[U-BRG, W38 bias-row gauge] The always-on bias row's drive is the top code value of its reader's input grid.

A parameter-encoded bias rides an always-on axon row of the shared weight grid. On the integer value path (a declared
readout law on value-domain cores without a bias register) every perceptron's activation and input scales are 1.0, so
the row's input was the raw value 1.0: a code of the reader's entry grid (step = full scale / levels) only by accident.
Where a folded bias is large in the network's own units, the bias and not the weights set the shared grid (W38
ResNet-50 arm W at 8 bits: max|b| up to 21.9 max|w|, 82 to 95 percent of eight layers' weights at code 0, Weight
Quantization 0 of 256).

The row now carries the largest code value of its reader's input grid, the grid's full scale ``x_fs``: the shared grid
sizes ``max(max|w|, max|b| / x_fs)``, the row's code is ``round(b s / x_fs)`` and the projected bias is that code times
``x_fs / s``. Two effects follow on every perceptron of the route (stated as boundaries, pinned here): every bias is
coded on the drive's lattice (step ``x_fs / s`` instead of ``1 / s``), and where ``x_fs < 1`` the shared grid can be
coarser than the raw gauge's. A normalized currency (no value grid, or not value-domain cores), an on-chip bias register,
a law-resolution bias and the register route carry no drive, and their projections are byte-identical.

The projection is read as the Weight Quantization step applies it (the step's own entry sequence with its tuner
intercepted, then the tuner's rate-1 projection), so a tree without the drive reads its own projection here.
"""

from __future__ import annotations

import copy
import hashlib
import io
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner as Tuner,
)

BITS = 8
Q_MAX = 2 ** (BITS - 1) - 1  # 127
X_FS = 20.0  # the first layer's entry grid full scale (the verify's fixture: input full scale 20)
W_MAX = 0.05
B_MAX = 5.0 * W_MAX  # a bias 5 times the largest weight in raw units; 0.25 of x_fs max|w| at the drive
X_FS_2 = 6.0  # the second layer's entry grid full scale
LAW = {"adc_bits": 32, "mult_bits": 24, "range": "calibrated", "output_law": "identity"}
PT = PerceptronTransformer()


def _flow(x_fs=(X_FS, X_FS_2), b_max: float = B_MAX):
    """A value-domain MLP built the pipeline's way: converted under the mvm packaging, fused, an entry grid on every
    segment entry (calibrated, then set to the fixture's full scales ``x_fs``) and a readout quantizer on every
    perceptron. Layer 0 holds max|w| = 0.05 and a bias up to ``b_max``; layer 2 holds a bias below its largest weight."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4)).eval()
    flow = convert_torch_model(model, (16,), 4, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        w0 = (torch.rand(12, 16, generator=gen) * 2 - 1) * W_MAX
        w0[0, 0] = W_MAX
        b0 = (torch.rand(12, generator=gen) * 2 - 1) * b_max
        b0[1] = b_max
        b0[2] = -0.8 * b_max
        perceptrons[0].layer.weight.copy_(w0)
        perceptrons[0].layer.bias.copy_(b0)
        w2 = (torch.rand(4, 12, generator=gen) * 2 - 1) * 0.3
        w2[0, 0] = 0.3
        b2 = torch.tensor([0.05, -0.11, 0.2, 0.013])
        perceptrons[1].layer.weight.copy_(w2)
        perceptrons[1].layer.bias.copy_(b2)
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, BITS)
    calibrate_boundary_scales(flow, entries, grids, [torch.randn(32, 16, generator=gen) * 5.0])
    for grid, value in zip(grids, x_fs):
        grid.calibrate(value)
    readouts = install_readout_quantizers(perceptrons, BITS)
    for quantizer, value in zip(readouts, (8.0, 3.0)):
        quantizer.calibrate(value)
    flow.get_mapper_repr().assign_perceptron_indices()
    return flow


class _Pipeline:
    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)
        self.reporter = SimpleNamespace(report=lambda *a, **k: None)

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _value_config(**extra) -> dict:
    """The integer value path's declaration as W38's arm W documents carry it: a readout law, cores without a bias
    register (the bias on an always-on row of the weight grid), 8-bit weights, the fractional scale rule."""
    return {"device": "cpu", "weight_quantization": True, "weight_bits": BITS, "activation_bits": BITS,
            "core_semantics": "mvm", "has_bias": False, "readout_law": dict(LAW), "weight_scale_rule": "fractional",
            **extra}


def _tuner(captured: dict):
    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits = captured["quantization_bits"]
    tuner.two_scale_projection = captured["two_scale_projection"]
    tuner.fractional_scale = captured["fractional_scale"]
    tuner.float_bias = captured["float_bias"]
    tuner.bias_rows_floor = captured.get("bias_rows_floor")
    tuner.register_bias = captured.get("register_bias", False)
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    return tuner


def _wq(monkeypatch, flow, config) -> dict:
    """The WQ step's own entry sequence on ``flow`` (its tuner intercepted), then the tuner's rate-1 projection."""
    captured: dict = {}
    monkeypatch.setattr(WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
    monkeypatch.setattr(WeightQuantizationStep, "_canonicalize_starved_bias_outliers", lambda self, model: None)
    WeightQuantizationStep(_Pipeline(config, {"model": flow, "adaptation_manager": object()})).process()
    transform = _tuner(captured)._get_new_perceptron_transform(1.0)
    with torch.no_grad():
        for perceptron in flow.get_perceptrons():
            transform(perceptron)
    return captured


def _entry_full_scale(perceptron) -> float:
    return float(boundary_grid_of(perceptron.input_activation).value_range[1])


def _stamp(holder) -> float:
    """A holder's row drive read without the unit's module (1.0 without a stamp), so the test runs on any tree."""
    return float(holder.__dict__.get("bias_row_drive", 1.0))


# ---------------------------------------------------------------------------
# G-1: a bias large in raw units no longer sets the shared grid
# ---------------------------------------------------------------------------


def test_a_bias_dominating_in_raw_units_no_longer_sets_the_shared_grid(monkeypatch):
    """G-1. Layer 0's bias is 5 times its largest weight in raw units and 0.25 times ``x_fs max|w|``: the shared grid
    is the weight-only grid, ``q_max / max|w|`` under the fractional rule. Without the drive the bias sets it,
    ``q_max / max|b|``, and most weights round to code 0."""
    flow = _flow()
    w_float = PT.get_effective_weight(flow.get_perceptrons()[0]).detach().clone()
    _wq(monkeypatch, flow, _value_config())
    p = flow.get_perceptrons()[0]
    s = float(p.parameter_scale)
    assert s == pytest.approx(Q_MAX / W_MAX, rel=1e-6), f"the bias set the grid: s = {s}, q_max/max|b| = {Q_MAX / B_MAX}"
    codes = torch.round(PT.get_effective_weight(p) * s)
    torch.testing.assert_close(codes, torch.round(w_float * (Q_MAX / W_MAX)), atol=0, rtol=0)
    zero_share = float((codes == 0).float().mean())
    assert zero_share < 0.05, f"{zero_share:.3f} of the weights at code 0"


def test_the_projected_bias_is_a_row_code_at_the_drive(monkeypatch):
    """The projected bias of layer 0 is ``c x_fs / s`` with integer ``c`` inside the row's ``±q_max``, within half a
    drive step ``x_fs / (2 s)`` of the float bias (one weight step at the full-scale input)."""
    flow = _flow()
    b_float = PT.get_effective_bias(flow.get_perceptrons()[0]).detach().clone()
    _wq(monkeypatch, flow, _value_config())
    p = flow.get_perceptrons()[0]
    s = float(p.parameter_scale)
    row = PT.get_effective_bias(p) * s / X_FS
    torch.testing.assert_close(row, torch.round(row), atol=1e-4, rtol=0)
    assert float(torch.round(row).abs().max()) <= Q_MAX
    assert float((PT.get_effective_bias(p) - b_float).abs().max()) <= X_FS / (2 * s) * (1 + 1e-5)


def test_the_drive_is_the_largest_code_of_the_reader_entry_grid(monkeypatch):
    """Every perceptron reading an armed entry grid carries that grid's top code value as its row drive (``levels``
    steps), so a crossbar row driven by the reader's own input DAC realizes it."""
    from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive  # absent without the unit

    flow = _flow()
    _wq(monkeypatch, flow, _value_config())
    for perceptron in flow.get_perceptrons():
        grid = boundary_grid_of(perceptron.input_activation)
        drive = bias_row_drive(perceptron)
        assert drive == _entry_full_scale(perceptron)
        assert drive / grid.step == pytest.approx(grid.levels, abs=1e-9)


def test_the_codes_are_invariant_under_a_function_preserving_input_rescaling(monkeypatch):
    """G-1b. Feeding layer 0 the input ``c x`` with weights ``W / c`` (its entry grid at ``c x_fs``) is the same
    function, so the deployed integers must not depend on ``c``. At the drive they do not (the grid scales by ``c``);
    with the row at the raw 1.0 the bias sets the grid in both and every weight code shrinks by ``c``."""
    c = 4.0
    flow_a, flow_b = _flow(), _flow()
    p_b = flow_b.get_perceptrons()[0]
    with torch.no_grad():
        p_b.layer.weight.div_(c)
    boundary_grid_of(p_b.input_activation)  # the entry grid exists
    quantizer = next(m for m in p_b.input_activation.modules() if hasattr(m, "calibrate") and hasattr(m, "scale"))
    quantizer.calibrate(c * X_FS)
    _wq(monkeypatch, flow_a, _value_config())
    _wq(monkeypatch, flow_b, _value_config())
    a, b = flow_a.get_perceptrons()[0], flow_b.get_perceptrons()[0]
    sa, sb = float(a.parameter_scale), float(b.parameter_scale)
    torch.testing.assert_close(torch.round(PT.get_effective_weight(b) * sb), torch.round(PT.get_effective_weight(a) * sa),
                               atol=0, rtol=0)
    torch.testing.assert_close(torch.round(PT.get_effective_bias(b) * sb / (c * X_FS)),
                               torch.round(PT.get_effective_bias(a) * sa / X_FS), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# G-2: the two effects on every perceptron of the route (M-1), stated as boundaries
# ---------------------------------------------------------------------------


def test_where_the_row_does_not_set_the_grid_the_weight_codes_are_unchanged_and_the_bias_is_within_half_a_drive_step(
        monkeypatch):
    """Guard (green on both trees; the verify's M-1 rename). Layer 2's row stays below its largest weight
    (``max|b| / x_fs <= max|w|``, and ``max|b| <= max|w|``), so its grid is the weight-only grid on both trees and every
    weight code is the same. What it holds of the bias is only the bound both trees meet: within half a drive step
    ``x_fs / (2 s)`` of the float bias (the codes themselves move to the drive's lattice, the next test)."""
    flow = _flow()
    p2 = flow.get_perceptrons()[1]
    w_float = PT.get_effective_weight(p2).detach().clone()
    b_float = PT.get_effective_bias(p2).detach().clone()
    _wq(monkeypatch, flow, _value_config())
    s = float(p2.parameter_scale)
    w_max = float(w_float.abs().max())
    assert float(b_float.abs().max()) / X_FS_2 <= w_max and float(b_float.abs().max()) <= w_max  # the precondition
    assert s == pytest.approx(Q_MAX / w_max, rel=1e-6)
    torch.testing.assert_close(torch.round(PT.get_effective_weight(p2) * s), torch.round(w_float * s), atol=0, rtol=0)
    assert float((PT.get_effective_bias(p2) - b_float).abs().max()) <= X_FS_2 / (2 * s) * (1 + 1e-5)


def test_every_bias_on_the_drive_route_is_coded_on_the_drive_lattice(monkeypatch):
    """[M-1 (i)] Even where the row does not set the grid (layer 2, ``x_fs`` 6, ``max|b|`` 0.2 below ``max|w|`` 0.3),
    the bias is coded on the drive's lattice: ``b s / x_fs`` is an integer, a step of ``x_fs / s`` instead of the raw
    gauge's ``1 / s``. On the verify's measurement 3 of the 4 codes move (``b s`` = 21, -47, 85, 6 against 24, -48, 84,
    6) and the largest bias error grows from 0.0012 to 0.0067: a boundary of the rule, stated with this denominator."""
    flow = _flow()
    p2 = flow.get_perceptrons()[1]
    b_float = PT.get_effective_bias(p2).detach().clone()
    _wq(monkeypatch, flow, _value_config())
    s = float(p2.parameter_scale)
    row = PT.get_effective_bias(p2) * s / X_FS_2
    torch.testing.assert_close(row, torch.round(row), atol=1e-4, rtol=0)
    raw_codes = torch.round(b_float * s)
    moved = int((torch.round(row) * X_FS_2 != raw_codes).sum())
    assert moved == 3, f"{moved} of 4 bias codes moved to the drive lattice"


def test_a_reader_full_scale_below_one_drives_its_row_at_that_full_scale(monkeypatch):
    """[M-1 (ii), the rule for full scales below 1] Layer 0 reads an entry grid of full scale 0.5, below 1, and its
    bias (0.8 of its largest weight in raw units) does not set the raw gauge's grid. The row is driven at the reader's
    full scale 0.5 (rule F: the top code value of the reader's grid, whatever its full scale, so the row stays a code
    the reader's DAC produces and the codes stay invariant under a function-preserving input rescaling), so the row
    ``max|b| / 0.5`` = 1.6 ``max|w|`` sets the shared grid: ``s = q_max / (max|b| / 0.5)``, coarser by 1.6 than the raw
    gauge's weight-only grid. Pinned as the declared behaviour (measured on ResNet-18 arm W, report section 4)."""
    from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive  # absent without the unit

    b_max = 0.8 * W_MAX
    flow = _flow(x_fs=(0.5, X_FS_2), b_max=b_max)
    _wq(monkeypatch, flow, _value_config())
    p = flow.get_perceptrons()[0]
    assert bias_row_drive(p) == _entry_full_scale(p) and _entry_full_scale(p) < 1.0
    s = float(p.parameter_scale)
    assert s == pytest.approx(Q_MAX / (b_max / _entry_full_scale(p)), rel=1e-6)
    assert s < Q_MAX / W_MAX  # coarser than the raw gauge's weight-only grid
    row = PT.get_effective_bias(p) * s / _entry_full_scale(p)
    torch.testing.assert_close(row, torch.round(row), atol=1e-4, rtol=0)
    assert float(torch.round(row).abs().max()) == Q_MAX  # the row at the register's edge sets the grid


# ---------------------------------------------------------------------------
# G-3: routes that take no drive are byte-identical
# ---------------------------------------------------------------------------


_NO_DRIVE_ROUTES = [
    pytest.param("on-chip-register", {"has_bias": True}, id="on-chip-register"),
    pytest.param("no-readout-law", {"readout_law": None}, id="no-readout-law"),
    pytest.param("law-resolution-bias", {"has_bias": True, "bias_resolution": "law"}, id="law-resolution-bias"),
    pytest.param("register-route", {"has_bias": True, "wq_two_scale_projection": True}, id="register-route"),
    pytest.param("rows-without-a-law", {"readout_law": None, "bias_row_splitting": "auto"}, id="rows-without-a-law"),
]

#: [U-LMP, U-BRG review m-2] The base's per-perceptron digests of each route without a drive (sha256 over every
#: parameter's name and float64 bytes, then ``repr`` of ``parameter_scale`` and of ``bias_scale``; first 16 hex),
#: read on the stacked R5 candidate without U-LMP and equal to U-BRG's review reading of the tree before the drive
#: (``xtree_probe.py``): a change of the drive-1 branch moves them, which the tuner-projection comparison cannot see.
BASE_NO_DRIVE_DIGESTS = {
    "on-chip-register": ["a64c5af3a114c1cd", "c8455d962cb6850f"],
    "no-readout-law": ["a64c5af3a114c1cd", "c8455d962cb6850f"],
    "law-resolution-bias": ["a01bc1973258e758", "4db39c8a0e60d021"],
    "register-route": ["531bcd85cd1cd4a2", "c8455d962cb6850f"],
    "rows-without-a-law": ["22a1a2edd72ec508", "c8455d962cb6850f"],
}


def _projection_digest(perceptron) -> str:
    h = hashlib.sha256()
    for name, parameter in perceptron.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().double().cpu().numpy().tobytes())
    h.update(repr(float(perceptron.parameter_scale)).encode())
    h.update(repr(float(getattr(perceptron, "bias_scale", perceptron.parameter_scale))).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("route, config", _NO_DRIVE_ROUTES)
def test_no_drive_where_the_row_has_no_value_grid_reader(monkeypatch, route, config):
    """Guard (green on both trees), one per route that takes no drive: an on-chip bias register (no always-on row), a
    program without the integer law, a law-resolution bias, the register route and the rows route without a law
    carry no drive, and every parameter and scale is byte-equal to the tuner's own projection (its captured settings)
    on an unstamped copy and to the base's own projection (its pinned per-perceptron digests)."""
    declared = _value_config(**config)
    declared = {k: v for k, v in declared.items() if v is not None}
    flow, ref = _flow(), _flow()
    captured = _wq(monkeypatch, flow, declared)
    transform = _tuner(captured)._get_new_perceptron_transform(1.0)
    with torch.no_grad():
        for p, r in zip(flow.get_perceptrons(), ref.get_perceptrons()):
            transform(r)
            assert _stamp(p) == 1.0
            for (name, a), (_, b) in zip(p.named_parameters(), r.named_parameters()):
                assert torch.equal(a.detach(), b.detach()), name
            assert float(p.parameter_scale) == float(r.parameter_scale)
            assert float(getattr(p, "bias_scale", p.parameter_scale)) == float(getattr(r, "bias_scale", r.parameter_scale))
    assert [_projection_digest(p) for p in flow.get_perceptrons()] == BASE_NO_DRIVE_DIGESTS[route]


def test_the_route_predicate_needs_a_readout_law_on_value_domain_cores_with_the_bias_on_a_row():
    """The route is a conjunction: a declared readout law, value-domain cores (the spiking wires drive the always-on
    axon at the normalized full scale already), and the bias on an always-on row of the weight grid."""
    from mimarsinan.mapping.support.bias_drive import bias_rows_take_a_drive  # absent without the unit
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import bias_row_drive_route

    law = {"readout_law": dict(LAW)}
    route = dict(on_chip_bias=False, law_bias=False, register_bias=False, value_domain=True)
    assert bias_rows_take_a_drive(law, **route)
    assert not bias_rows_take_a_drive({}, **route)
    for flag in ("on_chip_bias", "law_bias", "register_bias"):
        assert not bias_rows_take_a_drive(law, **{**route, flag: True})
    assert not bias_rows_take_a_drive(law, **{**route, "value_domain": False})
    assert bias_row_drive_route(_value_config())
    for config in _NO_DRIVE_ROUTES:
        declared = {k: v for k, v in _value_config(**config.values[1]).items() if v is not None}
        assert not bias_row_drive_route(declared), config.id


# ---------------------------------------------------------------------------
# G-4: bias-row splitting, verification, the canonicalization rung, the stamp's travel
# ---------------------------------------------------------------------------


def test_bias_row_splitting_counts_rows_at_the_drive(monkeypatch):
    """With ``bias_row_splitting`` auto (the weight-only grid, the bias on k rows), k is
    ``ceil(max|b| s_w / (x_fs q_max))``: one row for layer 0 (without the drive ``ceil(max|b| s_w / q_max)`` = 5)."""
    flow = _flow()
    captured = _wq(monkeypatch, flow, _value_config(bias_row_splitting="auto"))
    assert captured["two_scale_projection"] is True
    p = flow.get_perceptrons()[0]
    rows = float(p.parameter_scale) / float(p.bias_scale)
    assert rows == pytest.approx(1.0, abs=1e-6), f"{rows} rows"
    row = PT.get_effective_bias(p) * float(p.bias_scale) / X_FS
    torch.testing.assert_close(row, torch.round(row), atol=1e-4, rtol=0)


def test_quantization_verification_accepts_the_rows_at_the_drive(monkeypatch):
    """Guard: the verification's lattice checks read the row (the bias over its drive): the projected model passes."""
    from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
        assert_effective_parameters_on_chip_grid,
    )

    for extra in ({}, {"bias_row_splitting": "auto"}):
        flow = _flow()
        _wq(monkeypatch, flow, _value_config(**extra))
        for perceptron in flow.get_perceptrons():
            assert_effective_parameters_on_chip_grid(copy.deepcopy(perceptron), Q_MAX)


def test_the_nuisance_dominator_rung_reads_the_row_at_its_drive():
    """The starved-grid rungs compare the weights with the always-on row, the bias over its drive. Layer 0 reads a grid
    of full scale 0.001 with a bias below its largest weight in raw units: its row (40 max|w|) starves the grid at the
    drive, so the rung that removes grid-dominating channels acts on it (kept or restored by its own decision check).
    Without the drive the grid is not starved, and a rung reading the raw bias returns before any removal."""
    from mimarsinan.mapping.support.bias_drive import stamp_bias_row_drives  # absent without the unit
    from mimarsinan.transformations.perceptron.bias_canonicalization import canonicalize_starved_bias_outliers

    flow = _flow(x_fs=(0.001, X_FS_2), b_max=0.8 * W_MAX)
    stamp_bias_row_drives(flow)
    batches = [torch.randn(16, 16, generator=torch.Generator().manual_seed(3))]
    report = canonicalize_starved_bias_outliers(flow, batches, bits=BITS)
    assert report["removed"] + report["removal_restored"] >= 1, report


def test_the_drive_stamp_travels_with_the_model_through_fusion_copies_and_the_step_cache(monkeypatch):
    """The drive is a plain attribute: a deep copy, the step cache's whole-object pickle (``torch.save`` and
    ``torch.load``) and a later normalization fusion keep it, and the state_dict holds no new key."""
    flow = _flow()
    keys = sorted(flow.state_dict())
    _wq(monkeypatch, flow, _value_config())
    drives = [_stamp(p) for p in flow.get_perceptrons()]
    assert drives == [_entry_full_scale(p) for p in flow.get_perceptrons()] and min(drives) > 1.0
    buffer = io.BytesIO()
    torch.save(flow, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    copied = copy.deepcopy(flow)
    for perceptron in copied.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    for model in (loaded, copied):
        assert [_stamp(p) for p in model.get_perceptrons()] == drives
    assert sorted(flow.state_dict()) == keys


def test_a_projection_without_a_stamp_is_the_raw_gauge(monkeypatch):
    """Guard (green on both trees): the projection class alone, on a perceptron without a stamp, is byte-equal to the
    raw gauge's formula (the shared grid ``q_max / max(max|w|, max|b|)``)."""
    flow = _flow()
    p = flow.get_perceptrons()[0]
    NormalizationAwarePerceptronQuantization(BITS, "cpu", 1.0, fractional_scale=True).transform(p)
    assert float(p.parameter_scale) == pytest.approx(Q_MAX / B_MAX, rel=1e-6)
