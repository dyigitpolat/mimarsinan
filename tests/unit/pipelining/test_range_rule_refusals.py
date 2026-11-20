"""[U-BRANGE2 stage 1] The range sites' refusals and the absolute floor (the first record's m-2; the extended fuzz's
ruling, item 3).

Under the rule a site's range is its own least error at any magnitude: the search's interval is relative to the
calibration maximum, so the absolute floor ``MIN_BOUNDARY_SCALE = 1e-3`` (a constant in network units over the rule's
choice) is dropped. A site whose calibration values are all zero has no range to search: it is refused by name, as a
toolchain refusal class, with the measured condition (every magnitude the calibration batches send through it is 0),
never "never reached" and never a bare ``RuntimeError``: Boundary Quantization's two refusals, Readout Law's readout
and lane sites, and a bank's payload grid (formerly floored without a refusal at zero).
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization import boundary_quantization_step as boundary_step
from mimarsinan.pipelining.pipeline_steps.quantization import readout_law_step as readout_step


def _refusal_class():
    cls = getattr(boundary_step, "BoundaryQuantizationRefusal", None)
    if cls is None:
        pytest.fail("Boundary Quantization raises no toolchain refusal class (a bare RuntimeError)")
    return cls


def test_a_zero_boundary_is_refused_by_name_with_its_measured_condition():
    cls = _refusal_class()
    reached, dead = Perceptron(4, 4, name="reached"), Perceptron(4, 4, name="dead")
    quantizers = boundary_step.install_boundary_quantizers([reached, dead], 8)
    with pytest.raises(cls) as caught:
        boundary_step.calibrate_boundary_scales(reached, [reached, dead], quantizers, [torch.randn(16, 4)])
    text = str(caught.value)
    assert issubclass(cls, ToolchainRefusal) and "dead" in text and "every magnitude" in text and "is 0" in text
    assert "1 calibration batch" in text and "never reached" not in text


def test_a_boundary_quantization_without_an_entry_is_refused_as_a_toolchain_refusal(monkeypatch):
    cls = _refusal_class()
    monkeypatch.setattr(boundary_step, "segment_entry_perceptrons", lambda *a, **k: [])
    monkeypatch.setattr(boundary_step, "operand_entry_perceptrons", lambda *a, **k: [])
    monkeypatch.setattr(boundary_step, "lane_absorbed", lambda *a, **k: frozenset())

    class _Model:
        def get_mapper_repr(self):
            return None

    step = boundary_step.BoundaryQuantizationStep.__new__(boundary_step.BoundaryQuantizationStep)
    step.pipeline = type("P", (), {"config": {"activation_bits": 8}})()
    step.get_entry = lambda key: _Model()
    with pytest.raises(cls, match="no segment-entry perceptrons"):
        step.process()


def test_a_boundary_below_1e_3_takes_its_least_error_range_with_no_floor():
    p = Perceptron(4, 4)
    (q,) = boundary_step.install_boundary_quantizers([p], 8)
    g = torch.Generator().manual_seed(2)
    (row,) = boundary_step.calibrate_boundary_scales(p, [p], [q], [torch.randn(64, 4, generator=g) * 1e-5])
    assert 0.0 < q.grid.scale < 1e-3 and q.grid.scale == pytest.approx(row.scale, rel=1e-6)
    assert not hasattr(boundary_step, "MIN_BOUNDARY_SCALE")


def test_a_zero_readout_is_refused_by_name_with_its_measured_condition():
    names = [("p", readout_step.RangeCensus(0.0, 0.0, 0.0, 0, 128, 0, 0.0, 0.0, 0.0, 0.0, 0.0))]
    with pytest.raises(readout_step.ReadoutLawCalibrationError) as caught:
        readout_step._refuse_zero_sites(names, 2, "the range site(s)")
    assert "every magnitude" in str(caught.value) and "'p'" in str(caught.value)
    assert issubclass(readout_step.ReadoutLawCalibrationError, ToolchainRefusal)


def test_the_zero_site_refusal_counts_every_zero_site_s_entries():
    """[U-BRANGE2 stage 1e, the sweep's F-E64] The refusal's measured condition reads every zero site, its names and
    its entries summed over them (128 and 64: 192), not the first site's census alone. Red on F-E64."""
    def zero(entries):
        return readout_step.RangeCensus(0.0, 0.0, 0.0, 0, entries, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(readout_step.ReadoutLawCalibrationError) as caught:
        readout_step._refuse_zero_sites([("p", zero(128)), ("q", zero(64))], 2, "the range site(s)")
    assert "'p'" in str(caught.value) and "'q'" in str(caught.value) and "over 192 entries" in str(caught.value)


def test_a_bank_payload_grid_at_zero_is_refused_by_name(monkeypatch):
    """A re-snapped bank's payload grid whose calibration values are all zero: formerly floored at 1e-3 without a
    refusal (``readout_law_step.py:125-126`` in R5's numbering), now refused with the condition."""
    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block, operand_platform

    platform = operand_platform(payload_bits=4, grouped=False)  # no consumer groups: every bank re-snaps its payload
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    banks = [op for op in operand_perceptrons(flow.get_mapper_repr()) if op.payload_quantizer is not None]
    assert banks, "fixture: a re-snapped bank"
    payload = {id(op.payload_quantizer) for op in banks}
    rule = readout_step.least_squares_pooled

    def zero_payload(model, sites, unions, batches, **kwargs):
        rows = rule(model, sites, unions, batches, **kwargs)
        return [readout_step.RangeCensus(0.0, 0.0, 0.0, 0, r.entries, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
                if id(s.module) in payload else r for s, r in zip(sites, rows)]

    monkeypatch.setattr(readout_step, "least_squares_pooled", zero_payload)
    perceptrons = list(flow.get_perceptrons()) + banks
    quantizers = readout_step.install_readout_quantizers(perceptrons, 8)
    batches = [torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    with pytest.raises(readout_step.ReadoutLawCalibrationError, match="payload.*every magnitude"):
        readout_step.calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
