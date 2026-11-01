"""[U-WQB A, phase C defect D-WQ] The pipeline sites of the two-scale projection's on-chip register route.

The WQ step names the route (``test_napq_register_bias.py``); the post-install verification and Soft Core Mapping's
export read the same route from the run's config, so both parity sides see one contract: on the register route the
bias is an integer on the weight lattice within the span ``±q_max r`` the export allocates (``r`` from the bias's own
codes), and without it the bias stays inside its own grid's ``±q_max`` register range.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel

import mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step as qv_module
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    QuantizationVerificationStep,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

BITS = 4
Q_MAX = 7
S_W = 58.0
PT = PerceptronTransformer()


def _register_coded(codes=(40.0, -23.0, 5.0)) -> Perceptron:
    """A perceptron as the register route leaves it: weights within ±7 on s_w = 58, bias codes 40, -23 and 5 on the
    same lattice (span r = ceil(40 / 7) = 6), ``bias_scale == parameter_scale``."""
    p = Perceptron(len(codes), 4, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    gen = torch.Generator().manual_seed(0)
    w_codes = torch.randint(-Q_MAX, Q_MAX + 1, (len(codes), 4), generator=gen).double()
    with torch.no_grad():
        p.layer.weight.copy_(w_codes / S_W)
        p.layer.bias.copy_(torch.tensor(codes, dtype=torch.float64) / S_W)
    p.set_parameter_scale(S_W)
    return p


class _Pipeline:
    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)
        self.reporter = SimpleNamespace(report=lambda *a, **k: None)

    def get_entry(self, _step, key):
        return self.entries[key]


def _verify(monkeypatch, perceptron, **config):
    monkeypatch.setattr(qv_module, "make_basic_trainer", lambda pipeline, model: None)
    cfg = {"device": "cpu", "weight_quantization": True, "weight_bits": BITS, **config}
    model = SimpleNamespace(get_perceptrons=lambda: [perceptron])
    QuantizationVerificationStep(_Pipeline(cfg, {"model": model})).process()


def test_verification_accepts_a_register_bias_beyond_q_max(monkeypatch):
    """T-A3. The weight-lattice bias at code 40 (q_max 7, span 6) passes on the register route. R raises
    'bias outside its ±7 register range'."""
    _verify(monkeypatch, _register_coded(), wq_two_scale_projection=True, has_bias=True)


def test_verification_still_refuses_an_off_lattice_register_bias(monkeypatch):
    p = _register_coded()
    with torch.no_grad():
        p.layer.bias.data[0] += 0.5 / S_W
    with pytest.raises(AssertionError):
        _verify(monkeypatch, p, wq_two_scale_projection=True, has_bias=True)


@pytest.mark.parametrize("config", [
    {"wq_two_scale_projection": False, "has_bias": True},
    {"wq_two_scale_projection": True, "has_bias": False},
])
def test_without_the_register_route_the_bias_keeps_its_q_max_range(monkeypatch, config):
    """Guard: the shared grid and the param-encoded rows keep the ±q_max register range on the bias's own grid."""
    with pytest.raises(AssertionError, match="register range"):
        _verify(monkeypatch, _register_coded(), **config)


# ---------------------------------------------------------------------------
# Soft Core Mapping exports the register route's codes
# ---------------------------------------------------------------------------


def _scm_ir(tmp_path, *, two_scale: bool):
    """Soft Core Mapping (the conftest harness of ``test_core_quantization_verification_step.py``) on a tiny model
    whose perceptrons carry register-route parameters: every bias code beyond the weight register."""
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.normalization = nn.Identity()
        out_f, in_f = p.layer.weight.shape
        gen = torch.Generator().manual_seed(out_f + in_f)
        codes = torch.randint(-Q_MAX, Q_MAX + 1, (out_f, in_f), generator=gen).float()
        bias = torch.full((out_f,), 33.0)
        bias[::2] = -29.0
        theta = float(p.activation_scale)
        with torch.no_grad():
            p.layer.weight.copy_(codes / S_W * theta)
            p.layer.bias.copy_(bias / S_W * theta)
        p.set_parameter_scale(S_W)
    cfg = {**default_config(), "weight_quantization": True, "weight_bits": BITS, "firing_mode": "Default",
           "onchip_majority_gate": False, "scm_torch_sim_parity_check": False, "wq_two_scale_projection": two_scale,
           "has_bias": True}
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path / "cache"))
    pipeline.seed("fused_model", model, step_name="Normalization Fusion")
    pipeline.seed("platform_constraints_resolved",
                  {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20, "has_bias": True}],
                   "weight_bits": BITS, "target_tq": 4}, step_name="Model Configuration")
    scm = SoftCoreMappingStep(pipeline)
    scm.name = "Soft Core Mapping"
    pipeline.prepare_step(scm)
    scm.run()
    return pipeline.cache["Soft Core Mapping.ir_graph"]


def _core_biases(ir) -> np.ndarray:
    return np.concatenate([np.asarray(c.hardware_bias).reshape(-1) for c in ir.get_neural_cores()
                           if c.hardware_bias is not None and getattr(c, "perceptron_index", None) is not None])


def test_soft_core_mapping_exports_the_register_codes(tmp_path):
    """The register route's codes (33 and -29 at q_max 7) reach the program unclipped. R clips them to 7 and -8."""
    biases = _core_biases(_scm_ir(tmp_path, two_scale=True))
    assert set(np.unique(biases).tolist()) <= {33, -29}
    assert {33, -29} <= set(np.unique(biases).tolist())


def test_without_the_route_soft_core_mapping_keeps_the_weight_register(tmp_path):
    """Guard: without the two-scale flag the shared grid's export saturates as before."""
    biases = _core_biases(_scm_ir(tmp_path, two_scale=False))
    assert biases.max() <= Q_MAX and biases.min() >= -Q_MAX - 1


def test_the_register_route_is_a_run_fact_not_a_declaration():
    """The grids carry the route beside the declared rules; the route is no value-domain declaration."""
    grids = ParameterGrids.from_config({"bias_resolution": "law"}, register_bias=True)
    assert (grids.fractional_scale, grids.law_bias, grids.register_bias) == (False, True, True)
    assert ParameterGrids(register_bias=True).declared is False
    assert ParameterGrids.from_config({}) == ParameterGrids()
