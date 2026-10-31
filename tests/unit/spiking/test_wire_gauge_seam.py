"""[U-SC S-1, SD-10a D-10a-1 and SD-10c D-10c-3] The wire gauge is established before every walk that reads it.

The chip-aligned LIF walk is a WIRE-currency twin: every perceptron output travels as a rate, and a host op that computes
on values (a residual add, a host convolution) decodes each source at its producer's gauge through the
``ScaleNormalizingWrapper`` slots that ``establish_wire_gauge`` arms (the calls Weight Quantization and Soft Core
Mapping make). On RC4 the exact-QAT AQ install armed only the ``offload`` placement, and LIF Adaptation's entry repair
armed only a graph with a heterogeneous join, so the C5 programs (a subsumed encoding at a non-unit theta feeding host
residual adds, no heterogeneous join) were walked UNARMED by the AQ D-hat, by LIF Adaptation's twin and by its
retention-gate metric: the twin read the residual adds in rate units (SD-10a: 0.0092 against the armed 0.2252 on
5,000 E01, and the gate refused 0.0118 < 0.3042 x 0.05). S4's PyramidNet stops at the D-hat walk itself
(``mixed wire/absolute fan-in ... (ComputeOpMapper)``), the same root.

The seam now runs on every wire-currency LIF graph at the first stage that walks the deployed composition (the
exact-QAT AQ install, and LIF Adaptation's entry), and the D-hat gauge and the LIF finalize twin refuse by name a graph
the seam never established (a graph on which the seam would change nothing needs none).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    deployed_lif_gauge_forward,
    install_lif_entry_input_quantizers,
)

from unit.spiking.host_gather_forms import (
    T,
    count_mismatches,
    lif_net,
    samples,
    scm_sequence,
    scm_window_counts,
    Program,
)

THETA_ENC = 2.185


def _exact_config(placement: str = "subsume", **extra) -> dict:
    return {
        "lif_exact_qat": True, "spiking_mode": "lif", "firing_mode": "Default", "cycle_accurate_lif_forward": True,
        "lif_per_hop_retiming": True, "simulation_steps": T, "thresholding_mode": "<=",
        "encoding_layer_placement": placement, "input_data_scale": 1.0, **extra,
    }


def _unarmed_c5_form(seed: int = 0, theta_enc: float = THETA_ENC):
    """The C5 form as the AQ install meets it: a subsumed encoding at a non-unit theta, host residual add, no gauge yet."""
    shape, flow, perceptrons, thetas = lif_net("encoding_skip", seed, theta_enc)
    return shape, flow, perceptrons, thetas


def _scm_program(flow, perceptrons, shape, thetas) -> Program:
    """Soft Core Mapping's own sequence on a COPY: the deployed composition the metrics must read."""
    deployed = copy.deepcopy(flow)
    deployed_perceptrons = list(deployed.get_perceptrons())
    repr_, ir, hybrid = scm_sequence(deployed, deployed_perceptrons, wq=False, retimed=True)
    return Program("encoding_skip", shape, deployed, repr_, ir, hybrid, deployed_perceptrons, thetas, False)


def _walk_counts(model, perceptrons, x):
    rec: dict = {}
    with torch.no_grad():
        out = chip_aligned_segment_forward(model, x, T, retime=True, soma_law=DEFAULT_SOMA_LAW, node_value_recorder=rec)
    counts = {i: torch.round(rec[id(p)].reshape(x.shape[0], -1) / float(p.activation_scale) * T).double().numpy()
              for i, p in enumerate(perceptrons) if id(p) in rec}
    return out, counts


def _armed_add(flow):
    return next(n for n in flow.get_mapper_repr().execution_order()
                if isinstance(n, ComputeOpMapper) and n.name == "add")


# ---------------------------------------------------------------------------
# The seam at the exact-QAT AQ install, for every placement
# ---------------------------------------------------------------------------


def test_the_aq_install_seam_establishes_a_subsumed_encoding_graph():
    from mimarsinan.spiking.wire_gauge_seam import wire_gauge_established

    _, flow, _, _ = _unarmed_c5_form()
    assert _armed_add(flow).output_scale is None, "fixture: the graph arrives unarmed"
    install_lif_entry_input_quantizers(flow, _exact_config("subsume"))
    assert wire_gauge_established(flow)
    add = _armed_add(flow)
    assert add.output_scale is not None and add.per_source_scales is not None


def test_the_aq_install_seam_is_off_without_the_exact_arm():
    _, flow, _, _ = _unarmed_c5_form()
    install_lif_entry_input_quantizers(flow, _exact_config(lif_exact_qat=False))
    assert _armed_add(flow).output_scale is None


def test_the_lif_retention_metric_equals_the_scm_metric_on_the_c5_form():
    """LIF Adaptation's retention gate reads the model's own forward, the chip-aligned twin the finalize installs; after
    the AQ install seam it reads the composition Soft Core Mapping deploys, count for count and prediction for
    prediction (RC4: the twin walked the unarmed graph)."""
    shape, flow, perceptrons, thetas = _unarmed_c5_form()
    install_lif_entry_input_quantizers(flow, _exact_config())
    program = _scm_program(flow, perceptrons, shape, thetas)
    x = samples(program, 32, 0)
    nf_out, nf = _walk_counts(flow, perceptrons, x)
    scm_out, scm = scm_window_counts(program, x, "windowed_lif")
    assert count_mismatches(nf, scm) == {pi: 0 for pi in scm}
    nf_grid = nf_out.double() * T / float(perceptrons[-1].activation_scale)
    assert torch.equal(nf_grid.argmax(1), scm_out.double().argmax(1))
    assert float((nf_grid - scm_out.double()).abs().max()) < 1e-3


def test_the_aq_d_hat_reads_the_deployed_composition():
    shape, flow, perceptrons, thetas = _unarmed_c5_form()
    cfg = _exact_config()
    install_lif_entry_input_quantizers(flow, cfg)
    program = _scm_program(flow, perceptrons, shape, thetas)
    x = samples(program, 32, 0)
    gauge = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        d_hat = gauge(x).double() * T / float(perceptrons[-1].activation_scale)
    scm_out, _ = scm_window_counts(program, x, "windowed_lif")
    assert float((d_hat - scm_out.double()).abs().max()) < 1e-3


def test_the_d_hat_re_derives_a_stamped_clone_at_its_current_thetas():
    """The exact-QAT ladder trains theta after the install seam: the D-hat measures what Soft Core Mapping would deploy
    from the CURRENT thetas, not the install-time stamps."""
    shape, flow, perceptrons, thetas = _unarmed_c5_form()
    cfg = _exact_config()
    install_lif_entry_input_quantizers(flow, cfg)
    with torch.no_grad():
        perceptrons[1].activation_scale.mul_(1.3)
    thetas = [float(p.activation_scale) for p in perceptrons]
    program = _scm_program(flow, perceptrons, shape, thetas)
    x = samples(program, 32, 1)
    gauge = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        d_hat = gauge(x).double() * T / float(perceptrons[-1].activation_scale)
    scm_out, _ = scm_window_counts(program, x, "windowed_lif")
    assert float((d_hat - scm_out.double()).abs().max()) < 1e-3


# ---------------------------------------------------------------------------
# LIF Adaptation's entry establishes every LIF graph, classifiable or not
# ---------------------------------------------------------------------------


def _lif_step(config):
    from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import LIFAdaptationStep

    step = object.__new__(LIFAdaptationStep)
    step.pipeline = SimpleNamespace(config=config)
    return step


def test_the_lif_adaptation_entry_establishes_a_classifiable_graph():
    from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
    from mimarsinan.spiking.wire_gauge_seam import wire_gauge_established

    shape, flow, perceptrons, thetas = _unarmed_c5_form()
    assert not heterogeneous_domain_joins(flow.get_mapper_repr()), "fixture: no heterogeneous join (the C5 form)"
    _lif_step(_exact_config(lif_exact_qat=False))._establish_gauge(flow)
    assert wire_gauge_established(flow)
    assert _armed_add(flow).output_scale is not None
    program = _scm_program(flow, perceptrons, shape, thetas)
    x = samples(program, 16, 2)
    _, nf = _walk_counts(flow, perceptrons, x)
    _, scm = scm_window_counts(program, x, "windowed_lif")
    assert count_mismatches(nf, scm) == {pi: 0 for pi in scm}


# ---------------------------------------------------------------------------
# The twins refuse a graph the seam never established, by name
# ---------------------------------------------------------------------------


def test_the_d_hat_refuses_a_clone_whose_gauge_was_never_established():
    from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
    from mimarsinan.spiking.wire_gauge_seam import WireGaugeNotEstablishedError

    _, flow, _, _ = _unarmed_c5_form()
    assert issubclass(WireGaugeNotEstablishedError, ToolchainRefusal)
    with pytest.raises(WireGaugeNotEstablishedError, match="gauge"):
        deployed_lif_gauge_forward(copy.deepcopy(flow), _exact_config())


def _lif_tuner(config):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    tuner = object.__new__(LIFAdaptationTuner)
    tuner.pipeline = SimpleNamespace(config=config)
    tuner._cycle_accurate = True
    tuner._T = T
    tuner._per_hop_retiming = True
    tuner._phase_dither = False
    tuner._synchronized = False
    return tuner


def test_the_lif_finalize_twin_refuses_a_graph_whose_gauge_was_never_established():
    from mimarsinan.spiking.wire_gauge_seam import WireGaugeNotEstablishedError

    _, flow, _, _ = _unarmed_c5_form()
    with pytest.raises(WireGaugeNotEstablishedError, match="LIF finalize"):
        _lif_tuner(_exact_config(lif_exact_qat=False))._finalize_forward_for(flow)


def test_the_lif_finalize_twin_reads_an_established_graph():
    from mimarsinan.tuning.forward_install import ChipAlignedNFForward

    _, flow, _, _ = _unarmed_c5_form()
    cfg = _exact_config(lif_exact_qat=False)
    _lif_step(cfg)._establish_gauge(flow)
    assert isinstance(_lif_tuner(cfg)._finalize_forward_for(flow), ChipAlignedNFForward)


def test_a_graph_the_seam_would_not_change_needs_no_establishment():
    """Unit gauges everywhere: the seam arms nothing and moves no currency, so the twins read it as it is."""
    shape, flow, perceptrons, _ = lif_net("encoding_skip", 0, 1.0)
    with torch.no_grad():
        for p in perceptrons:
            p.activation_scale.fill_(1.0)
    x = torch.rand(4, *shape)
    with torch.no_grad():
        deployed_lif_gauge_forward(copy.deepcopy(flow), _exact_config())(x)
        _lif_tuner(_exact_config(lif_exact_qat=False))._finalize_forward_for(flow)(x)


# ---------------------------------------------------------------------------
# The route (b) members' graphs: S4's D-hat walk (D-10c-3), and S1's through the mixed concatenations (D-10c-2)
# ---------------------------------------------------------------------------


def _member(model_type: str, model_config: dict, shape=(3, 32, 32), classes=100):
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.converter import convert_torch_model

    builder = BUILDERS_REGISTRY[model_type]("cpu", shape, classes, {})
    torch.manual_seed(0)
    model = build_model(builder, model_config, encoding_placement="subsume").eval()
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, shape, classes, device="cpu", encoding_layer_placement="subsume")
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(1.0 + 0.37 * (i + 1)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow, shape


def _d_hat_runs(flow, shape):
    from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins

    cfg = _exact_config()
    install_lif_entry_input_quantizers(flow, cfg)
    assert not heterogeneous_domain_joins(flow.get_mapper_repr())
    gauge = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    torch.manual_seed(1)
    with torch.no_grad():
        out = gauge(torch.rand(2, *shape))
    assert torch.isfinite(out).all()


def test_a_pyramidnet_d_hat_walk_runs_after_the_seam():
    flow, shape = _member("cifar_pyramidnet", {"depth": 8, "alpha": 12}, shape=(3, 16, 16), classes=10)
    _d_hat_runs(flow, shape)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_s4_pyramidnet_d_hat_walk_runs_after_the_seam():
    """S4 (PyramidNet-110, alpha 84) at random init: RC4 stops here, ``NotImplementedError ... (ComputeOpMapper)``."""
    flow, shape = _member("cifar_pyramidnet", {"depth": 110, "alpha": 84})
    _d_hat_runs(flow, shape)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_s1_densenet_d_hat_walk_runs_after_the_seam():
    """S1 (DenseNet-BC-100, k 12) at random init: RC4 stops at ``(ConcatMapper)`` and its entry repair raised
    ``AttributeError``."""
    flow, shape = _member("cifar_densenet_bc", {"depth": 100, "growth_rate": 12, "compression": 0.5,
                                                "layer_order": "published"})
    _d_hat_runs(flow, shape)
