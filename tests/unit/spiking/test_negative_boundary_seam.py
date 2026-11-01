"""[U-SC2 A-2, D-USC-2] The negative-value boundary policy at the gauge seam.

Soft Core Mapping makes every negative ComputeOp -> core boundary encodable, but only there, so every walk before it
(the exact-QAT AQ D-hat, the LIF finalize twin, LIF Adaptation's retention metric) clamped a signed host chain at the
[0, 1] spike encode. S4's PyramidNet reads a signed BatchNorm at every core (units_k_bn1 -> units_k_conv1): on SD-10c's
cached S4 model the walk agreed with the model's own forward on 1 of 64 E01 images, departing at the first core
(final_units/usc/probes/out/s4_walk_layers.json), and the AQ D-hat read chance.

The seam now applies the policy (``NegativeBoundarySeam``, bound by the exact-QAT AQ install and LIF Adaptation's entry)
before any walk: each encoded, untrained signed boundary is lifted in its op's own function (sigma and its cover kappa,
the signed-seam capacity law), every consumer baked, so the plain forward, the walks and the deployed program read one
boundary, and Soft Core Mapping's later call finds nothing left to stamp there. The fixture is S4's family at a small
size (PyramidNet depth 8, alpha 12, 3x16x16) at random init with randomized BatchNorm statistics.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_mode
from mimarsinan.mapping.support.negative_boundary import apply_negative_boundary_policy
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.tuning.orchestration.negative_boundary_seam import NegativeBoundarySeam, lift_negative_boundaries

from unit.spiking.host_gather_forms import T, Program, count_mismatches, nf_window_counts, scm_sequence, scm_window_counts

SHAPE, CLASSES = (3, 16, 16), 10


def _pyramidnet(seed: int = 0, placement: str = "subsume"):
    """S4's family at a small size: every core (``units_k_conv1``) reads a host BatchNorm (``units_k_bn1``)."""
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.converter import convert_torch_model

    builder = BUILDERS_REGISTRY["cifar_pyramidnet"]("cpu", SHAPE, CLASSES, {})
    torch.manual_seed(seed)
    model = build_model(builder, {"depth": 8, "alpha": 12}, encoding_placement=placement).eval()
    gen = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.running_mean.uniform_(-0.3, 0.3, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
        model(torch.zeros(1, *SHAPE))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, SHAPE, CLASSES, device="cpu", encoding_layer_placement=placement)
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(1.5 + 0.2 * (i % 3)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow.eval()


def _inputs(seed: int, n: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return 2.0 * torch.randn(n, *SHAPE)


def _seam(ids: float = 2.0) -> NegativeBoundarySeam:
    return NegativeBoundarySeam(calibration_x=_inputs(7, 64), simulation_steps=T, input_data_scale=ids,
                                soma_law=DEFAULT_SOMA_LAW)


def _bn1_entries(flow) -> list:
    return [n for n in flow.get_mapper_repr().execution_order()
            if isinstance(n, ComputeOpMapper) and str(n.name).endswith("_bn1")]


class _Clamp(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = float(theta)

    def forward(self, z):
        return z.clamp(0.0, self.theta)


def _own_first_core(flow, x):
    """The model's own first core with continuous activations (clamp to [0, theta]): the limit the T-LIF walk reads."""
    ref = copy.deepcopy(flow)
    for p in ref.get_perceptrons():
        p.activation = _Clamp(p.activation_scale)
    out = {}
    hook = ref.get_perceptrons()[0].activation.register_forward_hook(lambda m, i, o: out.__setitem__("v", o.detach()))
    with torch.no_grad():
        ref(x)
    hook.remove()
    return out["v"].double()


def _walk_first_core(flow, x, steps: int):
    rec: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(flow, x, steps, retime=True, soma_law=DEFAULT_SOMA_LAW, node_value_recorder=rec)
    return rec[id(flow.get_perceptrons()[0])].double()


def _interior_error(walk, ref) -> float:
    """Mean |walk - ref| over the mean |ref|, on positions whose 3x3 kernel reads no padding (the border is a separate
    finding of this unit: a lifted boundary into a zero-padded convolution, compensated by one bias per channel)."""
    w = walk.reshape(ref.shape)[..., 1:-1, 1:-1]
    r = ref[..., 1:-1, 1:-1]
    return float((w - r).abs().mean() / r.abs().mean())


# ---------------------------------------------------------------------------
# The seam lifts every encoded signed boundary, and Soft Core Mapping finds nothing left to stamp
# ---------------------------------------------------------------------------


def test_the_seam_lifts_every_signed_core_entry_of_the_pyramidnet_form():
    flow = _pyramidnet()
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=_seam())
    entries = _bn1_entries(flow)
    assert len(entries) == 3, "fixture: one signed BatchNorm per core"
    assert all(op.output_value_offset is not None and float(op.output_value_offset) > 0 for op in entries)
    assert all(op.output_scale is not None and op.boundary_traffic_scale is not None for op in entries)


def test_soft_core_mapping_finds_the_lifted_boundaries_covered():
    """Soft Core Mapping's own policy (calibrated shift, its LIF calibration walk) after the seam stamps nothing: the walks
    and the deployment read the one boundary the seam installed."""
    flow = _pyramidnet()
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=_seam())
    result = apply_negative_boundary_policy(flow, _inputs(7, 64), T, shift_enabled=True,
                                            forward_fn=calibration_forward_for_mode("lif", soma_law=DEFAULT_SOMA_LAW))
    assert result.shifts == {} and result.subsumed == []


def test_the_seam_leaves_a_graph_without_a_signed_core_entry_unchanged():
    """The C5 form (a subsumed encoding at a non-unit theta feeding host residual adds) reads no signed core entry: the
    policy lifts nothing and arms nothing more than the seam does without it."""
    from unit.spiking.host_gather_forms import lif_net

    _, plain, _, _ = lif_net("encoding_skip", 0, 2.185)
    _, bound, _, _ = lif_net("encoding_skip", 0, 2.185)
    establish_deployed_wire_gauge(plain, input_data_scale=1.0)
    establish_deployed_wire_gauge(bound, input_data_scale=1.0, negative_boundary=NegativeBoundarySeam(
        calibration_x=torch.rand(16, 8) * 2.0, simulation_steps=T, input_data_scale=1.0, soma_law=DEFAULT_SOMA_LAW))
    for a, b in zip(plain.get_mapper_repr().execution_order(), bound.get_mapper_repr().execution_order()):
        assert getattr(b, "output_value_offset", None) is None
        assert (getattr(a, "output_scale", None) is None) == (getattr(b, "output_scale", None) is None)


# ---------------------------------------------------------------------------
# The walk reads the signed first core as the model's own forward does
# ---------------------------------------------------------------------------


def test_the_walk_reads_the_signed_first_core_as_the_model_forward_does():
    """S4's first-core profile on its family: after the seam the walk's first core approaches the model's own (continuous)
    first core, within the T-LIF quantization and the capacity law's tails; without the policy the clamp holds it away.
    Measured on this fixture (final_units/usc2/probes/out/pyramid_fixture_lift_q0.99.json, interior positions):
    lifted 0.221 at T 15 and 0.145 at T 63, unlifted 0.661 and 0.645; the bound below is the lifted T 63 reading with a
    third of margin, and the unlifted walk must stay above three times it."""
    x = _inputs(5, 16)
    lifted, clamped = _pyramidnet(), _pyramidnet()
    ref = _own_first_core(lifted, x)
    establish_deployed_wire_gauge(lifted, input_data_scale=2.0, negative_boundary=_seam())
    establish_deployed_wire_gauge(clamped, input_data_scale=2.0)
    err_lifted = _interior_error(_walk_first_core(lifted, x, 63), ref)
    err_clamped = _interior_error(_walk_first_core(clamped, x, 63), ref)
    assert err_lifted < 0.2
    assert err_clamped > 3 * err_lifted


def _border_error(walk, ref) -> float:
    """The same reading on the positions whose 3x3 kernel reads the zero padding."""
    w = walk.reshape(ref.shape)
    border = torch.ones(ref.shape[-2:], dtype=torch.bool)
    border[1:-1, 1:-1] = False
    return float((w[..., border] - ref[..., border]).abs().mean() / ref[..., border].abs().mean())


def test_the_walk_reads_the_signed_first_core_border_as_its_interior():
    """A lift into a zero-padded convolution: the lift adds sigma to every element the producer emits, the padding adds
    none, and the core's bake subtracts sigma times the WHOLE kernel at every position, so every position whose kernel
    reads the padding departed by sigma times the padded taps' weights (the departure the padding alone predicts to 5.5
    percent, final_units/usc2/probes/out/padding_border.json; 1.43 against the own forward at T 255, the interior 0.13).
    The lift now carries the padding of the convolutions that read it (the padded zeros are the op's own output, so the
    lift reaches them): the border reads within the interior's bound."""
    x = _inputs(5, 16)
    flow = _pyramidnet()
    ref = _own_first_core(flow, x)
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=_seam())
    walk = _walk_first_core(flow, x, 63)
    assert _border_error(walk, ref) < 0.2
    assert _interior_error(walk, ref) < 0.2


def test_the_lifted_first_core_converges_with_t():
    """The remaining departure of the lifted walk is quantization: it shrinks as T grows (the unlifted clamp does not)."""
    x = _inputs(5, 16)
    flow = _pyramidnet()
    ref = _own_first_core(flow, x)
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=_seam())
    assert _interior_error(_walk_first_core(flow, x, 63), ref) < 0.8 * _interior_error(_walk_first_core(flow, x, 15), ref)


# ---------------------------------------------------------------------------
# NF and SCM stay count-exact (atol 0) on the lifted program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ids", (1.0, 2.0))
def test_nf_and_scm_are_count_exact_on_the_lifted_pyramidnet(ids):
    """Every core's window counts, NF against the deployed flow, at atol 0, on the lifted program (the lift moves the
    first core's padding into units_0_bn1 and arms every bn1). At input data scale 2 U-SC's program disagreed at the
    first core (the absolute stem chain was encoded at the boundary table in the NF and at unit gauge in the deployed
    flow: 25,747 of 113,664 counts on this fixture, final_units/usc2/probes/out/pyramid_fixture_lift.json)."""
    flow = _pyramidnet()
    seam = _seam(ids)
    perceptrons = list(flow.get_perceptrons())
    repr_, ir, hybrid = scm_sequence(
        flow, perceptrons, wq=False, retimed=True,
        establish=lambda f: establish_deployed_wire_gauge(f, input_data_scale=ids, negative_boundary=seam))
    program = Program("pyramidnet", SHAPE, flow, repr_, ir, hybrid, perceptrons,
                      [float(p.activation_scale) for p in perceptrons], False)
    x = _inputs(5, 8)
    _, nf = nf_window_counts(program, x, "windowed_lif")
    _, scm = scm_window_counts(program, x, "windowed_lif")
    mism = count_mismatches(nf, scm)
    assert mism == {pi: 0 for pi in mism}


# ---------------------------------------------------------------------------
# The pipeline seams bind the policy
# ---------------------------------------------------------------------------


class _Trainer:
    def __init__(self, x):
        self._x = x

    def iter_validation_batches(self, n):
        for chunk in self._x.chunk(n):
            yield chunk, None


def _exact_config(**extra) -> dict:
    return {"lif_exact_qat": True, "spiking_mode": "lif", "firing_mode": "Default", "cycle_accurate_lif_forward": True,
            "lif_per_hop_retiming": True, "simulation_steps": T, "thresholding_mode": "<=",
            "encoding_layer_placement": "subsume", "input_data_scale": 2.0, **extra}


def test_the_aq_install_seam_applies_the_policy():
    from mimarsinan.tuning.orchestration.lif_exact_qat import install_lif_entry_input_quantizers

    flow = _pyramidnet()
    install_lif_entry_input_quantizers(flow, _exact_config(), trainer=_Trainer(_inputs(7, 64)))
    assert all(op.output_value_offset is not None for op in _bn1_entries(flow))


def test_the_aq_d_hat_reads_the_lifted_boundary():
    """The D-hat gauge's clone carries the lift, and reads the first core as the model does (bound as above)."""
    from mimarsinan.tuning.orchestration.lif_exact_qat import (
        deployed_lif_gauge_forward,
        install_lif_entry_input_quantizers,
    )

    x = _inputs(5, 16)
    flow = _pyramidnet()
    ref = _own_first_core(flow, x)
    cfg = _exact_config(simulation_steps=63)
    install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_inputs(7, 64)))
    clone = copy.deepcopy(flow)
    deployed_lif_gauge_forward(clone, cfg)
    assert _interior_error(_walk_first_core(clone, x, 63), ref) < 0.2


def test_the_lif_adaptation_entry_applies_the_policy():
    from types import SimpleNamespace

    from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import LIFAdaptationStep

    flow = _pyramidnet()
    step = object.__new__(LIFAdaptationStep)
    step.pipeline = SimpleNamespace(config=_exact_config(lif_exact_qat=False))
    step._establish_gauge(flow, trainer=lambda: _Trainer(_inputs(7, 64)))
    assert all(op.output_value_offset is not None for op in _bn1_entries(flow))


def test_offload_entries_keep_their_trained_clamp_and_the_opt_in_lift_stays_live():
    """[review M1] Under ``offload`` the install places a trained entry quantizer on every segment-entry core, so each
    signed boundary there is the QAT's trained clamp (Soft Core Mapping skips it), and lifting it is the opt-in
    ``lif_aq_negative_boundary``'s (off by default). The seam decides its scope with those carriers installed: it lifts
    nothing, and the opt-in still lifts. It decided before they existed and lifted units_0_bn1, units_1_bn1 and
    units_2_bn1 at the default (the review's probe, offload_new_offload.json)."""
    from mimarsinan.tuning.orchestration.lif_exact_qat import install_lif_entry_input_quantizers
    from mimarsinan.tuning.orchestration.signed_seam_install import ensure_offload_negative_boundary

    cfg = _exact_config(encoding_layer_placement="offload", device="cpu")
    flow = _pyramidnet(placement="offload")
    assert install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_inputs(7, 64))) == 3
    assert [op.name for op in _bn1_entries(flow) if op.output_value_offset is not None] == []
    ensure_offload_negative_boundary(flow, _Trainer(_inputs(7, 64)), {**cfg, "lif_aq_negative_boundary": True})
    assert any(op.output_value_offset is not None for op in _bn1_entries(flow))


class _TwoPaddings(torch.nn.Module):
    """A signed boundary no lift can carry: a host BatchNorm read by two convolutions that pad differently (3x3 padded
    by one, 1x1 unpadded), so its output cannot hold one padding for both, and the lift would reach a padded tap with
    no sigma (U-SC2's border finding)."""

    shape = (3, 8, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.bn = torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4)
        self.ca, self.cb = torch.nn.Conv2d(4, 4, 3, padding=1), torch.nn.Conv2d(4, 4, 1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 6)
        self.r0, self.ra, self.rb, self.r3 = torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()

    def forward(self, x):
        b = self.bn(self.r0(self.stem(x)))
        return self.r3(self.fc((self.ra(self.ca(b)) + self.rb(self.cb(b))).flatten(1)))


def _two_paddings():
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    net = _TwoPaddings().eval()
    gen = torch.Generator().manual_seed(100)
    with torch.no_grad():
        net.bn.running_mean.uniform_(0.2, 0.6, generator=gen)
        net.bn.running_var.uniform_(0.5, 1.5, generator=gen)
    flow = convert_torch_model(net, _TwoPaddings.shape, 6, device="cpu", encoding_layer_placement="subsume").eval()
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(1.3))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow


def test_a_boundary_no_lift_carries_gets_soft_core_mappings_own_policy_at_the_seam():
    """[review M2] A signed boundary the lift cannot carry kept D-USC-2: the walks before Soft Core Mapping read the
    [0, 1] clamp while Soft Core Mapping later deployed its own shift (on this fixture 0.41). The seam now applies Soft
    Core Mapping's own policy (its calibration walk, its shift and bake) to every boundary the lift leaves, so the walks
    read the boundary Soft Core Mapping deploys, and Soft Core Mapping's later call finds it covered."""
    flow = _two_paddings()
    x = 2.0 * torch.rand(64, *_TwoPaddings.shape, generator=torch.Generator().manual_seed(3))
    establish_deployed_wire_gauge(flow, input_data_scale=1.0, negative_boundary=NegativeBoundarySeam(
        calibration_x=x, simulation_steps=T, input_data_scale=1.0, soma_law=DEFAULT_SOMA_LAW))
    (bn,) = [n for n in flow.get_mapper_repr().execution_order()
             if isinstance(n, ComputeOpMapper) and n.name == "bn"]
    assert bn.output_value_offset is None, "fixture: no lift carries the BatchNorm read at two paddings"
    assert getattr(bn, "_negative_shift", None) is not None
    result = apply_negative_boundary_policy(flow, x, T, shift_enabled=True,
                                            forward_fn=calibration_forward_for_mode("lif", soma_law=DEFAULT_SOMA_LAW))
    assert result.shifts == {} and result.subsumed == []


def test_the_policy_preserves_the_run_random_state_and_training_flags():
    flow = _pyramidnet()
    establish_deployed_wire_gauge(flow, input_data_scale=2.0)
    flow.train()
    calibration = _inputs(7, 64)
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    assert lift_negative_boundaries(flow, calibration, simulation_steps=T, input_data_scale=2.0,
                                    soma_law=DEFAULT_SOMA_LAW) == []
    assert torch.equal(torch.rand(4), expected)
    assert all(m.training for m in flow.modules())
    assert all(op.output_value_offset is not None for op in _bn1_entries(flow))


def test_the_policy_calibration_walks_bounded_chunks(monkeypatch):
    """S4's calibration batch (Soft Core Mapping's two validation batches, 2 x 1,024 E01 images) walked at once held
    every host value of PyramidNet-110 and ran the A-3 S4 session past its 28G class (peak 29.6 GB at the AQ install
    seam; the seam alone on the cached S4 model: 29.56 GB at once, 2.95 GB in chunks of 128 images,
    final_units/usc2/probes/out/seam_memory_chunk*.time). The seam's lift and Soft Core Mapping's minima walk at most
    ``CALIBRATION_CHUNK`` images at once, and the minima equal one pass's."""
    import mimarsinan.tuning.orchestration.negative_boundary_seam as seam_module
    from mimarsinan.mapping.support.negative_boundary import calibrated_compute_op_minima

    CALIBRATION_CHUNK = seam_module.CALIBRATION_CHUNK

    flow = _pyramidnet()
    establish_deployed_wire_gauge(flow, input_data_scale=2.0)
    x = _inputs(7, 3 * CALIBRATION_CHUNK + 5)
    forward = calibration_forward_for_mode("lif", soma_law=DEFAULT_SOMA_LAW)
    sizes: list = []

    def spy(model, xb, steps, **kw):
        sizes.append(int(xb.shape[0]))
        return forward(model, xb, steps, **kw)

    chunked = calibrated_compute_op_minima(flow, x, T, forward_fn=spy)
    once: dict = {}
    with torch.no_grad():
        forward(flow, x, T, compute_min_recorder=once)
    assert sizes and max(sizes) <= CALIBRATION_CHUNK and sum(sizes) == x.shape[0]
    assert set(chunked) == set(once)
    assert all(torch.allclose(chunked[op], once[op], rtol=1e-5, atol=1e-6) for op in once)

    walked: list = []

    class _Driver(seam_module.SegmentForwardDriver):
        def __call__(self, xb, **kw):
            walked.append(int(xb.shape[0]))
            return super().__call__(xb, **kw)

    monkeypatch.setattr(seam_module, "SegmentForwardDriver", _Driver)
    lift_negative_boundaries(flow, x, simulation_steps=T, input_data_scale=2.0, soma_law=DEFAULT_SOMA_LAW)
    assert walked and max(walked) <= CALIBRATION_CHUNK and sum(walked) == x.shape[0]


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_the_s4_member_lifts_every_core_entry():
    """S4 (PyramidNet-110, alpha 84) at random init with randomized BatchNorm statistics: every core reads a signed
    BatchNorm, and the seam lifts all 54 (SD-10c's cached S4 model: 54 negative encoded boundaries,
    final_units/usc2/probes/out/s4_boundaries_base.json)."""
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.converter import convert_torch_model

    shape = (3, 32, 32)
    builder = BUILDERS_REGISTRY["cifar_pyramidnet"]("cpu", shape, 100, {})
    torch.manual_seed(0)
    model = build_model(builder, {"depth": 110, "alpha": 84}, encoding_placement="subsume").eval()
    gen = torch.Generator().manual_seed(100)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.3, 0.3, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
        model(torch.zeros(1, *shape))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, shape, 100, device="cpu", encoding_layer_placement="subsume")
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(1.7))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
    torch.manual_seed(7)
    seam = NegativeBoundarySeam(calibration_x=2.0 * torch.randn(16, *shape), simulation_steps=T, input_data_scale=2.0,
                                soma_law=DEFAULT_SOMA_LAW)
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=seam)
    entries = _bn1_entries(flow)
    assert len(entries) == 54
    assert all(op.output_value_offset is not None for op in entries)
