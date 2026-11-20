"""[U-LIF] The synchronized LIF step lossless at entry: the exact-QAT network is the deployed composition, and the
windowed LIF swap reads it.

Three defects of the step (lif_diag's diagnosis, its check and the U-LIF ruling): D1, a host-side (subsumed) encoder's
entry quantizer clamps a signed input to [0, s] (a host-side encoder takes no input grid); D2, the exact-QAT network
omits the deployed host-to-chip entry encode on segment entries a host value feeds, whose gauge at a residual join is
the mean of its sources' scales (the encode on every host-fed entry, at a gauge covering the calibration quantile of the
entry values); P1, the tuner's threshold moves outrun the per-cycle charge a neuron can emit at one spike per cycle (no
move takes a threshold below the per-cycle charge it was installed to carry, the half-step bias placing the staircase).
Read on the C5 form (a subsumed encoding, a core, a host linear, a residual add and its ReLU, a core reading the join:
one residual block of the rows' ResNet) with the toolchain's own installers, its AQ decorator at rate 1 (the quantized
network), its synchronized and windowed chip-aligned walks and the AQ gate's own D-hat instrument.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.support.bias_compensation import apply_lif_half_step_bias_compensation
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer, LeakyGradReLU
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.wire_gauge_seam import require_established_wire_gauge
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    deployed_lif_gauge_forward,
    install_lif_entry_input_quantizers,
)
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

from unit.spiking.host_gather_forms import N_OUT, T, lif_net

#: The rows' declared input range after normalization (C5's CIFAR-100 image: min -1.897, max 2.024).
INPUT_MIN, INPUT_MAX = -1.897, 2.024
#: The entry gauge's calibration tail the tests hold the install to (its 0.999 quantile).
ENTRY_TAIL = 1e-3
#: The tuner's in-loop threshold descent the tuned C5 row measured (thetas fell two- to threefold).
TUNED_DESCENT = 3.0
N_CAL, N_READ = 2048, 2048
#: The weight gain of the general (mixed-sign) forms: residual-join sums well above their sources' mean gauge.
GENERAL_GAIN = 3.0
#: The stacked residual form's weight gain.
STACK_GAIN = 1.5


def _cfg(placement: str = "subsume", **extra) -> dict:
    return {"lif_exact_qat": True, "spiking_mode": "lif", "firing_mode": "Default", "cycle_accurate_lif_forward": True,
            "lif_per_hop_retiming": True, "simulation_steps": T, "thresholding_mode": "<=",
            "encoding_layer_placement": placement, "input_data_scale": INPUT_MAX, **extra}


class _Trainer:
    """The calibration batches the AQ install reads (two validation batches, as Soft Core Mapping's)."""

    def __init__(self, x):
        self._x = x

    def iter_validation_batches(self, n):
        for chunk in self._x.chunk(n):
            yield chunk, None


def _signed(n: int, seed: int) -> torch.Tensor:
    """Normalized images: signed, inside the declared range."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 8, generator=g).clamp(INPUT_MIN, INPUT_MAX)


def _unsigned(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return INPUT_MAX * torch.rand(n, 8, generator=g)


def _c5_form(seed: int = 0, *, gain: float = 2.0, nonnegative: bool = False, placement: str = "subsume"):
    _, flow, perceptrons, _ = lif_net("encoding_skip", seed, 2.185, gain, placement=placement)
    if nonnegative:
        with torch.no_grad():
            for m in flow.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.abs_()
                    if m.bias is not None:
                        m.bias.abs_()
    return flow, perceptrons


class _Chain(nn.Module):
    """A subsumed encoding and two cores, perceptron to perceptron: no host value feeds a core (D1 alone)."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2 = nn.Linear(8, 12), nn.Linear(12, 12), nn.Linear(12, N_OUT)
        self.r0, self.r1, self.r2 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.r2(self.fc2(self.r1(self.fc1(self.r0(self.stem(x.flatten(1)))))))


class _ResidualStack(nn.Module):
    """C5's residual blocks stacked: a subsumed encoding, then per block a core, a host linear, the add with the skip
    and its ReLU, the next block's core reading the join; a core head."""

    shape = (8,)

    def __init__(self, depth: int = 4, width: int = 8):
        super().__init__()
        self.stem, self.r0 = nn.Linear(8, width), nn.ReLU()
        self.cores = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.hosts = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.core_relus = nn.ModuleList(nn.ReLU() for _ in range(depth))
        self.join_relus = nn.ModuleList(nn.ReLU() for _ in range(depth))
        self.head, self.rh = nn.Linear(width, N_OUT), nn.ReLU()

    def forward(self, x):
        h = self.r0(self.stem(x.flatten(1)))
        for core, host, rc, rj in zip(self.cores, self.hosts, self.core_relus, self.join_relus):
            h = rj(h + host(rc(core(h))))
        return self.rh(self.head(h))


def _chain_form(seed: int = 0, *, gain: float = 2.0, module=None):
    """``_Chain`` (or ``module``) converted as ``lif_net`` converts its forms (seeded weights times ``gain``, positive
    biases, a LIF node at each perceptron's theta)."""
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(seed)
    net = _Chain() if module is None else module()
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(gain)
                m.bias.copy_(0.05 + 0.25 * torch.rand(m.bias.shape, generator=gen))
    flow = convert_torch_model(net.eval(), net.shape, N_OUT, device="cpu", encoding_layer_placement="subsume").eval()
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(1.0))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow, list(flow.get_perceptrons())


def _producer(flow, perceptron):
    """The first non-structural node up the perceptron's source chain."""
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    for node in repr_._exec_order:
        if perceptron_of(node) is perceptron:
            src = node.source_mapper
            while src is not None and not isinstance(src, (InputMapper, ComputeOpMapper)) and perceptron_of(src) is None:
                src = src.source_mapper
            return src
    raise AssertionError("perceptron not in the graph")


def _host_fed(flow) -> list:
    return [p for p in flow.get_perceptrons()
            if not getattr(p, "is_encoding_layer", False) and isinstance(_producer(flow, p), ComputeOpMapper)]


def _carries_entry_encode(perceptron) -> bool:
    return any(isinstance(m, ChipInputQuantizer) for m in perceptron.input_activation.modules())


def _theta(perceptron) -> torch.Tensor:
    return torch.as_tensor(perceptron.activation_scale).detach().float()


def _rate_zero(flow):
    """The Activation Shifting network as the AQ install meets it (rate 0: the float chip ReLU, no staircase)."""
    for p in flow.get_perceptrons():
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
    return flow


def _aq_network(flow, cfg):
    """The quantized network: the AQ decorator at rate 1 (the deployed LIF count staircase, theta in-loop)."""
    manager = AdaptationManager()
    manager.quantization_rate = 1.0
    for p in flow.get_perceptrons():
        p.base_activation = LeakyGradReLU()
        manager.update_activation(cfg, p)
    return flow.eval()


def _lif_swap(flow):
    """The LIF swap: a fresh LIF node at each perceptron's own theta, as the finalize and the D-hat gauge install it."""
    swap = copy.deepcopy(flow)
    for p in swap.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    require_established_wire_gauge(swap, input_data_scale=INPUT_MAX, reader="the U-LIF test walk")
    return swap.eval()


def _aq_counts(aq, x) -> list:
    """Each perceptron's window count in the quantized network's own forward, by perceptron index."""
    out, hooks = {}, []
    for k, p in enumerate(aq.get_perceptrons()):
        hooks.append(p.activation.register_forward_hook(
            lambda m, a, o, k=k, p=p: out.__setitem__(k, torch.round(o.detach() * T / _theta(p)))))
    with torch.no_grad():
        aq(x)
    for h in hooks:
        h.remove()
    return [out[k] for k in range(len(out))]


def _walk_counts(swap, x, *, synchronized: bool) -> list:
    """Each perceptron's decoded window count in the chip-aligned walk (re-timed), by perceptron index."""
    rec: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(swap, x, T, soma_law=DEFAULT_SOMA_LAW, retime=True, synchronized=synchronized,
                                     node_value_recorder=rec)
    return [torch.round(rec[id(p)] * T / _theta(p)) for p in swap.get_perceptrons()]


def _entry_values(flow, perceptron, x) -> torch.Tensor:
    seen = []
    handle = perceptron.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
    with torch.no_grad():
        flow(x)
    handle.remove()
    return seen[0]


def _labels(teacher, x, seed: int) -> torch.Tensor:
    """The float teacher's classes with a seeded 30 percent label noise (reads strictly between chance and one)."""
    with torch.no_grad():
        y = teacher(x).argmax(dim=1)
    g = torch.Generator().manual_seed(seed)
    flip = torch.rand(y.shape, generator=g) < 0.3
    return torch.where(flip, torch.randint(0, int(teacher(x[:1]).shape[1]), y.shape, generator=g), y)


def _accuracy(logits, y) -> float:
    return float((logits.argmax(dim=1) == y).double().mean())


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 1e-12) / n)


def _aq_install(flow, cfg, x_cal):
    """The Activation Quantization step's LIF exact-QAT install on ``flow`` (theta in-loop, the half-step fold, the
    entry encode and its gauge), through the tuner's own install seam."""
    tuner = SimpleNamespace(model=flow, pipeline=SimpleNamespace(config=cfg, reporter=None), trainer=_Trainer(x_cal))
    ActivationQuantizationTuner._install_lif_exact_qat(tuner)
    return tuner


def _tuner_moves(tuner, descent: float = TUNED_DESCENT) -> None:
    """The in-loop threshold moves of the tuned row (every trainable theta falls ``descent``-fold), then the AQ step's
    own exit seam (the theta pin before the LIF finalize builds its nodes)."""
    with torch.no_grad():
        for p in tuner.model.get_perceptrons():
            theta = p.activation_scale
            if isinstance(theta, torch.Tensor) and theta.requires_grad:
                theta.div_(descent)
    ActivationQuantizationTuner._pin_lif_theta_positivity_floor(tuner)


def _analysed_thetas(flow, x) -> None:
    """Each perceptron's theta as the Activation Analysis step sets it: the count quantile of its positive activations
    in the float network (``scale_from_activations``, the step's default quantile)."""
    from mimarsinan.transformations.activation_scale_policy import scale_from_activations

    reference = _rate_zero(copy.deepcopy(flow))
    acts, hooks = {}, []
    for k, p in enumerate(reference.get_perceptrons()):
        hooks.append(p.activation.register_forward_hook(lambda m, a, o, k=k: acts.__setitem__(k, o.detach())))
    with torch.no_grad():
        reference(x)
    for h in hooks:
        h.remove()
    for k, p in enumerate(flow.get_perceptrons()):
        with torch.no_grad():
            p.activation_scale.copy_(torch.tensor(scale_from_activations(acts[k].flatten())))


def _one_spike_per_cycle(perceptron, gauge: float) -> None:
    """Scale a core's non-negative weights so its per-cycle charge (every input spiking in one cycle at ``gauge``, the
    bias and any half-step fold included) stays within one spike per cycle at its theta."""
    with torch.no_grad():
        w = perceptron.layer.weight
        b = perceptron.layer.bias if perceptron.layer.bias is not None else torch.zeros(w.shape[0])
        theta = _theta(perceptron).reshape(-1).expand(w.shape[0])
        per_cycle = gauge * w.reshape(w.shape[0], -1).sum(dim=1)
        room = (0.95 * theta - b).clamp(min=0.0) / per_cycle.clamp(min=1e-9)
        w.mul_(room.clamp(max=1.0).reshape(-1, *([1] * (w.dim() - 1))))


def _entry_gauge(perceptron) -> float:
    return float(torch.as_tensor(perceptron.input_activation_scale).detach().reshape(-1).max())


# ---- D1: a host-side encoder takes no input grid ---------------------------------------------------------------------

def test_t1_a_host_side_encoder_reads_every_value_of_the_declared_range():
    flow, perceptrons = _c5_form()
    install_lif_entry_input_quantizers(flow, _cfg(), trainer=_Trainer(_signed(N_CAL, 1)))
    encoder = perceptrons[0]
    assert encoder.is_encoding_layer
    span = torch.linspace(INPUT_MIN, INPUT_MAX, 4001)
    with torch.no_grad():
        read = encoder.input_activation(span)
    torch.testing.assert_close(read, span, atol=0.0, rtol=0.0)


def test_t1_the_lif_step_wraps_no_grid_on_a_host_side_encoder():
    flow, perceptrons = _c5_form()
    encoder = perceptrons[0]
    LIFAdaptationTuner._wrap_encoding_input(SimpleNamespace(_T=T), encoder)
    assert not _carries_entry_encode(encoder)


def test_t2_the_aq_install_at_rate_0_reads_the_shifted_network_within_one_se():
    cfg = _cfg()
    flow, _ = _chain_form(gain=4.0)
    _analysed_thetas(flow, _signed(N_CAL, 1))
    assert not _host_fed(flow)
    # The reference carries the install's own half-step fold (P-L6, the declared bias move of the install), so the
    # read isolates the encoder's entry.
    shifted = _rate_zero(copy.deepcopy(flow))
    apply_lif_half_step_bias_compensation(shifted, T)
    x, y_seed = _signed(N_READ, 7), 11
    y = _labels(shifted, x, y_seed)
    _aq_install(flow, cfg, _signed(N_CAL, 1))
    installed = _rate_zero(flow)
    with torch.no_grad():
        before, after = _accuracy(shifted(x), y), _accuracy(installed(x), y)
    assert abs(after - before) <= _se(before, N_READ), (before, after)


# ---- D2: the deployed entry encode on every host-fed entry, at a calibrated gauge -------------------------------------

def test_t3_the_quantized_network_is_the_synchronized_walk_count_for_count():
    cfg = _cfg()
    flow, perceptrons = _c5_form()
    install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_unsigned(N_CAL, 1)))
    x = _unsigned(N_READ, 2)
    aq = _aq_counts(_aq_network(copy.deepcopy(flow), cfg), x)
    sync = _walk_counts(_lif_swap(flow), x, synchronized=True)
    differ = {k: int((a != b).sum()) for k, (a, b) in enumerate(zip(aq, sync))}
    assert all(v == 0 for v in differ.values()), differ


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_t4_the_aq_gate_dhat_reads_the_quantized_network_within_one_se(seed, window_firing_declared):
    """The AQ gate's own D-hat instrument at entry, two-window so the read isolates the entry encode (P1 is t5's):
    stacked residual blocks, each join's sum above its sources' mean gauge. [U-LIF2] the two-window discipline is
    the declared window firing integrate_then_emit (the removed internal key carried it before)."""
    window_firing_declared()
    cfg = _cfg(lif_window_firing="integrate_then_emit")
    flow, _ = _chain_form(seed, gain=STACK_GAIN, module=_ResidualStack)
    _analysed_thetas(flow, _signed(N_CAL, 1))
    teacher = _rate_zero(copy.deepcopy(flow))
    x = _signed(N_READ, 3)
    y = _labels(teacher, x, 13)
    _aq_install(flow, cfg, _signed(N_CAL, 1))
    aq = _aq_network(copy.deepcopy(flow), cfg)
    dhat = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        post_logits, full_logits = aq(x), dhat(x)
    post, full = _accuracy(post_logits, y), _accuracy(full_logits, y)
    agreement = float((post_logits.argmax(dim=1) == full_logits.argmax(dim=1)).double().mean())
    assert agreement == 1.0 and abs(full - post) <= _se(post, N_READ), (post, full, agreement)


def test_t5_the_windowed_lif_swap_reads_the_quantized_network_within_one_se_after_the_tuner():
    cfg = _cfg()
    flow, _ = _c5_form(gain=GENERAL_GAIN)
    _analysed_thetas(flow, _signed(N_CAL, 1))
    teacher = _rate_zero(copy.deepcopy(flow))
    x = _signed(N_READ, 4)
    y = _labels(teacher, x, 17)
    tuner = _aq_install(flow, cfg, _signed(N_CAL, 1))
    _tuner_moves(tuner)
    aq = _aq_network(copy.deepcopy(flow), cfg)
    swap = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        post, full = _accuracy(aq(x), y), _accuracy(swap(x), y)
    assert abs(full - post) <= _se(post, N_READ), (post, full)


def test_t6_the_installer_covers_every_host_fed_segment_entry_and_no_other():
    cfg = _cfg()
    flow, perceptrons = _c5_form()
    host_fed = _host_fed(flow)
    assert host_fed, "the C5 form has a core reading the residual join"
    install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_unsigned(N_CAL, 1)))
    assert {id(p) for p in perceptrons if _carries_entry_encode(p)} == {id(p) for p in host_fed}
    assert install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_unsigned(N_CAL, 1))) == 0
    for p in host_fed:
        assert p.input_activation.activation_scale is p.input_activation_scale


def test_t6_offload_keeps_its_entry_encode_on_every_segment_entry():
    cfg = _cfg("offload")
    flow, perceptrons = _c5_form(placement="offload")
    install_lif_entry_input_quantizers(flow, cfg)
    for p in perceptrons:
        fed_by_host = isinstance(_producer(flow, p), (InputMapper, ComputeOpMapper))
        assert _carries_entry_encode(p) == fed_by_host


def test_t7_each_host_fed_entry_gauge_covers_the_calibration_quantile():
    cfg = _cfg()
    flow, _ = _c5_form()
    install_lif_entry_input_quantizers(flow, cfg, trainer=_Trainer(_unsigned(N_CAL, 1)))
    for draw in (1, 5):
        x = _unsigned(N_CAL, draw)
        for p in _host_fed(flow):
            gauge = _entry_gauge(p)
            above = float((_entry_values(flow, p, x) > gauge * (1 + 1e-6)).double().mean())
            assert above <= 3 * ENTRY_TAIL, (draw, gauge, above)


def test_t7_a_residual_join_entry_is_not_gauged_at_the_mean_of_its_sources():
    cfg = _cfg()
    flow, _ = _c5_form()
    install_lif_entry_input_quantizers(flow, cfg)  # no calibration batch: the propagated gauges
    mean_gauge = {id(p): _entry_gauge(p) for p in _host_fed(flow)}
    calibrated, _ = _c5_form()
    install_lif_entry_input_quantizers(calibrated, cfg, trainer=_Trainer(_unsigned(N_CAL, 1)))
    x = _unsigned(N_CAL, 1)
    for p, q in zip(_host_fed(flow), _host_fed(calibrated)):
        entry = _entry_values(calibrated, q, x)
        assert float((entry > mean_gauge[id(p)]).double().mean()) > ENTRY_TAIL, "the form's join sum exceeds the mean"
        assert _entry_gauge(q) > mean_gauge[id(p)]


# ---- P1: per-cycle firing within one spike per cycle after the tuner --------------------------------------------------

def test_the_tuner_moves_no_threshold_below_its_installed_per_cycle_charge():
    cfg = _cfg()
    flow, perceptrons = _c5_form()
    _analysed_thetas(flow, _signed(N_CAL, 1))
    tuner = _aq_install(flow, cfg, _signed(N_CAL, 1))
    installed = {id(p): _theta(p).clone() for p in perceptrons}
    _tuner_moves(tuner)
    for p in perceptrons:
        assert bool((_theta(p) >= installed[id(p)] - 1e-7).all()), (float(_theta(p).min()), installed[id(p)])


def test_the_aq_gate_dhat_reads_the_bounded_threshold_in_the_loop():
    """Mid-loop (thresholds moved, before the exit pin) the gate's D-hat reads the theta the quantized network reads."""
    cfg = _cfg()
    flow, _ = _c5_form(gain=GENERAL_GAIN)
    _analysed_thetas(flow, _signed(N_CAL, 1))
    teacher = _rate_zero(copy.deepcopy(flow))
    x = _signed(N_READ, 4)
    y = _labels(teacher, x, 17)
    _aq_install(flow, cfg, _signed(N_CAL, 1))
    with torch.no_grad():
        for p in flow.get_perceptrons():
            if isinstance(p.activation_scale, torch.Tensor) and p.activation_scale.requires_grad:
                p.activation_scale.div_(TUNED_DESCENT)
    aq = _aq_network(copy.deepcopy(flow), cfg)
    dhat = deployed_lif_gauge_forward(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        post, full = _accuracy(aq(x), y), _accuracy(dhat(x), y)
    assert abs(full - post) <= _se(post, N_READ), (post, full)


def test_the_lif_step_twin_and_its_endpoint_run_no_threshold_below_the_bound(monkeypatch):
    """The LIF step's finalize twin and its endpoint stage end every theta at its per-cycle charge bound."""
    import mimarsinan.tuning.tuners.lif_adaptation_tuner as lif_tuner

    cfg = _cfg()
    flow, perceptrons = _c5_form()
    _analysed_thetas(flow, _signed(N_CAL, 1))
    _aq_install(flow, cfg, _signed(N_CAL, 1))
    installed = {id(p): _theta(p).clone() for p in perceptrons}

    def descend(model=None):
        with torch.no_grad():
            for p in flow.get_perceptrons():
                if isinstance(p.activation_scale, torch.Tensor) and p.activation_scale.requires_grad:
                    p.activation_scale.div_(TUNED_DESCENT)

    def held(p):
        return bool((_theta(p) >= installed[id(p)] - 1e-7).all())

    descend()
    LIFAdaptationTuner._finalize_forward_for(SimpleNamespace(_cycle_accurate=False), flow)
    assert all(held(p) for p in perceptrons)
    monkeypatch.setattr(lif_tuner, "run_endpoint_recovery", lambda tuner, base_steps: descend())
    stage = SimpleNamespace(_adaptation_plan=SimpleNamespace(distmatch=False), _fixed_ladder_policy=True,
                            _endpoint_recovery_steps=0, model=flow)
    LIFAdaptationTuner._post_stabilization_hook(stage)
    assert all(held(p) for p in perceptrons)


def test_the_bound_is_the_first_install_s_and_survives_a_second_install():
    from mimarsinan.tuning.orchestration.lif_exact_qat import lif_charge_bound, record_lif_charge_bounds

    flow, perceptrons = _c5_form()
    _analysed_thetas(flow, _signed(N_CAL, 1))
    assert record_lif_charge_bounds(flow) == 2
    installed = {id(p): _theta(p).clone() for p in perceptrons}
    with torch.no_grad():
        for p in perceptrons:
            p.activation_scale.div_(TUNED_DESCENT)
    assert record_lif_charge_bounds(flow) == 0
    for p in perceptrons:
        if not p.is_encoding_layer:
            torch.testing.assert_close(lif_charge_bound(p), installed[id(p)], atol=0.0, rtol=0.0)


def test_the_quantized_network_never_reads_a_threshold_below_the_bound_in_the_loop():
    cfg = _cfg()
    flow, perceptrons = _c5_form()
    _aq_install(flow, cfg, _unsigned(N_CAL, 1))
    x = _unsigned(N_READ, 6)
    aq = _aq_network(copy.deepcopy(flow), cfg)
    with torch.no_grad():
        at_install = aq(x)
        for p in aq.get_perceptrons():
            if isinstance(p.activation_scale, torch.Tensor) and p.activation_scale.requires_grad:
                p.activation_scale.div_(TUNED_DESCENT)
        in_loop = aq(x)
    torch.testing.assert_close(in_loop, at_install, atol=0.0, rtol=0.0)


def test_count_identity_where_per_cycle_firing_is_exact_by_construction():
    """Non-negative charges within one spike per cycle: the windowed LIF swap's per-neuron window count IS the
    quantized network's level (the half-step fold included), atol 0, after the tuner's moves."""
    cfg = _cfg()
    flow, perceptrons = _c5_form(nonnegative=True)
    _analysed_thetas(flow, _unsigned(N_CAL, 1))
    encoder, first, join_core = perceptrons
    tuner = _aq_install(flow, cfg, _unsigned(N_CAL, 1))
    _one_spike_per_cycle(first, float(_theta(encoder)))
    _one_spike_per_cycle(join_core, _entry_gauge(join_core))
    _tuner_moves(tuner)
    x = _unsigned(N_READ, 8)
    aq = _aq_counts(_aq_network(copy.deepcopy(flow), cfg), x)
    windowed = _walk_counts(_lif_swap(flow), x, synchronized=False)
    differ = {k: int((a != b).sum()) for k, (a, b) in enumerate(zip(aq, windowed))}
    assert all(v == 0 for v in differ.values()), differ


#: The measured departure bound of the general case (mixed-sign charges, the per-cycle charge at its installed
#: quantile): the share of each per-cycle hop's neuron windows whose windowed count departs from the quantized level.
DEPARTURE_BOUND = 0.06


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_general_case_departure_is_bounded_after_the_tuner(seed):
    cfg = _cfg()
    flow, perceptrons = _c5_form(seed)
    _analysed_thetas(flow, _signed(N_CAL, 1))
    tuner = _aq_install(flow, cfg, _signed(N_CAL, 1))
    _tuner_moves(tuner)
    x = _signed(N_READ, 9)
    aq = _aq_counts(_aq_network(copy.deepcopy(flow), cfg), x)
    windowed = _walk_counts(_lif_swap(flow), x, synchronized=False)
    departed = {k: float((a != b).double().mean()) for k, (a, b, p) in enumerate(zip(aq, windowed, perceptrons))
                if not getattr(p, "is_encoding_layer", False)}
    assert max(departed.values()) <= DEPARTURE_BOUND, departed
