"""[U-TGAUGE2] Every comparison of two outputs is made in one unit (the U-TGAUGE verify's m-1, by one rule).

U-TGAUGE made the walk return its output in the value domain: it decodes its output node by the inverse of its own
encode there. The deployed program's executors keep their count currency: each output element is its producer's held
state (``value / gauge``) times the window. A gate or a readout that ranked that read beside the walk's compared two
units. At one scalar gauge the classes agree; at a per-channel output gauge they do not (the verify's per-channel host
output: decision agreement 0.875). The rule: every reader of an output as a value decodes it by the inverse of its
producer's encode through one decode, ``decode_held`` (the held value times its gauge per element). The walk applies it
at its own gauge; a deployed readout at its program's gauge (``program_output_gauge``, read from the program alone).

The class, swept (one test per site): the two decision gates (the cascaded NF against the identity executor, the torch
twin against the deployed flow), the deployed census readout, the membrane diagnostic's class flips, the chip
simulators' readouts (the hybrid and flat nevresim runners and the nevresim driver, the Loihi runner), the ODIN board's
readout (it ranks raw counts on chip, so it refuses a readout at more than one gauge), the pretrained bridge's deployed
logits, the timing family's distillation teacher, and the three readers the verify's m-3 names: the learning-rate
explorer's loss ranking, the gradient-alignment ledger's pair in the activation step, and the prefix ramp's plain loss.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import unit.spiking.host_gather_forms as hg

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.pipelining.core import nf_scm_parity
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.torch_mapping.converter import convert_torch_model

T = hg.T
INPUT_MAX = 2.0
#: The verify's per-channel host output: a host normalization reading a core at six thresholds (its ready red).
PER_CHANNEL = (0.5, 0.9, 1.4, 2.0, 2.7, 3.6)
SEMANTICS = tuple(hg.SEMANTICS)
RTOL, ATOL = 1e-5, 1e-6


def _od():
    """The unit's decode module, imported where a test reads it (the gates' tests read only what existed before it)."""
    from mimarsinan.chip_simulation.hybrid_run import output_decode

    return output_decode


class HostClassifier(nn.Module):
    """A subsumed encoding, two cores, a host linear classifier: a scalar output gauge."""

    shape = (10,)

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c, self.head = nn.Linear(10, 7), nn.Linear(7, 9), nn.Linear(9, 8), nn.Linear(8, hg.N_OUT)
        self.ra, self.rb, self.rc = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.head(self.rc(self.c(self.rb(self.b(self.ra(self.a(x.flatten(1))))))))


class HostNormOut(nn.Module):
    """A subsumed encoding, a core, a host normalization as the output node: per-channel at per-channel thresholds."""

    shape = (10,)

    def __init__(self):
        super().__init__()
        self.a, self.b, self.bn = nn.Linear(10, 7), nn.Linear(7, hg.N_OUT), nn.BatchNorm1d(hg.N_OUT)
        self.ra, self.rb = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.bn(self.rb(self.b(self.ra(self.a(x.flatten(1))))))


class CoreOut(nn.Module):
    """A subsumed encoding and a core as the output node."""

    shape = (10,)

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(10, 7), nn.Linear(7, hg.N_OUT)
        self.ra, self.rb = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.rb(self.b(self.ra(self.a(x.flatten(1)))))


class CoreLayoutOut(nn.Module):
    """A convolutional encoding and core, a flatten after it as the output node."""

    shape = (3, 4, 4)

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(3, 2, 3, padding=1), nn.Conv2d(2, 3, 3, padding=1)
        self.ra, self.rb = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.rb(self.b(self.ra(self.a(x)))).flatten(1)


class MixedOut(nn.Module):
    """An absolute host chain concatenated with a core output as the output node (a MIXED gather)."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1 = nn.Linear(8, 6), nn.Linear(6, 6)
        self.r1 = nn.ReLU()

    def forward(self, x):
        s = self.stem(x.flatten(1))
        return torch.cat([s, self.r1(self.fc1(s))], dim=1)


FORMS = {
    "host_scalar": (HostClassifier, (INPUT_MAX, 1.7, 2.6, 1.9)),
    "host_per_channel": (HostNormOut, (INPUT_MAX, PER_CHANNEL)),
    "host_norm_scalar": (HostNormOut, (INPUT_MAX, 1.8)),
    "core": (CoreOut, (INPUT_MAX, 1.3)),
    "core_per_channel": (CoreOut, (INPUT_MAX, PER_CHANNEL)),
    "core_layout": (CoreLayoutOut, (INPUT_MAX, 1.6)),
    "mixed": (MixedOut, (1.7,)),
}


def _program(form: str, semantics: str) -> hg.Program:
    """The form at seeded weights, LIF nodes at its thresholds, through the Soft Core Mapping sequence (the wire gauge
    established by the seam); the verify's own construction for the per-channel case."""
    cls, thetas = FORMS[form]
    torch.manual_seed(11)
    net = cls()
    gen = torch.Generator().manual_seed(12)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.copy_(torch.rand(m.weight.shape, generator=gen) * 0.6 - 0.1)
                m.bias.copy_(0.05 + 0.2 * torch.rand(m.bias.shape, generator=gen))
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.copy_(torch.linspace(-0.2, 0.3, m.num_features))
                m.running_var.copy_(torch.linspace(0.4, 1.6, m.num_features))
                m.weight.copy_(torch.linspace(0.6, 1.8, m.num_features))
                m.bias.copy_(torch.linspace(-0.3, 0.3, m.num_features))
    net.eval()
    n_out = 2 * 6 if cls is MixedOut else hg.N_OUT
    flow = convert_torch_model(net, net.shape, n_out, device="cpu", encoding_layer_placement="subsume").eval()
    perceptrons = list(flow.get_perceptrons())
    for p, theta in zip(perceptrons, thetas):
        p.set_activation_scale(torch.as_tensor(theta, dtype=torch.float32).clone())
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_, ir, hybrid = hg.scm_sequence(
        flow, perceptrons, wq=False, retimed=hg.SEMANTICS[semantics][0],
        establish=lambda f: establish_deployed_wire_gauge(f, input_data_scale=INPUT_MAX))
    th = [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons]
    return hg.Program(form, cls.shape, flow, repr_, ir, hybrid, perceptrons, th, False)


class _Walk(nn.Module):
    """The NF twin as a gate reads it: the LIF walk of the program's model, its output in the value domain."""

    def __init__(self, program: hg.Program, semantics: str):
        super().__init__()
        self.flow = program.flow
        self.program, self.semantics = program, semantics

    def forward(self, x):
        return SegmentForwardDriver(self.program.repr_, T, hg.SEMANTICS[self.semantics][2]())(x)


def _nf(program, semantics, x):
    with torch.no_grad(), measurement_plane():
        return nf_scm_parity.twin_as_program(program.hybrid, nf_scm_parity.forward_of, _Walk(program, semantics), x)


# ---- the one decode: both twins' outputs read in the value domain ------------------------------------------------------

@pytest.mark.parametrize("form", ["host_per_channel", "core_per_channel"])
def test_the_program_s_output_gauge_places_a_per_channel_gauge_element_by_element(form):
    """A host op's emitted gauge and a core's thresholds, per output element (each twin may hold an output in its own
    currency, a MIXED gather for one; what both read is the decoded value)."""
    program = _program(form, "streamed_lif")
    np.testing.assert_allclose(_od().program_output_gauge(program.hybrid), np.asarray(PER_CHANNEL), rtol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", sorted(FORMS) + ["densenet_bc", "vit_tiny"])
def test_the_deployed_read_decoded_equals_the_walk_s_output(form, semantics):
    program = hg.build(form, 0, 1.3, semantics, False) if form not in FORMS else _program(form, semantics)
    x = hg.samples(program, 16, 0)
    flow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        deployed = _od().executor_output_values(flow, flow(x))
    torch.testing.assert_close(deployed.double(), _nf(program, semantics, x).double(), rtol=RTOL, atol=ATOL)


def test_decode_held_is_the_held_value_times_its_gauge():
    decode_held = _od().decode_held
    held = torch.tensor([[1.0, 2.0, -3.0]])
    assert decode_held(held, None) is held
    torch.testing.assert_close(decode_held(held, 2.0), held * 2.0)
    torch.testing.assert_close(decode_held(held, np.array([1.0, 0.5, 2.0])), torch.tensor([[1.0, 1.0, -6.0]]))
    np.testing.assert_array_equal(decode_held(np.array([[3, 4]]), np.array([0.5, 2.0])), np.array([[1.5, 8.0]]))


# ---- the ready red: the decision gates at a per-channel host output -------------------------------------------------

@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_torch_against_deployed_gate_reads_both_twins_in_one_unit(semantics):
    """The verify's case: 16 samples, a per-channel host output; the agreement was 0.875 (classes of two units)."""
    program = _program("host_per_channel", semantics)
    x = hg.samples(program, 16, 0)
    agreement = nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise(
        _Walk(program, semantics), hg.deployed_flow(program, semantics), x, min_agreement=0.0)
    assert agreement == 1.0


class _Pipeline:
    config: dict = {"device": "cpu"}
    reporter = None


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_cascaded_decision_gate_reads_both_twins_in_one_unit(semantics, monkeypatch):
    program = _program("host_per_channel", semantics)
    monkeypatch.setattr(nf_scm_parity, "_build_cascaded_identity_executor",
                        lambda _pipeline, _model, _ir: hg.deployed_flow(program, semantics))
    x = hg.samples(program, 16, 0)
    agreement = nf_scm_parity._cascaded_nf_scm_agreement(_Pipeline(), _Walk(program, semantics), program.hybrid, x,
                                                         min_agreement=0.0)
    assert agreement == 1.0


# ---- the deployed readouts --------------------------------------------------------------------------------------------

def _value_classes(program, x):
    with torch.no_grad():
        return type(program.flow).forward(program.flow, x).argmax(1)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_deployed_readout_reads_the_value_forward_s_classes(semantics):
    program = _program("host_per_channel", semantics)
    x = hg.samples(program, 256, 1)
    with torch.no_grad():
        got = _od().DeployedValueReadout(hg.deployed_flow(program, semantics))(x)
    assert bool((got.argmax(1) == _value_classes(program, x)).all())
    torch.testing.assert_close(got.double(), _nf(program, semantics, x).double(), rtol=RTOL, atol=ATOL)


def test_the_census_reads_the_deployed_readout(monkeypatch):
    from mimarsinan.pipelining.core import simulation_factory as sf

    program = _program("host_per_channel", "streamed_lif")
    flow = hg.deployed_flow(program, "streamed_lif")
    seen = {}
    monkeypatch.setattr(sf, "spiking_census_arithmetic", lambda *a, **k: type("A", (), {"device": "cpu"})())
    monkeypatch.setattr(sf, "announce", lambda *a, **k: None)
    monkeypatch.setattr(sf, "borrowed_placement", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(sf, "build_deployed_metric_flow", lambda *a, **k: flow)
    monkeypatch.setattr(sf, "run_hcm_spiking_test", lambda _p, model, **k: seen.setdefault("model", model) and 0.0)
    sf._run_spiking_census(None, program.hybrid, "census", model=None, device=None, max_batch_cap=None,
                           retry_on_oom=False, announce_census=False)
    x = hg.samples(program, 256, 1)
    with torch.no_grad():
        classes = seen["model"](x).argmax(1)
    assert bool((classes == _value_classes(program, x)).all())


def test_the_membrane_diagnostic_counts_class_flips_in_one_unit(monkeypatch):
    """Its counts read and its membrane read differ by a correction in count units; the flips it reports are the value
    domain's classes' flips."""
    from mimarsinan.pipelining.core import simulation_factory as sf

    program = _program("host_per_channel", "streamed_lif")
    flow = hg.deployed_flow(program, "streamed_lif")
    shift = torch.zeros(hg.N_OUT, dtype=torch.float64)
    shift[0] = 8.0  # count units: decisive for the raw counts, small at the column's gauge (the smallest)

    class _Corrected(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base, self.hybrid_mapping = base, base.hybrid_mapping
            self.simulation_length, self.spiking_mode = base.simulation_length, base.spiking_mode

        def forward(self, x):
            return self.base(x) + shift.to(torch.float64)

    corrected = _Corrected(flow)
    flows = iter([flow, corrected])
    monkeypatch.setattr(sf, "build_spiking_hybrid_flow", lambda *a, **k: next(flows))
    monkeypatch.setattr(sf, "build_deployment_contract", lambda _p: type("C", (), {"spiking_mode": "lif"})())
    monkeypatch.setattr(sf, "deployed_host_device", lambda _d: "cpu")
    monkeypatch.setattr(sf, "deployed_membrane_decode_enabled", lambda _p: False)
    monkeypatch.setattr(sf, "emit_reporter_event", lambda *a, **k: None)
    monkeypatch.setattr(sf, "final_only_output_nodes", lambda _m: [])
    monkeypatch.setattr(sf, "borrowed_host_placement", lambda *a, **k: contextlib.nullcontext())
    pipeline = type("P", (), {"config": {"device": "cpu", "lif_membrane_readout": True}, "reporter": None})()
    x = hg.samples(program, sf.MEMBRANE_DIAGNOSTIC_MAX_SAMPLES, 2)
    stats = sf.run_membrane_readout_diagnostic(pipeline, program.hybrid, x, model=None)
    gauge = torch.tensor(PER_CHANNEL)  # the form's output gauge per class (pinned by the gauge test above)
    with torch.no_grad():
        counts = flow(x)

    def decoded(read):
        return read / float(T) * gauge.to(read.dtype)

    want = int((decoded(counts).argmax(1) != decoded(counts + shift).argmax(1)).sum())
    assert int((counts.argmax(1) != (counts + shift).argmax(1)).sum()) != want  # the case tells the two units apart
    assert stats is not None and stats["pred_flips"] == want


def test_the_program_classes_of_a_held_read_are_the_value_forward_s():
    """The chip simulators' readouts hold each output in its producer's state (no window): their classes."""
    program = _program("host_per_channel", "streamed_lif")
    x = hg.samples(program, 256, 1)
    flow = hg.deployed_flow(program, "streamed_lif")
    with torch.no_grad():
        held = (flow(x) / float(T)).numpy()
    classes = np.argmax(_od().decode_held(held, _od().program_output_gauge(program.hybrid)), axis=1)
    np.testing.assert_array_equal(classes, _value_classes(program, x).numpy())


@pytest.mark.parametrize("site", ["simulation_runner.hybrid", "simulation_runner.flat", "lava_loihi.runner"])
def test_every_chip_simulator_readout_ranks_decoded_outputs(site):
    """Their executors are external (nevresim, Lava), so the readout is read from the module's source: every class
    ranking (an ``argmax``) ranks decoded outputs, and the flat route hands the driver its program's gauge."""
    import importlib

    module = importlib.import_module(f"mimarsinan.chip_simulation.{site}")
    tree = ast.parse(inspect.getsource(module))
    argmaxes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "argmax"]
    assert argmaxes, site
    for call in argmaxes:
        inner = call.args[0]
        assert isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "decode_held", ast.unparse(call)
    if site.endswith("flat"):
        assert "output_gauge=gauge" in inspect.getsource(module)


def test_the_nevresim_driver_ranks_decoded_outputs():
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

    counts = [10, 3, 2, 9]  # two samples, two classes
    gauge = np.array([0.2, 1.0])
    predictions, total = NevresimDriver._simulator_output_to_predictions(object(), counts, 2, gauge)
    assert list(predictions) == [1, 1] and total == 24
    plain, _ = NevresimDriver._simulator_output_to_predictions(object(), counts, 2)
    assert list(plain) == [0, 1]


def test_the_board_readout_refuses_more_than_one_gauge(monkeypatch):
    """The ODIN board ranks a core's raw counts: they rank as the classes only at one gauge."""
    from mimarsinan.chip_simulation.odin_hacc import program_freeze as pf

    monkeypatch.setattr(pf, "_require_segment_is_the_classifier", lambda *a, **k: None)
    monkeypatch.setattr(pf, "used_neurons", lambda _core: 3)
    src = [type("S", (), {"core_": 0, "neuron_": i})() for i in range(3)]
    segment = type("M", (), {"output_sources": np.array(src, dtype=object), "cores": [None]})()
    gauges = {"uniform": np.array([1.5, 1.5, 1.5]), "per_neuron": np.array([1.0, 1.5, 2.0])}
    monkeypatch.setattr(pf, "program_output_gauge", lambda _program: gauges["uniform"])
    assert pf.readout_core_of(object(), segment, classes=3) == 0
    monkeypatch.setattr(pf, "program_output_gauge", lambda _program: gauges["per_neuron"])
    with pytest.raises(pf.OdinHaccExportRefusal, match="one gauge"):
        pf.readout_core_of(object(), segment, classes=3)


def test_the_pretrained_bridge_s_deployed_logits_are_decoded(monkeypatch):
    from mimarsinan.models.pretrained_bridge import deploy_and_eval

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c, self.fc = nn.Conv2d(1, 2, 3), nn.Linear(2 * 4 * 4, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.c(x)).flatten(1))

    torch.manual_seed(0)
    model = _Tiny()
    x, y = torch.rand(8, 1, 6, 6), torch.randint(0, 3, (8,))
    unit = deploy_and_eval(model, (1, 6, 6), 3, x, y, simulation_length=4)
    monkeypatch.setattr(_od(), "program_output_gauge", lambda *a, **k: np.array([1.0, 2.0, 4.0]))
    scaled = deploy_and_eval(model, (1, 6, 6), 3, x, y, simulation_length=4)
    torch.testing.assert_close(scaled.logits, unit.logits * torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))


def test_the_timing_family_s_teacher_decodes_by_the_program_s_gauge():
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import _Rung2TeacherFlow

    program = _program("host_per_channel", "streamed_lif")
    flow = hg.deployed_flow(program, "streamed_lif")
    teacher = _Rung2TeacherFlow(flow, T, _od().program_output_gauge(program.hybrid, timing=True))
    x = hg.samples(program, 32, 3)
    with torch.no_grad():
        got, want = teacher(x), _od().executor_output_values(flow, flow(x))
    torch.testing.assert_close(got.double(), want.double(), rtol=1e-6, atol=1e-6)


def test_the_timing_teacher_s_gauge_is_the_program_s(monkeypatch):
    from mimarsinan.models.spiking.hybrid import identity_flow
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner

    program = _program("host_per_channel", "streamed_lif")
    flow = hg.deployed_flow(program, "streamed_lif")
    monkeypatch.setattr(identity_flow, "build_identity_spiking_flow", lambda *a, **k: flow)
    host = type("H", (), {"pipeline": type("P", (), {"config": {"input_shape": program.shape}})(), "_T": T,
                          "_teacher": None, "_thresholding_mode": "<=", "_map_teacher_to_ir": lambda self: program.ir})()
    teacher = TTFSCycleAdaptationTuner._build_rung2_teacher(host)
    np.testing.assert_allclose(teacher._output_scales.double().numpy(),
                               _od().program_output_gauge(program.hybrid, timing=True), rtol=1e-6)


# ---- the verify's m-3: the three readers it names ---------------------------------------------------------------------

def _installed_pair():
    """(value twin, LIF-installed model) of U-TGAUGE's host classifier in the count-identity form (the walk and the
    value forward compute one function), the walk installed as the model's forward."""
    import unit.spiking.test_walk_output_value_domain as tw

    twin, model = tw._pair("windowed")
    return tw, twin, model


def test_the_learning_rate_explorer_ranks_value_domain_losses():
    """``make_loss_slope_signal`` ranks candidate rates by the trainer's loss on a training batch."""
    from mimarsinan.model_training.training_utilities import BasicClassificationLoss
    from mimarsinan.tuning.learning_rate_explorer import make_loss_slope_signal

    tw, twin, model = _installed_pair()
    x, y = tw._samples(tw.HostHead.shape), tw._labels()

    def _trainer(m):
        t = type("Tr", (), {})()
        t.model, t.device, t.loss_function = m, "cpu", BasicClassificationLoss()
        t.next_training_batch = lambda: (x, y)
        from mimarsinan.model_training.basic_trainer_eval import evaluate_loss_on_batch
        t.evaluate_loss_on_batch = lambda batch: evaluate_loss_on_batch(t, batch)
        return t

    assert make_loss_slope_signal(_trainer(model))() == pytest.approx(make_loss_slope_signal(_trainer(twin))(),
                                                                       rel=1e-5)


def test_the_ledger_s_alignment_pair_reads_the_walk_in_the_twin_s_unit():
    """Under the LIF exact QAT the activation step's full-transform forward is the walk: its gradient against the value
    forward's gradient (rho, the norm) reads one function in one unit."""
    from mimarsinan.tuning.orchestration import mbh_ledger

    from mimarsinan.tuning.forward_install import ChipAlignedNFForward

    tw, twin, _model = _installed_pair()
    x, y = tw._samples(tw.HostHead.shape), tw._labels()
    host = type("Tu", (), {})()
    host.trainer, host.model = object(), twin
    host.pipeline = type("P", (), {"config": {"device": "cpu"}})()
    host._mbh_full_transform_forward = lambda clone: ChipAlignedNFForward(clone, T, soma_law=DEFAULT_SOMA_LAW,
                                                                          **tw.DISCIPLINES["windowed"])
    original = mbh_ledger._fixed_validation_batch
    try:
        mbh_ledger._fixed_validation_batch = lambda _trainer: (x, y)
        rho, norm = mbh_ledger.transfer_alignment(host)
    finally:
        mbh_ledger._fixed_validation_batch = original
    assert rho == pytest.approx(1.0, rel=1e-4) and norm > 0.0


def test_the_prefix_ramp_s_plain_loss_reads_one_unit():
    from mimarsinan.tuning.orchestration.blend_ramp import PlainClassificationLoss

    tw, twin, model = _installed_pair()
    x, y = tw._samples(tw.HostHead.shape), tw._labels()
    with torch.no_grad():
        got, want = PlainClassificationLoss()(model, x, y), F.cross_entropy(twin(x), y)
    assert float(got) == pytest.approx(float(want), rel=1e-5)


@pytest.mark.parametrize("output", ["core", "host"])
def test_the_timing_family_s_decoded_read_equals_its_walk(output):
    """The cascaded timing program pair (the converted flow walked by ``TTFSSegmentForward`` and its identity
    executor): the executor's read decoded at the timing family's convention equals the walk's output value for value,
    at a core output node and at a host output node (an affine host op after the last core)."""
    import unit.spiking.test_cascaded_boundary_transcoding as cb
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    class HostTail(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = cb._AffineChainMLP([10, 12, 12, 4])

        def forward(self, x):
            return self.body(x) * 0.7 + 0.3

    torch.manual_seed(4)
    model = cb._AffineChainMLP([10, 12, 12, 4]) if output == "core" else HostTail()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.uniform_(m.bias, -0.05, 0.05)
    flow, executor, _ir = cb._build_pair(model, (10,), 4, 32, "<=")
    x = cb._samples((10,))
    with torch.no_grad():
        walk = TTFSSegmentForward(flow.get_mapper_repr(), 32)(x)
        decoded = _od().executor_output_values(executor, executor(x))
    torch.testing.assert_close(decoded.double(), walk.double(), rtol=1e-6, atol=1e-6)


def test_a_host_op_s_held_gauge_follows_its_family_s_stage_convention():
    """A host op that owns no scale domain: a rate family leaves its value at its buffer gauge, a timing family
    re-normalizes it by the op's out-scale (``resolve_stage_compute_scales``); a domain-owning op at its emitted gauge."""
    from mimarsinan.mapping.ir import IRSource

    plain = type("Op", (), {"id": 5, "params": {"module": nn.Identity()}})()
    stage = type("St", (), {"kind": "compute", "compute_op": plain})()
    program = type("Pr", (), {"stages": [stage], "output_sources": np.array([IRSource(5, 0), IRSource(5, 1), IRSource(-2, 0)],
                                                                             dtype=object),
                              "node_activation_scales": {5: 2.5}, "node_buffer_scales": {5: 1.25}})()
    np.testing.assert_allclose(_od().program_output_gauge(program, timing=False), [1.25, 1.25, 1.0])
    np.testing.assert_allclose(_od().program_output_gauge(program, timing=True), [2.5, 2.5, 1.0])
