"""[U-TGAUGE] A forward returns its output in the value domain (T1-A of the T1 read).

The LIF install makes the chip-aligned walk the model's forward for every later stage. The walk carries every value in
the wire currency: a perceptron emits ``clamp(value / theta)`` and a host op through its ScaleNormalizingWrapper computes
``f(r_1 s_1, ...) / s_out``. So an output node leaves the walk divided by the gauge it was encoded at. An argmax read is
blind to a positive scalar factor; every reader of the output as a value is not (the training loss of a Weight
Quantization endpoint read logits divided by the output gauge, about 2.3 on the measured rows). The rule: the walk decodes its output node by the
inverse of its own encode there, and nothing inside the walk changes.

The forms are converted torch blocks (the encoding subsumed on the host) in the count-identity form (non-negative charge
within one spike per cycle at every core, U-LIF's form), so the LIF walk and the value forward compute one function and
any departure is the output's unit:
- a host linear output node at gauge 2.3 (a host classifier, t1-1);
- a core output node (t1-2) and a layout node after a core as the output (the core's gauge passes through it);
- a per-channel output gauge (a host normalization reading a core at per-channel thresholds, t1-5).

The class of readers, swept over the toolchain (one test each): the pipeline's classification losses (plain and with the
activation term), the regression loss, the distillation loss and its temperature, the deployed-risk loop's
distillation, the gradient-alignment ledger's loss gradients, the published argmax reads, the NF-SCM parity records and
the decision gate (t1-4: unchanged), and the twin comparison of the NF output with the deployed flow's output.
"""

from __future__ import annotations

import contextlib
import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.decorators.clamp_quantize import LIFCountStaircaseDecorator
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.nn.lif_kernels import window_counts
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.tuning.forward_install import ChipAlignedNFForward

T = 15
N_OUT = 6
#: A host classifier's output gauge (the measured rows' 2.307), the factor the walk's output carried.
GAUGE = 2.3
INPUT_MAX = 2.0
N = 64
#: Float rounding between the walk's wire decode and the value forward (the defect is a factor of 1 / 2.3).
RTOL, ATOL = 1e-5, 1e-6
DISCIPLINES = {"windowed": dict(retime=True, synchronized=False), "two_window": dict(synchronized=True)}


class HostHead(nn.Module):
    """A subsumed encoding, a core, and a host linear output node (a host classifier): the output gauge is the core's."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.head = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, N_OUT)
        self.r0, self.r1 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.head(self.r1(self.fc1(self.r0(self.stem(x.flatten(1))))))


class CoreHead(HostHead):
    """The same block with a core as the output node."""

    def __init__(self):
        super().__init__()
        self.r2 = nn.ReLU()

    def forward(self, x):
        return self.r2(super().forward(x))


class CoreLayout(nn.Module):
    """A layout node after a core as the output node (the core's gauge passes through it)."""

    shape = (3, 4, 4)

    def __init__(self):
        super().__init__()
        self.stem, self.c1 = nn.Conv2d(3, 2, 3, padding=1), nn.Conv2d(2, 2, 3, padding=1)
        self.r0, self.r1 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.r1(self.c1(self.r0(self.stem(x)))).flatten(1)


class HostNormHead(HostHead):
    """A host normalization as the output node, reading a core at per-channel thresholds: a per-channel output gauge."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(6)

    def forward(self, x):
        return self.bn(self.r1(self.fc1(self.r0(self.stem(x.flatten(1))))))


#: Per-channel thresholds of the core read by the per-channel output node.
PER_CHANNEL = (0.6, 1.1, 1.7, 2.3, 2.9, 3.5)


def _count_identity(net: nn.Module, gauges: dict) -> None:
    """Non-negative weights and biases, every core's per-cycle charge within one spike per cycle at its threshold."""
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.copy_(torch.rand(m.weight.shape, generator=gen))
                m.bias.copy_(0.02 + 0.05 * torch.rand(m.bias.shape, generator=gen))
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.copy_(torch.linspace(-0.1, 0.1, m.num_features))
                m.running_var.copy_(torch.linspace(0.5, 1.5, m.num_features))
                m.weight.copy_(torch.linspace(0.5, 2.0, m.num_features))
                m.bias.copy_(torch.linspace(-0.2, 0.2, m.num_features))
        for name, (gauge_in, theta) in gauges.items():
            layer = getattr(net, name)
            w = layer.weight.reshape(layer.weight.shape[0], -1)
            per_cycle = gauge_in * w.sum(dim=1) + layer.bias
            room = (0.95 * torch.as_tensor(theta).reshape(-1).expand(w.shape[0]) / per_cycle).clamp(max=1.0)
            layer.weight.mul_(room.reshape(-1, *([1] * (layer.weight.dim() - 1))))
            layer.bias.mul_(room)


def _form(cls, thetas):
    """``cls`` converted (encoding subsumed), LIF nodes at ``thetas`` (the perceptrons in order), its wire gauge
    established by the seam; every perceptron's charge kept within one spike per cycle of its input gauge."""
    torch.manual_seed(0)
    net = cls()
    theta_list = [torch.as_tensor(t, dtype=torch.float32) for t in thetas]
    chain = [n for n in ("stem", "fc1", "c1", "head") if hasattr(net, n)][: len(theta_list)]
    gauges_in = [INPUT_MAX] + [float(t.max()) for t in theta_list[:-1]]
    _count_identity(net, {name: (g, t) for name, g, t in zip(chain, gauges_in, theta_list)})
    net.eval()
    flow = convert_torch_model(net, net.shape, N_OUT, device="cpu", encoding_layer_placement="subsume").eval()
    for p, theta in zip(flow.get_perceptrons(), theta_list):
        p.set_activation_scale(theta.clone())
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    establish_deployed_wire_gauge(flow, input_data_scale=INPUT_MAX)
    return flow


def _installed(flow, discipline: str):
    """The LIF-installed model: the chip-aligned walk as ``model.forward``, as the LIF step's finalize installs it."""
    model = copy.deepcopy(flow)
    model.forward = ChipAlignedNFForward(model, T, soma_law=DEFAULT_SOMA_LAW, **DISCIPLINES[discipline])
    return model


def _samples(shape, n: int = N, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return INPUT_MAX * torch.rand(n, *shape, generator=g)


def _labels(n: int = N, seed: int = 4) -> torch.Tensor:
    return torch.randint(0, N_OUT, (n,), generator=torch.Generator().manual_seed(seed))


def _host_head():
    return _form(HostHead, (INPUT_MAX, GAUGE))


def _pair(discipline: str, cls=HostHead, thetas=(INPUT_MAX, GAUGE)):
    """(value twin, LIF-installed model) of one form: the twin is the flow's own value forward."""
    flow = _form(cls, thetas)
    return flow, _installed(flow, discipline)


def _output_gauge(flow) -> torch.Tensor:
    return torch.as_tensor(flow.get_mapper_repr().output_layer_mapper.output_scale).reshape(-1)


# ---- the rule: the output in the value domain -----------------------------------------------------------------------

@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_t1_1_a_host_output_node_leaves_the_walk_in_the_value_domain(discipline):
    twin, model = _pair(discipline)
    torch.testing.assert_close(_output_gauge(twin), torch.tensor([GAUGE]))
    x = _samples(HostHead.shape)
    with torch.no_grad():
        torch.testing.assert_close(model(x), twin(x), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_t1_2_a_core_output_node_leaves_the_walk_in_the_value_domain(discipline):
    twin, model = _pair(discipline, CoreHead, (INPUT_MAX, GAUGE, 1.4))
    x = _samples(CoreHead.shape)
    with torch.no_grad():
        torch.testing.assert_close(model(x), twin(x), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_t1_2_a_layout_output_after_a_core_is_decoded_at_the_core_s_threshold(discipline):
    twin, model = _pair(discipline, CoreLayout, (INPUT_MAX, GAUGE))
    x = _samples(CoreLayout.shape)
    with torch.no_grad():
        torch.testing.assert_close(model(x), twin(x), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_t1_5_a_per_channel_output_gauge_reads_the_value_forward_s_classes(discipline):
    twin, model = _pair(discipline, HostNormHead, (INPUT_MAX, PER_CHANNEL))
    assert _output_gauge(twin).unique().numel() == len(PER_CHANNEL)
    x = _samples(HostNormHead.shape, n=256)
    with torch.no_grad():
        got, want = model(x), twin(x)
    assert bool((got.argmax(1) == want.argmax(1)).all())
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


class MixedOutput(nn.Module):
    """An absolute host chain (a stem without activation, rooted at the input) concatenated with a core output as the
    output node: a MIXED gather, each element in its producer's currency."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1 = nn.Linear(8, 6), nn.Linear(6, 6)
        self.r1 = nn.ReLU()

    def forward(self, x):
        s = self.stem(x.flatten(1))
        return torch.cat([s, self.r1(self.fc1(s))], dim=1)


@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_a_mixed_output_node_leaves_the_walk_element_by_element_in_its_producers_values(discipline):
    """The absolute part as the host computed it (signed, unclamped), the core's part at its decoded value: no element is
    re-encoded for a reader the output does not have."""
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    torch.manual_seed(0)
    flow = convert_torch_model(MixedOutput().eval(), MixedOutput.shape, 2 * N_OUT, device="cpu",
                               encoding_layer_placement="subsume").eval()
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(GAUGE))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    establish_deployed_wire_gauge(flow, input_data_scale=INPUT_MAX)
    x = _samples(MixedOutput.shape)
    stem = next(m for n, m in flow.named_modules() if isinstance(m, nn.Linear) and m.in_features == 8)
    rec: dict = {}
    with torch.no_grad():
        out = chip_aligned_segment_forward(flow, x, T, soma_law=DEFAULT_SOMA_LAW, node_value_recorder=rec,
                                           **DISCIPLINES[discipline])
        host = stem(x)
    (core,) = flow.get_perceptrons()
    assert bool((host < 0).any())
    torch.testing.assert_close(out, torch.cat([host, rec[id(core)]], dim=1), rtol=RTOL, atol=ATOL)


# ---- the scope: what the rule leaves unchanged ----------------------------------------------------------------------

def _records(model, x) -> dict:
    """Each perceptron's decoded window value in the walk (the parity gate's per-perceptron records)."""
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    rec: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(model, x, T, soma_law=DEFAULT_SOMA_LAW, retime=True, node_value_recorder=rec)
    return {k: rec[id(p)] for k, p in enumerate(model.get_perceptrons())}


def _value_activations(twin, x) -> dict:
    out, hooks = {}, []
    for k, p in enumerate(twin.get_perceptrons()):
        hooks.append(p.activation.register_forward_hook(lambda m, a, o, k=k: out.__setitem__(k, o.detach())))
    with torch.no_grad():
        twin(x)
    for h in hooks:
        h.remove()
    return out


def test_t1_4_the_published_reads_and_the_per_perceptron_records_are_unchanged():
    """Argmax reads (every published accuracy read) and the walk's per-perceptron records (the NF-SCM parity gate's) are
    the value forward's, as they were before the rule: the rule changes the output's unit, nothing inside the walk."""
    twin, model = _pair("windowed")
    x = _samples(HostHead.shape, n=256)
    with torch.no_grad():
        assert bool((model(x).argmax(1) == twin(x).argmax(1)).all())
    records, values = _records(model, x), _value_activations(twin, x)
    assert sorted(records) == sorted(values)
    for k in records:
        torch.testing.assert_close(records[k], values[k], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("semantics", ["streamed_lif", "windowed_lif"])
def test_t1_4_the_decision_gate_and_the_parity_records_against_the_deployed_program_are_unchanged(semantics):
    """The identity program of the form: the NF walk's window counts equal the deployed flow's on every core, and the two
    twins' classes agree on every sample (the decision gate), with the output node a host op."""
    from unit.spiking import host_gather_forms as hg

    flow = _host_head()
    perceptrons = list(flow.get_perceptrons())
    repr_, ir, hybrid = hg.scm_sequence(flow, perceptrons, wq=False, retimed=hg.SEMANTICS[semantics][0],
                                        establish=lambda f: establish_deployed_wire_gauge(f, input_data_scale=INPUT_MAX))
    program = hg.Program("host_head", HostHead.shape, flow, repr_, ir, hybrid, perceptrons,
                         [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons], False)
    x = _samples(HostHead.shape, n=32)
    nf_out, nf = hg.nf_window_counts(program, x, semantics)
    scm_out, scm = hg.scm_window_counts(program, x, semantics)
    assert all(v == 0 for v in hg.count_mismatches(nf, scm).values())
    assert bool((nf_out.argmax(1) == scm_out.argmax(1)).all())


def test_the_twin_comparison_of_outputs_reads_both_twins_in_one_unit():
    """The NF-SCM twin comparison of the output (host_gather_forms.compare) reads the NF output, now in the value
    domain, in the deployed flow's units: the outputs agree on the count grid at a host output node."""
    from unit.spiking import host_gather_forms as hg

    flow = _host_head()
    perceptrons = list(flow.get_perceptrons())
    repr_, ir, hybrid = hg.scm_sequence(flow, perceptrons, wq=False, retimed=False,
                                        establish=lambda f: establish_deployed_wire_gauge(f, input_data_scale=INPUT_MAX))
    program = hg.Program("host_head", HostHead.shape, flow, repr_, ir, hybrid, perceptrons,
                         [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons], False)
    x = _samples(HostHead.shape, n=32)
    nf_out, _ = hg.nf_window_counts(program, x, "streamed_lif")
    scm_out, _ = hg.scm_window_counts(program, x, "streamed_lif")
    ref, dep = hg.output_counts(program, nf_out, scm_out, x.shape[0])
    assert float((ref - dep).abs().max()) <= 1e-3


# ---- the class swept: every reader of the output as a value ---------------------------------------------------------

#: [composed head, merge rule 21] Gauges at which the walk's divide and multiply by a gauge are exact (powers of two):
#: the count branch's gradient is read against its reference bit for bit there. At the tests' gauge 2.3 the two differ
#: only by the rounding of ``x / 2.3 * 2.3`` (about 1e-8).
EXACT_GAUGES = (INPUT_MAX, 4.0)


def _stand_in(flow):
    """[composed head, merge rule 21] Each core hop's LIF node replaced by its exact-QAT stand-in, the LIF count
    staircase (``LIFCountStaircaseDecorator``): the synchronized walk evaluates such a hop by its own activation, the
    kernel the value twin keeps."""
    for p in flow.get_perceptrons():
        if not getattr(p, "is_encoding_layer", False):
            stand_in = TransformedActivation(nn.Identity(), [LIFCountStaircaseDecorator(T, p.activation_scale)])
            p.base_activation = stand_in
            p.activation = stand_in
    return flow


@contextlib.contextmanager
def _count_branch(flow):
    """[composed head, merge rule 21] The value twin read as the synchronized walk reads a LIF hop ([U-LIF2 W10]): each
    core hop's LIF node by its window count (``window_counts``), whose gradient is the straight-through clamp of the
    charge. The tests' reference on the count branch."""
    with contextlib.ExitStack() as stack:
        for p in flow.get_perceptrons():
            if not getattr(p, "is_encoding_layer", False):
                stack.enter_context(window_counts(unwrap_lif_activation(p.activation)))
        yield flow


def _sync_pair(thetas=(INPUT_MAX, GAUGE), stand_in=False):
    """(value twin, LIF-installed model) of the host-head form under the synchronized walk (integrate then emit)."""
    flow = _form(HostHead, thetas)
    if stand_in:
        _stand_in(flow)
    return flow, _installed(flow, "two_window")


def _grads(model, loss_fn, x, y, scope=contextlib.nullcontext) -> tuple:
    model.zero_grad(set_to_none=True)
    with scope():
        loss = loss_fn(model, x, y)
        loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same_loss_and_gradients(loss_fn):
    """[composed head, merge rule 21] The loss through the installed walk is the value twin's on every form. Its
    gradient is the twin's as written where the walk evaluates each hop by the kernel the twin keeps (an exact-QAT
    stand-in hop). On the count branch (a LIF hop of the synchronized walk) the parameters that receive a gradient are
    the twin's, and the gradient is the twin's read in the count branch's form (``_count_branch``), bit for bit."""
    x, y = _samples(HostHead.shape), _labels()
    for discipline in sorted(DISCIPLINES):
        twin, model = _pair(discipline)
        with torch.no_grad():
            torch.testing.assert_close(loss_fn(model, x, y), loss_fn(twin, x, y), rtol=RTOL, atol=ATOL)
    twin, model = _sync_pair(stand_in=True)
    want_loss, want_grads = _grads(twin, loss_fn, x, y)
    got_loss, got_grads = _grads(model, loss_fn, x, y)
    torch.testing.assert_close(got_loss, want_loss, rtol=RTOL, atol=ATOL)
    assert sorted(got_grads) == sorted(want_grads)
    for name in want_grads:
        torch.testing.assert_close(got_grads[name], want_grads[name], rtol=1e-4, atol=1e-6)
    for thetas in ((INPUT_MAX, GAUGE), EXACT_GAUGES):
        twin, model = _sync_pair(thetas)
        want_loss, want_grads = _grads(twin, loss_fn, x, y)
        ref_loss, ref_grads = _grads(twin, loss_fn, x, y, scope=lambda: _count_branch(twin))
        got_loss, got_grads = _grads(model, loss_fn, x, y)
        torch.testing.assert_close(got_loss, want_loss, rtol=RTOL, atol=ATOL)
        assert sorted(got_grads) == sorted(want_grads) == sorted(ref_grads) and want_grads
    for name in ref_grads:
        assert torch.equal(got_grads[name], ref_grads[name]), name


def test_t1_3_the_pipeline_classification_loss_and_its_gradient_equal_the_value_twin_s():
    from mimarsinan.model_training.training_utilities import BasicClassificationLoss

    _assert_same_loss_and_gradients(BasicClassificationLoss())


@pytest.mark.parametrize("discipline", sorted(DISCIPLINES))
def test_t1_3_the_pipeline_classification_loss_equals_the_value_twin_s_in_both_disciplines(discipline):
    from mimarsinan.model_training.training_utilities import BasicClassificationLoss

    twin, model = _pair(discipline)
    x, y = _samples(HostHead.shape), _labels()
    loss = BasicClassificationLoss()
    with torch.no_grad():
        torch.testing.assert_close(loss(model, x, y), loss(twin, x, y), rtol=RTOL, atol=ATOL)


def test_the_classification_loss_with_its_activation_term_equals_the_value_twin_s():
    """The loss decorates each perceptron's activation (a TransformedActivation around the LIF node) to read its output."""
    from mimarsinan.model_training.training_utilities import CustomClassificationLoss
    from mimarsinan.models.nn.layers import TransformedActivation

    twin = _form(HostHead, (INPUT_MAX, GAUGE))
    for p in twin.get_perceptrons():
        p.activation = TransformedActivation(p.activation, [])
    model = _installed(twin, "two_window")
    x, y = _samples(HostHead.shape), _labels()
    loss = CustomClassificationLoss()
    with torch.no_grad():
        torch.testing.assert_close(loss(model, x, y), loss(twin, x, y), rtol=RTOL, atol=ATOL)


def test_the_regression_loss_equals_the_value_twin_s():
    from mimarsinan.data_handling.data_provider import RegressionMode

    twin, model = _pair("two_window")
    x = _samples(HostHead.shape)
    target = torch.linspace(-1.0, 1.0, N * N_OUT).reshape(N, N_OUT)
    loss = RegressionMode().create_loss()
    with torch.no_grad():
        torch.testing.assert_close(loss(model, x, target), loss(twin, x, target), rtol=RTOL, atol=ATOL)


def test_t1_6_the_distillation_loss_compares_like_units_with_its_teacher():
    """The KD loss (temperature 3, the LIF step's stabilization loss) on the LIF-installed student against its
    value-domain teacher equals the loss of the teacher's own function as the student."""
    from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss

    twin, model = _pair("two_window")
    x, y = _samples(HostHead.shape), _labels()
    loss = KDClassificationLoss(copy.deepcopy(twin), temperature=3.0, alpha=0.3)
    with torch.no_grad():
        torch.testing.assert_close(loss(model, x, y), loss(twin, x, y), rtol=RTOL, atol=ATOL)


def test_the_deployed_risk_loop_trains_the_installed_walk_as_its_value_twin():
    """The deployed-risk loop's CE and distillation (temperature 4) through the LIF-installed walk: two steps move every
    parameter as through the value twin where the walk evaluates each hop by the kernel the twin keeps (an exact-QAT
    stand-in hop); [composed head, merge rule 21] on the count branch, bit for bit as through the twin read in the
    count branch's form."""
    from mimarsinan.tuning.orchestration.deployed_risk_finetune import run_deployed_risk_finetune

    x, y = _samples(HostHead.shape), _labels()
    steps = dict(train_batches=[(x, y), (x, y)], eval_genuine=lambda: 0.0, steps=2, lr=1e-2, warmup_frac=0.5,
                 eval_every=100)
    twin, model = _sync_pair(stand_in=True)
    teacher = copy.deepcopy(twin)
    for student in (twin, model):
        run_deployed_risk_finetune(student, teacher, student, **steps)
    for (name, a), (_, b) in zip(model.named_parameters(), twin.named_parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6, msg=name)
    twin, model = _sync_pair(EXACT_GAUGES)
    teacher = copy.deepcopy(twin)
    with _count_branch(twin):
        run_deployed_risk_finetune(twin, teacher, twin, **steps)
    run_deployed_risk_finetune(model, teacher, model, **steps)
    for (name, a), (_, b) in zip(model.named_parameters(), twin.named_parameters()):
        assert torch.equal(a, b), name


def test_the_gradient_alignment_ledger_reads_the_installed_walk_s_loss_gradient_as_the_twin_s():
    """[composed head, merge rule 21] The ledger's loss gradients (an evaluation-mode read) through the installed walk:
    the twin's as written at a hop by the kernel the twin keeps (an exact-QAT stand-in hop). On the count branch they
    are the twin's read in the count branch's form, bit for bit, over its parameters: the twin's and the core hops'
    layers, which the straight-through clamp reaches where the IF kernel's evaluation-mode read gives no gradient."""
    from mimarsinan.tuning.orchestration.mbh_ledger import _ce_parameter_grads

    x, y = _samples(HostHead.shape), _labels()
    twin, model = _sync_pair(stand_in=True)
    want = _ce_parameter_grads(twin, twin, x, y, "cpu")
    got = _ce_parameter_grads(model, model, x, y, "cpu")
    assert sorted(got) == sorted(want) and want
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=1e-4, atol=1e-6)
    for thetas in ((INPUT_MAX, GAUGE), EXACT_GAUGES):
        twin, model = _sync_pair(thetas)
        want = _ce_parameter_grads(twin, twin, x, y, "cpu")
        with _count_branch(twin):
            ref = _ce_parameter_grads(twin, twin, x, y, "cpu")
        got = _ce_parameter_grads(model, model, x, y, "cpu")
        core = {n for n, q in twin.named_parameters() for p in twin.get_perceptrons()
                if not getattr(p, "is_encoding_layer", False) and any(q is r for r in p.layer.parameters())}
        assert sorted(got) == sorted(ref) and set(got) == set(want) | core and not set(want) & core and core
    for name in ref:
        assert torch.equal(got[name], ref[name]), name
