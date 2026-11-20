"""[U-POST] The steps after the LIF fix: Weight Quantization's integer program, its published reads and its host
normalizations.

Four toolchain defects of the steps from Weight Quantization through Normalization Fusion (lif_diag's Weight
Quantization study, its independent check and the U-POST ruling), each with its generic fix:
- G4, a membrane lattice armed where the membrane does not live on it: the arming inferred a lattice from any integer
  ``parameter_scale`` (the default 1.0 included) and armed every encoding layer at ``s T``, which presumes an input
  on the 1/T entry grid; a host-side encoder reading a float input had its membrane rounded every cycle. The fix: the
  projection records the lattice's kind (a core, an encoder on the entry grid, none; an encoding layer is an encoder
  on the entry grid only when its input carries that grid) and every arming derives the lattice from the kind and the
  persisted scale (U-POST2: derived, never cached, so every restore carries it).
- G2, the published step reads of an integer program decided exact threshold ties by float noise (outside the
  measurement plane): the fix reads them inside the plane with the derived lattices armed, the parity gate's read.
- G3, the projection and its verification applied to every perceptron: a host-placed (subsumed) encoder now keeps its
  float weights; a perceptron the placement maps to cores (the encoder under offload included) is projected.
- G1, no step re-estimated the host normalizations after the conversions: at the end of Weight Quantization each one
  outside the projected perceptrons is re-estimated over a declared number of training batches, kept only if the
  calibration read does not fall.
The encoder's staircase placement (the theta/(2T) head start on a host-side encoder) was read on the development data
and is not adopted: -0.60 points on the untuned entry and +0.08 on the tuned one, each at a paired standard error of
0.39, so neither read departs from zero by two standard errors (U-POST's and U-POST2's records state both reads).

Every form is a converted torch block with the toolchain's own classes: a subsumed encoding (BN-paired), a BN-paired
core, a host linear with its BatchNorm (a host normalization), the residual add and its ReLU, and a BN-paired core head
(one residual block of a residual network whose host normalizations follow its cores).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
from mimarsinan.models.nn.lif_kernels import in_measurement_plane, measurement_plane
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice, unwrap_lif_activation
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

from unit.spiking.host_gather_forms import Program, compare

T = 15
N_OUT = 6
WQ_BITS = 4
Q_MAX = 2 ** (WQ_BITS - 1) - 1
N_READ = 2000


class _ResidualBN(nn.Module):
    """A residual block: a BN-paired subsumed encoding, a BN-paired core, a host linear with its BatchNorm (no
    activation: a host op), the add with the skip and its ReLU, a BN-paired core head."""

    shape = (8,)

    def __init__(self, width: int = 12):
        super().__init__()
        self.stem, self.bn0, self.r0 = nn.Linear(8, width), nn.BatchNorm1d(width), nn.ReLU()
        self.c1, self.bn1, self.r1 = nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()
        self.h2, self.bn2 = nn.Linear(width, width), nn.BatchNorm1d(width)
        self.rj = nn.ReLU()
        self.c3, self.bn3, self.r3 = nn.Linear(width, N_OUT), nn.BatchNorm1d(N_OUT), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.bn0(self.stem(x.flatten(1))))
        y = self.bn2(self.h2(self.r1(self.bn1(self.c1(s)))))
        return self.r3(self.bn3(self.c3(self.rj(s + y))))


class _TwoResidualBN(nn.Module):
    """Two residual blocks of ``_ResidualBN``'s form: two host normalizations, the second reading the first's join."""

    shape = (8,)

    def __init__(self, width: int = 12):
        super().__init__()
        self.stem, self.bn0, self.r0 = nn.Linear(8, width), nn.BatchNorm1d(width), nn.ReLU()
        self.c1, self.bn1, self.r1 = nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()
        self.h2, self.bn2, self.rj2 = nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()
        self.c3, self.bn3, self.r3 = nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()
        self.h4, self.bn4, self.rj4 = nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()
        self.head, self.bnh, self.rh = nn.Linear(width, N_OUT), nn.BatchNorm1d(N_OUT), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.bn0(self.stem(x.flatten(1))))
        j = self.rj2(s + self.bn2(self.h2(self.r1(self.bn1(self.c1(s))))))
        k = self.rj4(j + self.bn4(self.h4(self.r3(self.bn3(self.c3(j))))))
        return self.rh(self.bnh(self.head(k)))


def _form(seed: int = 0, *, placement: str = "subsume", module=_ResidualBN):
    """The converted block at seeded weights (a live cascade), a LIF node at each perceptron's theta, the wire gauge
    established as Soft Core Mapping's sequence establishes it. Returns ``(flow, perceptrons)``."""
    torch.manual_seed(seed)
    net = module()
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(2.0)
                m.bias.copy_(0.05 + 0.25 * torch.rand(m.bias.shape, generator=gen))
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.uniform_(-0.2, 0.2, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
                m.weight.data.uniform_(0.7, 1.3, generator=gen)
                m.num_batches_tracked.fill_(7000)  # a trained normalization's batch count
    net.eval()
    flow = convert_torch_model(net, (8,), N_OUT, device="cpu", encoding_layer_placement=placement).eval()
    perceptrons = list(flow.get_perceptrons())
    rng = np.random.default_rng(2000 + seed)
    for i, p in enumerate(perceptrons):
        p.set_activation_scale(torch.tensor(2.185 if i == 0 else float(rng.uniform(0.4, 1.6))))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    return flow, perceptrons


def _tuner():
    """The Weight Quantization tuner's projection, built without a pipeline (the step's own 4-bit shared grid)."""
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner as Tuner,
    )

    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits, tuner.two_scale_projection = WQ_BITS, False
    tuner.fractional_scale, tuner.float_bias = False, False
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    return tuner


def _step_projection(flow) -> None:
    """The Weight Quantization step's projection at rate 1, through its tuner (the probe-replica route)."""
    with torch.no_grad():
        _tuner()._apply_rate_to(flow, 1.0)


def _core_projection(perceptrons) -> None:
    """The projection of the cores alone, direct (the toolchain's NAPQ, the step's grid)."""
    napq = NormalizationAwarePerceptronQuantization(WQ_BITS, "cpu", rate=1.0, two_scale=False)
    for p in perceptrons:
        if not getattr(p, "is_encoding_layer", False):
            napq.transform(p)


def _walk(flow, x):
    return SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))(x)


def _inputs(n: int = N_READ, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 2.0 * torch.randn(n, 8, generator=g)


def _exact_reference(flow, x):
    """The integer program's reference read: a float64 copy, its derived lattices armed, inside the measurement plane
    (float64 noise is far below half a lattice quantum, so every exact tie is decided by its exact value)."""
    ref = copy.deepcopy(flow).double()
    arm_integer_membrane_lattice(ref)
    with torch.no_grad(), measurement_plane():
        return _walk(ref, x.double())


def _counts(out: torch.Tensor, theta_out: float) -> torch.Tensor:
    return torch.round(out.double() / float(theta_out) * T)


class _Reader:
    """A trainer's published test read on fixed samples (``basic_trainer_eval.test``'s form): the model's own
    forward, the accuracy against ``labels``; every output kept."""

    def __init__(self, model, x, labels, forward=_walk):
        self.model, self._x, self._labels, self._forward = model, x, labels, forward
        self.outputs: list = []
        self.planes: list = []

    def test(self):
        self.model.eval()
        self.planes.append(in_measurement_plane())
        with torch.no_grad():
            out = self._forward(self.model, self._x)
        self.outputs.append(out.detach().clone())
        return float((out.argmax(1) == self._labels).double().mean())


class _Pipeline:
    def __init__(self, config, entries=None, working_directory=None):
        self.config = config
        self.entries = dict(entries or {})
        self.reported: list = []
        self.events: list = []
        self.cache: dict = {}
        self.working_directory = working_directory
        self.reporter = SimpleNamespace(report=lambda *a, **k: self.reported.append(a),
                                        event=lambda kind, payload: self.events.append((kind, payload)))

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _scm_program(flow, perceptrons) -> Program:
    """Soft Core Mapping's sequence on a projected flow: the IR at the step's q_max, the chip quantization, the arming,
    the per-sample host kernels the records declare, the identity program."""
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=float(Q_MAX), firing_mode="Default", max_axons=256, max_neurons=256).map(repr_)
    quantize_ir_graph(ir, WQ_BITS, weight_quantization=True)
    arm_integer_membrane_lattice(flow)
    declare_host_kernel_evaluation(ir, "per_sample")
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=False)
    thetas = [float(p.activation_scale) for p in perceptrons]
    return Program("residual_bn", (8,), flow, repr_, ir, hybrid, perceptrons, thetas, True)


def _float64_window_counts(perceptron, x: torch.Tensor) -> torch.Tensor:
    """A LIF perceptron's window counts from its own effective parameters in float64, no lattice: a constant per-cycle
    charge, '<=' firing, subtract reset (the host float arithmetic every evaluator approximates)."""
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    pt = PerceptronTransformer()
    xin = perceptron.input_activation(x) if not isinstance(perceptron.input_activation, nn.Identity) else x
    scale = pt.input_scale(perceptron)
    w = pt.get_effective_weight(perceptron).double() / torch.as_tensor(scale).double()
    b = pt.get_effective_bias(perceptron).double()
    charge = xin.double() @ w.T + b
    membrane = torch.zeros_like(charge)
    count = torch.zeros_like(charge)
    for _ in range(T):
        membrane += charge
        fired = membrane >= 1.0
        count += fired.double()
        membrane -= fired.double()
    return count


def _encoder_counts_in_plane(perceptron, x: torch.Tensor) -> torch.Tensor:
    """The deployed host op's counts: the encoder's own module inside the measurement plane (the host tie rule)."""
    with torch.no_grad(), measurement_plane():
        out = perceptron(x)
    return torch.round(out.double() / float(perceptron.activation_scale) * T)


# ---------------------------------------------------------------------------------------------------------------------
# G4: a membrane lattice armed only where the membrane lives on it
# ---------------------------------------------------------------------------------------------------------------------


def _encoder_perceptron(n_out: int = 400, n_in: int = 16, *, grid: bool, seed: int = 0):
    """An encoding-layer perceptron (theta 1, no normalization) with a LIF node; ``grid``: its input passes the
    entry quantizer at the window (the 1/T entry grid)."""
    torch.manual_seed(seed)
    p = Perceptron(n_out, n_in, normalization=nn.Identity(), name="encoder")
    p.is_encoding_layer = True
    p.set_activation_scale(torch.tensor(1.0))
    with torch.no_grad():
        p.layer.weight.mul_(1.5)
        p.layer.bias.uniform_(-0.2, 0.4)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    if grid:
        p.append_input_wire_op(ChipInputQuantizer(T=T, activation_scale=1.0))
    return p


class _Holder(nn.Module):
    def __init__(self, *perceptrons):
        super().__init__()
        self.ps = nn.ModuleList(perceptrons)

    def get_perceptrons(self):
        return list(self.ps)


class TestG4MembraneLattice:
    def test_g4_1_encoder_lattice_only_on_the_entry_grid(self):
        """M4 as a unit: with a float input, the armed exact-plane counts equal the float64 reference (the lattice
        ``s T`` rounded 0.38 percent of windows up); with the 1/T entry grid, they equal the integer reference."""
        g = torch.Generator().manual_seed(3)
        floating = _encoder_perceptron(grid=False)
        NormalizationAwarePerceptronQuantization(WQ_BITS, "cpu", rate=1.0).transform(floating)
        arm_integer_membrane_lattice(_Holder(floating))
        x = torch.rand(50, 16, generator=g)
        assert torch.equal(_encoder_counts_in_plane(floating, x), _float64_window_counts(floating, x))

        gridded = _encoder_perceptron(grid=True)
        NormalizationAwarePerceptronQuantization(WQ_BITS, "cpu", rate=1.0).transform(gridded)
        arm_integer_membrane_lattice(_Holder(gridded))
        s = float(gridded.parameter_scale)
        lif = unwrap_lif_activation(gridded.activation)
        assert lif.if_node.lattice_scale == 2.0 * s * T
        # The integer reference: codes times grid steps over s T, an integer membrane, '<=' at s T.
        codes = torch.round(gridded.layer.weight.detach().double() * s).long()
        bias_codes = torch.round(gridded.layer.bias.detach().double() * s).long()
        k = torch.round(x.double().clamp(0, 1) * T).long()
        charge = k @ codes.T + bias_codes * T
        membrane = torch.zeros_like(charge)
        count = torch.zeros_like(charge)
        for _ in range(T):
            membrane += charge
            fired = membrane >= int(round(s)) * T
            count += fired.long()
            membrane -= fired.long() * int(round(s)) * T
        assert torch.equal(_encoder_counts_in_plane(gridded, x).long(), count)

    def test_g4_2_an_unprojected_perceptron_stays_unarmed(self):
        """A perceptron the projection never reached carries no lattice record, whatever its parameter_scale (the
        default 1.0 armed a core at 2 and an encoder at 2T)."""
        core = Perceptron(4, 8, normalization=nn.Identity(), name="core")
        encoder = _encoder_perceptron(4, 8, grid=True)
        for p in (core, encoder):
            p.set_activation_scale(torch.tensor(1.0))
            lif = LIFActivation(T=T, activation_scale=p.activation_scale)
            p.base_activation = lif
            p.activation = lif
        assert float(core.parameter_scale) == 1.0 and float(encoder.parameter_scale) == 1.0
        assert arm_integer_membrane_lattice(_Holder(core, encoder)) == 0
        assert unwrap_lif_activation(core.activation).if_node.lattice_scale is None
        assert unwrap_lif_activation(encoder.activation).if_node.lattice_scale is None

    def test_g4_2b_the_lattice_record_travels_with_the_projected_parameters(self):
        """The transform trainer projects a copy and writes the parameters back: the projection's lattice record
        reaches the live perceptron with them, so a recreated activation is re-armed from the record and the persisted
        scale."""
        from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer

        flow, perceptrons = _form(0)
        trainer = PerceptronTransformTrainer.__new__(PerceptronTransformTrainer)
        trainer.device = "cpu"
        trainer.model = flow
        trainer.aux_model = copy.deepcopy(flow)
        trainer.perceptron_transformation = _tuner()._mixed_transform(1.0)
        trainer._perceptron_slots = []
        for name, module in flow.named_modules():
            if isinstance(module, Perceptron):
                aux = trainer._get_module_by_name(trainer.aux_model, name)
                trainer._perceptron_slots.append((module, aux, copy.deepcopy(aux)))
        with torch.no_grad():
            trainer._update_and_transform_model()
        for p in perceptrons:
            p.activation = LIFActivation(T=T, activation_scale=p.activation_scale)  # a tuning stage recreates it
        assert arm_integer_membrane_lattice(flow) == len(perceptrons) - 1
        for p in perceptrons:
            scale = unwrap_lif_activation(p.activation).if_node.lattice_scale
            if p.is_encoding_layer:
                assert scale is None
            else:
                assert scale == 2.0 * float(p.parameter_scale) and float(p.parameter_scale) > 1.0

    def test_g4_3_c5_form_host_encoder_counts_are_the_float_reference_and_parity_holds(self):
        """On the residual form with a subsumed float-input encoder: the deployed host op's counts equal the float64
        reference (today rounded on the lattice 2T of the default parameter_scale), and NF-SCM parity reads 0 flips."""
        for seed in (0, 1):
            flow, perceptrons = _form(seed)
            _core_projection(perceptrons)
            program = _scm_program(flow, perceptrons)
            encoder = perceptrons[0]
            assert encoder.is_encoding_layer
            x = _inputs(400, seed)
            assert torch.equal(_encoder_counts_in_plane(encoder, x), _float64_window_counts(encoder, x))
            for semantics in ("streamed_lif", "windowed_lif"):
                mismatches, compared, out_diff = compare(program, semantics, n=24, seed=seed)
                assert compared > 0 and sum(mismatches.values()) == 0, (seed, semantics, mismatches)
                assert out_diff < 1e-3, out_diff  # a count apart is 1.0 (the parity tests' bound)


    def test_g4_4_an_entry_quantizer_at_another_window_is_not_the_entry_grid(self):
        """The 1/T entry grid is the window's: an entry quantizer at another window length puts the input on another
        grid, so the encoder carries no s T lattice; at the window's own length it does."""
        for window, expected in ((T + 1, None), (T, "sT")):
            encoder = _encoder_perceptron(grid=False)
            encoder.append_input_wire_op(ChipInputQuantizer(T=window, activation_scale=1.0))
            NormalizationAwarePerceptronQuantization(WQ_BITS, "cpu", rate=1.0).transform(encoder)
            arm_integer_membrane_lattice(_Holder(encoder))
            lattice = unwrap_lif_activation(encoder.activation).if_node.lattice_scale
            if expected is None:
                assert lattice is None, window
            else:
                assert lattice == 2.0 * float(encoder.parameter_scale) * T


# ---------------------------------------------------------------------------------------------------------------------
# G2: the published step reads of an integer program read inside the measurement plane
# ---------------------------------------------------------------------------------------------------------------------


def _quantized_form(seed: int = 0):
    flow, perceptrons = _form(seed)
    _step_projection(flow)
    return flow, perceptrons


class TestG2PublishedReads:
    def test_g2_1_wq_qv_and_nf_published_reads_of_one_program_are_equal(self):
        """Weight Quantization, Quantization Verification and Normalization Fusion publish reads of one integer
        program (NF's of its fused form): equal, prediction for prediction (today the fused and unfused float paths
        decide exact ties differently)."""
        from mimarsinan.pipelining.pipeline_steps.quantization.normalization_fusion_step import NormalizationFusionStep
        from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
            QuantizationVerificationStep,
        )
        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep

        for seed in (0, 1, 2):
            flow, _ = _quantized_form(seed)
            x = _inputs(seed=seed)
            labels = _exact_reference(flow, x).argmax(1)
            fused = copy.deepcopy(flow)
            for p in fused.get_perceptrons():
                fuse_into_perceptron(p, device="cpu")
            pipeline = _Pipeline({"device": "cpu"})
            wq = WeightQuantizationStep(pipeline)
            wq.tuner = SimpleNamespace(trainer=_Reader(flow, x, labels))
            qv = QuantizationVerificationStep(pipeline)
            qv.trainer = _Reader(flow, x, labels)
            nf = NormalizationFusionStep(pipeline)
            nf.trainer = _Reader(fused, x, labels)
            reads = [wq.pipeline_metric(), qv.pipeline_metric(), nf.validate(), nf.pipeline_metric()]
            assert reads[0] == 1.0 and len(set(reads)) == 1, (seed, reads)
            outs = [wq.tuner.trainer.outputs[0], qv.trainer.outputs[0], *nf.trainer.outputs]
            preds = [o.argmax(1) for o in outs]
            assert all(torch.equal(preds[0], p) for p in preds[1:])

    def test_g2_2_the_published_read_is_the_integer_reference_and_not_without_the_arming(self):
        """A toy integer program with exact threshold ties: the published read equals the integer reference count for
        count; the same model read outside the plane (no arming) differs."""
        from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
            QuantizationVerificationStep,
        )

        flow, perceptrons = _quantized_form(0)
        x = _inputs()
        reference = _exact_reference(flow, x)
        theta_out = float(perceptrons[-1].activation_scale)
        for p in perceptrons:  # a tuning stage recreated the activations: no lattice armed on the live nodes
            unwrap_lif_activation(p.activation).set_membrane_lattice(None)
        step = QuantizationVerificationStep(_Pipeline({"device": "cpu"}))
        step.trainer = _Reader(flow, x, reference.argmax(1))
        assert step.pipeline_metric() == 1.0
        assert step.trainer.planes == [True]
        assert torch.equal(_counts(step.trainer.outputs[0], theta_out), _counts(reference, theta_out))
        # The read leaves the live nodes as it found them (the tuning plane is untouched).
        assert all(unwrap_lif_activation(p.activation).if_node.lattice_scale is None for p in perceptrons)
        with torch.no_grad():
            free = _walk(flow, x)
        assert not torch.equal(_counts(free, theta_out), _counts(reference, theta_out))

    def test_g2_2b_a_float_program_reads_as_it_is(self):
        """The published read moves only an integer program: a model the projection never reached reads outside the
        plane, output for output (the float model's plane read differs from its free read by the walk's fold)."""
        from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
            QuantizationVerificationStep,
        )

        flow, _ = _form(0)
        x = _inputs(500)
        with torch.no_grad():
            free = _walk(flow, x)
        step = QuantizationVerificationStep(_Pipeline({"device": "cpu"}))
        step.trainer = _Reader(flow, x, free.argmax(1))
        assert step.pipeline_metric() == 1.0
        assert torch.equal(step.trainer.outputs[0], free)
        assert step.trainer.planes == [False]
        assert all(unwrap_lif_activation(p.activation).if_node.lattice_scale is None for p in flow.get_perceptrons())

    def test_g2_3_an_integer_program_reads_alike_on_every_evaluator(self):
        """One integer program's published read is equal on two evaluators of different float arithmetic (float32
        and float64 copies) and, where a GPU is present, on CPU and GPU (today the free plane's ties differ)."""
        from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
            QuantizationVerificationStep,
        )

        for seed in (0, 1):
            flow, perceptrons = _quantized_form(seed)
            x = _inputs(seed=seed)
            labels = _exact_reference(flow, x).argmax(1)
            theta_out = float(perceptrons[-1].activation_scale)
            reads = []
            evaluators = [(flow, x), (copy.deepcopy(flow).double(), x.double())]
            if torch.cuda.is_available():
                evaluators.append((copy.deepcopy(flow).cuda(), x.cuda()))
            for model, xs in evaluators:
                step = QuantizationVerificationStep(_Pipeline({"device": str(xs.device)}))
                step.trainer = _Reader(model, xs, labels.to(xs.device))
                metric = step.pipeline_metric()
                reads.append((metric, _counts(step.trainer.outputs[0].cpu(), theta_out)))
            assert all(r[0] == reads[0][0] for r in reads), (seed, [r[0] for r in reads])
            assert all(torch.equal(r[1], reads[0][1]) for r in reads[1:])


    @pytest.mark.parametrize("cap", [0, 300])
    def test_g2_4_a_pipeline_trainers_subsample_read_is_in_the_plane(self, cap):
        """The pipeline's trainers carry a subsample read: the step metric of an integer program reads inside the
        measurement plane on both of its branches (no cap: the full read; a cap: the seeded subsample)."""
        from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
            QuantizationVerificationStep,
        )

        class _SubsampleReader(_Reader):
            def test_on_subsample(self, *, max_samples, seed):
                self.planes.append(("subsample", in_measurement_plane()))
                return self.test()

        flow, _ = _quantized_form(0)
        x = _inputs(400)
        step = QuantizationVerificationStep(_Pipeline({**_qv_config(), "eval_max_samples": cap}))
        step.trainer = _SubsampleReader(flow, x, _exact_reference(flow, x).argmax(1))
        assert step.pipeline_metric() == 1.0
        expected = [True] if cap == 0 else [("subsample", True), True]
        assert step.trainer.planes == expected


# ---------------------------------------------------------------------------------------------------------------------
# G3: the projection and its verification apply to the perceptrons the placement maps to cores
# ---------------------------------------------------------------------------------------------------------------------


def _qv_config() -> dict:
    from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints

    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({"weight_bits": WQ_BITS, "device": "cpu", "spiking_family": "lif", "spiking_variant": "streamed",
                "simulation_steps": T, "weight_quantization": True})
    return cfg


def _run_qv(monkeypatch, flow) -> None:
    import mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step as qv_module

    monkeypatch.setattr(qv_module, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    qv_module.QuantizationVerificationStep(_Pipeline(_qv_config(), {"model": flow})).process()


def _parameters(p):
    return {k: v.detach().clone() for k, v in p.named_parameters()}


class TestG3CoreProjection:
    def test_g3_1_a_subsumed_encoder_keeps_its_float_weights_and_verification_passes(self, monkeypatch):
        flow, perceptrons = _form(0)
        encoder = perceptrons[0]
        assert encoder.is_encoding_layer
        before = _parameters(encoder)
        _step_projection(flow)
        after = _parameters(encoder)
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert getattr(encoder, "membrane_lattice_kind", None) is None
        for p in perceptrons[1:]:
            assert float(p.parameter_scale) > 1.0
        _run_qv(monkeypatch, flow)

    def test_g3_2_under_offload_the_encoder_is_projected(self, monkeypatch):
        flow, perceptrons = _form(0, placement="offload")
        first = perceptrons[0]
        assert not first.is_encoding_layer
        _step_projection(flow)
        assert float(first.parameter_scale) > 1.0
        _run_qv(monkeypatch, flow)

    def test_g3_3_parity_with_a_float_host_encoder(self):
        """The step's projection (the encoder float) then Soft Core Mapping's sequence: NF-SCM parity reads 0 flips on
        both LIF semantics, and the host encoder's counts are its float64 reference (no lattice on a float encoder)."""
        for seed in (0, 1):
            flow, perceptrons = _form(seed)
            encoder = perceptrons[0]
            before = _parameters(encoder)
            _step_projection(flow)
            assert all(torch.equal(before[k], v) for k, v in _parameters(encoder).items())
            program = _scm_program(flow, perceptrons)
            x = _inputs(400, seed)
            assert torch.equal(_encoder_counts_in_plane(encoder, x), _float64_window_counts(encoder, x))
            for semantics in ("streamed_lif", "windowed_lif"):
                mismatches, compared, out_diff = compare(program, semantics, n=24, seed=seed)
                assert compared > 0 and sum(mismatches.values()) == 0, (seed, semantics, mismatches)
                assert out_diff < 1e-3, out_diff  # a count apart is 1.0 (the parity tests' bound)


    def test_g3_4_the_verifications_grid_report_lists_the_perceptrons_on_cores(self, monkeypatch):
        """The verification's per-layer grid report is over the perceptrons it verifies: those on cores, the subsumed
        encoder left out (it keeps its float weights and runs on the host)."""
        import mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step as qv_module

        flow, perceptrons = _quantized_form(0)
        monkeypatch.setattr(qv_module, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
        pipeline = _Pipeline(_qv_config(), {"model": flow})
        qv_module.QuantizationVerificationStep(pipeline).process()
        reports = [payload for kind, payload in pipeline.events if kind == "quantization_report"]
        cores = [p for p in perceptrons if not p.is_encoding_layer]
        assert len(reports) == 1 and len(reports[0]["layers"]) == len(cores) == len(perceptrons) - 1


# ---------------------------------------------------------------------------------------------------------------------
# G1: the host normalizations re-estimated at the end of Weight Quantization
# ---------------------------------------------------------------------------------------------------------------------


def _host_bns(flow):
    owned = {id(m) for p in flow.get_perceptrons() for m in p.modules()}
    return [m for m in flow.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and id(m) not in owned]


def _training_batches(k: int, n: int = 64, seed: int = 11, shift: float = 0.6):
    g = torch.Generator().manual_seed(seed)
    return [2.0 * torch.randn(n, 8, generator=g) + shift for _ in range(k)]


def _bn_input_stats(flow, bn, batches):
    """The BatchNorm's input statistics over ``batches`` through the model's own forward: the mean of the batch means
    and of the unbiased batch variances (the cumulative running average a reset BatchNorm keeps with momentum None)."""
    captured = []
    handle = bn.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach().clone()))
    try:
        with torch.no_grad():
            for x in batches:
                flow(x)
    finally:
        handle.remove()
    means = torch.stack([c.mean(0) for c in captured])
    variances = torch.stack([c.var(0, unbiased=True) for c in captured])
    return means.mean(0), variances.mean(0)


class TestG1HostNormalizations:
    def test_g1_1_the_reestimated_statistics_are_the_projected_networks_batch_statistics(self, monkeypatch):
        from mimarsinan.pipelining.pipeline_steps.quantization.host_normalization_recalibration import (
            host_normalizations,
            reestimate_host_normalizations,
        )

        flow, perceptrons = _quantized_form(0)
        bns = _host_bns(flow)
        assert [m for _, m in host_normalizations(flow)] == bns and len(bns) == 1
        batches = _training_batches(6)
        mean, var = _bn_input_stats(flow, bns[0], batches)
        codes = {id(p): _parameters(p) for p in perceptrons}
        report = reestimate_host_normalizations(flow, iter(batches), calibration_read=lambda: 1.0, device="cpu")
        assert report.kept and report.batches == 6 and report.modules == 1
        assert torch.allclose(bns[0].running_mean, mean, atol=1e-5)
        assert torch.allclose(bns[0].running_var, var, atol=1e-5)
        assert not bns[0].training
        for p in perceptrons:
            assert all(torch.equal(codes[id(p)][k], v) for k, v in _parameters(p).items())
        _run_qv(monkeypatch, flow)

    def test_g1_2_kept_only_if_the_calibration_read_does_not_fall(self):
        """Keep-best: a re-estimation whose calibration read falls is restored (the read after equals the entry's), one
        whose read holds or rises is kept; zero declared batches leave the model as it is."""
        from mimarsinan.pipelining.pipeline_steps.quantization.host_normalization_recalibration import (
            reestimate_host_normalizations,
        )

        for reads, kept in (((0.5, 0.4), False), ((0.5, 0.5), True), ((0.5, 0.6), True)):
            flow, _ = _quantized_form(0)
            bn = _host_bns(flow)[0]
            entry = {k: v.clone() for k, v in bn.state_dict().items()}
            sequence = iter(reads)
            report = reestimate_host_normalizations(flow, iter(_training_batches(4)),
                                                    calibration_read=lambda: next(sequence), device="cpu")
            assert report.kept is kept and (report.before, report.after) == reads
            same = all(torch.equal(entry[k], v) for k, v in bn.state_dict().items())
            assert same is (not kept)
        flow, _ = _quantized_form(0)
        bn = _host_bns(flow)[0]
        entry = {k: v.clone() for k, v in bn.state_dict().items()}
        report = reestimate_host_normalizations(flow, iter(()), calibration_read=lambda: 0.5, device="cpu")
        assert not report.kept and all(torch.equal(entry[k], v) for k, v in bn.state_dict().items())

    def test_g1_2b_the_step_reestimates_at_its_commit_over_the_declared_batches(self, monkeypatch):
        """At the end of Weight Quantization (its commit) the step re-estimates over the declared batch count (the
        documented default when undeclared), reads the calibration read before and after, and commits the model."""
        import mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step as wq_module
        from mimarsinan.pipelining.pipeline_steps.quantization.host_normalization_recalibration import (
            DEFAULT_HOST_NORMALIZATION_BATCHES,
            HOST_NORMALIZATION_BATCHES_KEY,
        )

        assert DEFAULT_HOST_NORMALIZATION_BATCHES == 60
        for declared, expected in ((None, DEFAULT_HOST_NORMALIZATION_BATCHES), (3, 3), (0, 0)):
            flow, _ = _quantized_form(0)
            bn = _host_bns(flow)[0]
            entry = bn.running_mean.clone()
            drawn = []
            batches = iter(_training_batches(80))

            def next_training_batch():
                drawn.append(1)
                return next(batches), None

            config = {"device": "cpu"}
            if declared is not None:
                config[HOST_NORMALIZATION_BATCHES_KEY] = declared
            pipeline = _Pipeline(config)
            step = wq_module.WeightQuantizationStep(pipeline)
            step._accessed_entries, step._updated_entries = set(), set()
            step.tuner = SimpleNamespace(trainer=SimpleNamespace(next_training_batch=next_training_batch), model=flow)
            reads, planes = iter((0.5, 0.75)), []

            def calibration_read(tuner):
                planes.append(in_measurement_plane())
                return next(reads)

            monkeypatch.setattr(wq_module, "live_model_acc_fp32", calibration_read)
            step._commit_tuner_entries(flow, object())
            assert pipeline.entries["model"] is flow
            assert len(drawn) == expected
            assert planes == ([True, True] if expected else [])  # the integer program's reads, in its plane
            assert torch.equal(bn.running_mean, entry) is (expected == 0)

    def test_g1_3_parity_after_reestimation(self):
        """NF-SCM parity reads 0 flips after the host normalizations are re-estimated."""
        from mimarsinan.pipelining.pipeline_steps.quantization.host_normalization_recalibration import (
            reestimate_host_normalizations,
        )

        for seed in (0, 1):
            flow, perceptrons = _quantized_form(seed)
            bn = _host_bns(flow)[0]
            entry = bn.running_mean.clone()
            report = reestimate_host_normalizations(flow, iter(_training_batches(5, seed=seed)),
                                                    calibration_read=lambda: 1.0, device="cpu")
            assert report.kept and not torch.equal(bn.running_mean, entry)
            program = _scm_program(flow, perceptrons)
            for semantics in ("streamed_lif", "windowed_lif"):
                mismatches, compared, out_diff = compare(program, semantics, n=24, seed=seed)
                assert compared > 0 and sum(mismatches.values()) == 0, (seed, semantics, mismatches)
                assert out_diff < 1e-3, out_diff  # a count apart is 1.0 (the parity tests' bound)


    def test_g1_4_every_host_normalization_is_reestimated_and_its_momentum_returns(self):
        """A form with two host normalizations, the second reading the first's join: both are reset and re-estimated
        over the drawn batches (each counts exactly those batches), the first equals its batch statistics, and each
        one's momentum and mode return to what they were."""
        from mimarsinan.pipelining.pipeline_steps.quantization.host_normalization_recalibration import (
            host_normalizations,
            reestimate_host_normalizations,
        )

        flow, _ = _form(0, module=_TwoResidualBN)
        _step_projection(flow)
        bns = _host_bns(flow)
        assert [m for _, m in host_normalizations(flow)] == bns and len(bns) == 2
        entry = [(m.momentum, m.training, m.running_mean.clone()) for m in bns]
        batches = _training_batches(5)
        mean, var = _bn_input_stats(flow, bns[0], batches)
        report = reestimate_host_normalizations(flow, iter(batches), calibration_read=lambda: 1.0, device="cpu")
        assert report.kept and report.modules == 2 and report.batches == 5
        for m, (momentum, training, running_mean) in zip(bns, entry):
            assert int(m.num_batches_tracked) == 5
            assert not torch.equal(m.running_mean, running_mean)
            assert m.momentum == momentum and m.training == training
        assert torch.allclose(bns[0].running_mean, mean, atol=1e-5)
        assert torch.allclose(bns[0].running_var, var, atol=1e-5)

    def test_g1_5_the_ledger_and_the_high_water_hold_the_reestimated_exit(self, monkeypatch, tmp_path):
        """The step's commit moves the model after the tuner's exit read: the retention ledger's exit and the D-hat
        high-water hold the read of the model it commits (the re-estimation's calibration read), not the tuner's."""
        import json

        import mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step as wq_module
        from mimarsinan.tuning.orchestration import dhat_highwater

        flow, _ = _quantized_form(0)
        batches = iter(_training_batches(10))
        pipeline = _Pipeline({"device": "cpu", "wq_host_normalization_batches": 4}, working_directory=str(tmp_path))
        step = wq_module.WeightQuantizationStep(pipeline)
        step._accessed_entries, step._updated_entries = set(), set()
        step.tuner = SimpleNamespace(trainer=SimpleNamespace(next_training_batch=lambda: (next(batches), None)),
                                     model=flow, exit_metric_estimate=lambda: 0.5)
        reads = iter((0.5, 0.75))
        monkeypatch.setattr(wq_module, "live_model_acc_fp32", lambda tuner: next(reads))
        step._commit_tuner_entries(flow, object())
        assert dhat_highwater.peek(pipeline) == 0.75
        ledger = json.loads((tmp_path / "retention_ledger.json").read_text())
        assert ledger["entries"][-1]["exit_metric"] == 0.75
