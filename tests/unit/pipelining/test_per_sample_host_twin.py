"""[U-PS P-1] The parity gates with the per-sample host-kernel evaluation on a hybrid program with fp32 host ops.

The CIFAR fixture: a (3, 32, 32) input, the encoding convolution subsumed on the host (an fp32 host convolution whose
values every chip core downstream reads, the C5 T-LIF host form), a residual block whose second convolution, add and
relu run on the host, a chip convolution that reads the join, and an fp32 classifier matmul on the host (the ``mms``
kernel U-TT's T-6 names). It is built by the Soft Core Mapping sequence of the parity unit's fixture
(``tests/unit/spiking/encoding_skip_program.py``) and read on both twins: the NF segment walk and the deployed flow.

With ``host_kernel_evaluation = per_sample`` declared on the program, every host kernel of the deployed program runs
one sample at a time and the NF twin follows the program's mode wherever a gate compares it with the program, so the
gates stay exact at atol 0 (fixed, never budgeted), and the host condition reads established by construction where on
the batch it reads outside scope (the fp32 kernels' measured class on this host; a host where they are batch-invariant
skips by name)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.host_condition import ESTABLISHED, OUTSIDE_SCOPE, host_condition_of
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.host_kernel_evaluation import (
    PER_SAMPLE,
    declare_host_kernel_evaluation,
    forward_of,
    program_evaluates_per_sample,
    twin_as_program,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.nf_scm_parity import (
    assert_streamed_nf_scm_exact_or_raise,
    assert_torch_vs_deployed_sim_parity_or_raise,
)
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

T = 8
SHAPE = (3, 32, 32)
C = 4
N_CLASSES = 10
THETAS = (0.5, 1.0)
THETA_ENC = 2.185


class _CifarBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem, self.r0 = nn.Conv2d(3, C, 3, stride=2, padding=1), nn.ReLU()
        self.c1, self.r1 = nn.Conv2d(C, C, 3, padding=1), nn.ReLU()
        self.c2, self.r2 = nn.Conv2d(C, C, 3, padding=1), nn.ReLU()
        self.c3, self.r3 = nn.Conv2d(C, C, 3, stride=2, padding=1), nn.ReLU()
        self.fc = nn.Linear(C * 8 * 8, N_CLASSES)

    def forward(self, x):
        h = self.r0(self.stem(x))
        z = self.r2(h + self.c2(self.r1(self.c1(h))))
        return self.fc(self.r3(self.c3(z)).flatten(1))


class _NF(nn.Module):
    """The torch twin as the pipeline holds it after LIF adaptation: ``forward`` IS the NF segment walk (streamed, or
    the synchronized two-window walk the T-LIF rows of route (b) run)."""

    def __init__(self, flow, synchronized: bool = False):
        super().__init__()
        self.repr_ = flow.get_mapper_repr()
        self._perceptrons = nn.ModuleList(list(self.repr_.get_perceptrons()))
        self.synchronized = synchronized

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        policy = LifSegmentPolicy(synchronized=self.synchronized, soma_law=DEFAULT_SOMA_LAW)
        return SegmentForwardDriver(self.repr_, T, policy)(x)


def _program(weight_bits=None):
    torch.manual_seed(0)
    net = _CifarBlock()
    with torch.no_grad():  # a live cascade: every core's counts span the window
        for module, bias in ((net.stem, 0.3), (net.c1, 0.2), (net.c2, 0.1), (net.c3, 0.2)):
            module.weight.mul_(2.0)
            module.bias.fill_(bias)
    flow = convert_torch_model(net, SHAPE, N_CLASSES, device="cpu", encoding_layer_placement="subsume").eval()
    perceptrons = list(flow.get_perceptrons())
    for p, theta in zip(perceptrons, (THETA_ENC, *THETAS)):
        p.set_activation_scale(torch.tensor(float(theta)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    q_max = 127.0
    if weight_bits is not None:
        napq = NormalizationAwarePerceptronQuantization(weight_bits, "cpu", rate=1.0, two_scale=False)
        for p in perceptrons:
            napq.transform(p)
        q_max = float(2 ** (weight_bits - 1) - 1)
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=q_max, firing_mode="Default", max_axons=64, max_neurons=64).map(repr_)
    if weight_bits is not None:
        quantize_ir_graph(ir, weight_bits, weight_quantization=True)
        arm_integer_membrane_lattice(flow)
    return flow, ir


def _pipeline(weight_bits=None):
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": T,
                "input_shape": SHAPE, "device": "cpu", "weight_quantization": weight_bits is not None})
    return SimpleNamespace(config=cfg)


def _deployed(ir, weight_bits=None, synchronized=False):
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=False)
    extra = dict(lif_execution_synchronized=True, pass_transfer=COLLAPSE) if synchronized else dict(pass_transfer=VERBATIM)
    return SpikingHybridCoreFlow(SHAPE, hybrid, simulation_length=T, thresholding_mode="<=",
                                 membrane_integer_lattice=weight_bits is not None, spiking_mode="lif",
                                 cycle_accurate_lif_forward=True, **extra).eval()


def _samples(n=8):
    torch.manual_seed(1)
    return 2.0 * torch.rand(n, *SHAPE)


def _host_modules(ir):
    return {op.name: op.params["module"] for op in ir.nodes if isinstance(op, ComputeOp)}


def _leaf_kernels(ir):
    """The fp32 convolution and matmul leaves of the host ops, by op name: the module objects both twins call (the
    host ``c2`` convolution inside its wrapper and the ``fc_col0`` classifier; the stem's mapper computes its
    convolution functionally, so no leaf of it is called)."""
    leaves = {}
    for name, module in _host_modules(ir).items():
        for leaf in module.modules():
            if isinstance(leaf, (nn.Conv2d, nn.Linear)):
                leaves.setdefault(name, leaf)
    return leaves


def _batch_sizes(modules):
    """A pre-hook on each module: the batch of every call, by op name."""
    seen = {name: [] for name in modules}
    handles = [m.register_forward_pre_hook(lambda _m, args, _n=name: seen[_n].append(int(args[0].shape[0])))
               for name, m in modules.items()]
    return seen, handles


@pytest.fixture(params=[None, 4], ids=["float_weights", "wq4"])
def program(request):
    flow, ir = _program(request.param)
    host = _host_modules(ir)
    assert {"stem", "c2", "fc_col0"} <= set(host), sorted(host)
    return request.param, flow, ir


def _fp32_host_kernels_batch_dependent_here(flow, ir, weight_bits):
    with torch.no_grad():
        certificate = host_condition_of(_deployed(ir, weight_bits), _samples())
    return certificate.status == OUTSIDE_SCOPE, certificate


class TestTheHostConditionOnTheCifarFixture:
    def test_outside_scope_with_the_key_off_and_established_by_construction_with_it_on(self, program):
        weight_bits, flow, ir = program
        dependent, off = _fp32_host_kernels_batch_dependent_here(flow, ir, weight_bits)
        if not dependent:
            pytest.skip(f"the fixture's fp32 host kernels are batch-invariant on this host: {off.verdict()[:200]}")
        assert {k.kind for k in off.batch_dependent} & {"mms"}, [k.label() for k in off.batch_dependent]
        assert declare_host_kernel_evaluation(ir, PER_SAMPLE) == len(_host_modules(ir))
        with torch.no_grad():
            on = host_condition_of(_deployed(ir, weight_bits), _samples())
        assert on.status == ESTABLISHED and on.passed, on.verdict()
        assert set(on.per_sample_ops) == {k.op_id for k in on.kernels}
        assert {k.name for k in off.batch_dependent} <= {k.name for k in on.kernels}


class TestEveryParityGatePassesWithTheKeyOn:
    def test_the_streamed_exactness_gate_passes_at_atol_0_and_both_twins_see_one_sample_per_host_call(self, program):
        weight_bits, flow, ir = program
        assert_streamed_nf_scm_exact_or_raise(_pipeline(weight_bits), _NF(flow), ir, _samples())  # the key off
        declare_host_kernel_evaluation(ir, PER_SAMPLE)
        assert program_evaluates_per_sample(ir)
        seen, handles = _batch_sizes(_leaf_kernels(ir))
        try:
            assert_streamed_nf_scm_exact_or_raise(_pipeline(weight_bits), _NF(flow), ir, _samples())
        finally:
            for handle in handles:
                handle.remove()
        assert seen["c2"] and seen["fc_col0"], seen
        assert {name: sorted(set(sizes)) for name, sizes in seen.items() if sizes} == {
            name: [1] for name, sizes in seen.items() if sizes}

    @pytest.mark.parametrize("synchronized", [False, True], ids=["streamed", "synchronized"])
    def test_the_torch_versus_deployed_gate_agrees_on_every_sample_with_the_key_on(self, program, synchronized):
        """The decision-level NF gate every LIF semantics runs (the T-LIF rows of route (b) are synchronized)."""
        weight_bits, flow, ir = program
        declare_host_kernel_evaluation(ir, PER_SAMPLE)
        seen, handles = _batch_sizes(_leaf_kernels(ir))
        try:
            agreement = assert_torch_vs_deployed_sim_parity_or_raise(
                _NF(flow, synchronized), _deployed(ir, weight_bits, synchronized), _samples(), min_agreement=1.0)
        finally:
            for handle in handles:
                handle.remove()
        assert agreement == 1.0
        assert {name: sorted(set(sizes)) for name, sizes in seen.items() if sizes} == {
            name: [1] for name, sizes in seen.items() if sizes} and seen["c2"], seen

    def test_the_twin_does_not_follow_a_program_that_declares_nothing(self, program):
        weight_bits, flow, ir = program
        seen, handles = _batch_sizes(_leaf_kernels(ir))
        try:
            assert_streamed_nf_scm_exact_or_raise(_pipeline(weight_bits), _NF(flow), ir, _samples())
        finally:
            for handle in handles:
                handle.remove()
        assert 8 in seen["c2"], "absent the key, the NF twin runs its host kernels on its batch as before"


class TestTheTwinFollowsAtTheHostValues:
    def test_the_deployed_host_values_are_the_per_sample_twin_s_bit_for_bit(self, program):
        """The NF twin must follow: on its batch its fp32 host convolution (``c2``, whose values a chip core reads
        through the join) gives other bits than the program's per-sample kernel; one sample at a time it gives the
        program's, bit for bit."""
        weight_bits, flow, ir = program
        declare_host_kernel_evaluation(ir, PER_SAMPLE)
        conv = _leaf_kernels(ir)["c2"]
        outputs = {"deployed": [], "nf": [], "batched": []}
        side = {"now": "deployed"}
        handle = conv.register_forward_hook(lambda _m, _a, out: outputs[side["now"]].append(out.detach().clone()))
        x = _samples()
        try:
            with torch.no_grad(), measurement_plane():  # the plane both twins are read in by every gate
                _deployed(ir, weight_bits)(x)
                side["now"] = "nf"
                twin_as_program(ir, forward_of, _NF(flow), x)
                side["now"] = "batched"
                _NF(flow)(x)
        finally:
            handle.remove()
        deployed = torch.cat(outputs["deployed"]).reshape(x.shape[0], -1)
        nf = torch.cat(outputs["nf"]).reshape(x.shape[0], -1)
        batched = torch.cat(outputs["batched"]).reshape(x.shape[0], -1)
        assert torch.equal(deployed, nf)
        if torch.equal(batched, nf):
            pytest.skip("the stem convolution is batch-invariant on this host at this batch")
        assert int((batched != nf).sum()) > 0
        assert np.isfinite(deployed.numpy()).all()
