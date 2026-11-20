"""Host-gather program forms beyond the encoding skip, for the NF to SCM window-count tests (U-SC).

Each form is a torch block converted with the encoding layer subsumed on the host, built by the Soft Core Mapping sequence
(per-source scales with the wire value ops armed, boundary scales, optionally the Weight Quantization step's NAPQ with
``quantize_ir_graph`` and the integer membrane lattice, the identity program), and read on both twins: the NF segment walk
inside the measurement plane (the decision gate's plane) and the deployed flow (``SpikingHybridCoreFlow``, per-sample
records). The deployed records evaluate every host kernel on one sample, and fp32 host kernels are batch-dependent at
the ulp level (Theorem 1's host condition), so the program declares ``host_kernel_evaluation = per_sample`` and the NF
twin follows the program's mode (``twin_as_program``), as the parity gate of record does (U-SC2 targeted pass, B1: a
batched NF moved a host op's rate across a comb's half count where the deployed record did not). Every core perceptron's window counts are compared as sorted rows per sample (the gate's convention). The forms
are the shakedown's (SD-10c): the C5 residual control, a subsumed encoding concatenated with a core output (read by a
core, and by a host linear), PyramidNet's channel-padded shortcut, a projection shortcut, a max or an average pool on the
encoding before a join, and a pool before a strided block with a strided projection shortcut; plus the minimal mixed
concatenation of an ABSOLUTE host chain with a WIRE core output (DenseNet-BC's published order), a DenseNet-BC
program at the smallest depth, and that concatenation nested inside a concatenation that also takes core outputs (read by
a host op, and by a core). U-SC3's memory forms (``MEMORY_FORMS``) add a wide encoding read by a narrow core, DenseNet-BC's
dense-unit read (a wide host value read by a narrow core), a wide core whose output only a host op reads, and chains of
two and six cores in one segment. U-DOOR's token forms (``TOKEN_FORMS``) map one mapper to several IR nodes per sample:
a token-wise subsumed encoding (one host node per token) read by token-wise cores, the same encoding beside a host add
over the whole token value (one host node), and t2_04's torchvision ViT at its smallest size (host LayerNorms,
attention and adds over the whole token value, its MLP's first Linear on chip per token).

U-IN adds the encoding layer's placement (``offload``: the first perceptron a core reading the raw input), the forms whose
raw input a core and an armed host op read at once, or a core reads concatenated with a core's output (``INPUT_FORMS``),
depth-balancing relays, and the pruned programs of the parity gate's
survival projection: output channels killed (zeroed, and declared pruned on the layer for the model-mask path), IR
pruning before the build, t0_51's network family (``lenet5``) and the mapping's core limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.depth_balancing import insert_depth_balancing_relays
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation, twin_as_program
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.nf_scm_parity import _group_record_by_perceptron
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

T = 15
WQ_BITS = 4
N_OUT = 6

#: ``name -> (re-timed level stages, flow kwargs, NF policy)``: the two LIF semantics a deployment selects.
SEMANTICS: Dict[str, Tuple[bool, dict, Callable[[], LifSegmentPolicy]]] = {
    "streamed_lif": (False, dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM),
                     lambda: LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW)),
    "windowed_lif": (True, dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE),
                     lambda: LifSegmentPolicy(retime=True, soma_law=DEFAULT_SOMA_LAW)),
}


class EncodingSkip(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        z = self.r2(s + self.fc2(self.r1(self.fc1(s))))
        return self.r3(self.fc3(z))


class DenseConcat(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(12, 6)
        self.r0, self.r1, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        y = self.r1(self.fc1(s))
        return self.r3(self.fc3(torch.cat([s, y], dim=1)))


class DenseConcatHost(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(12, 6), nn.Linear(6, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        y = self.r1(self.fc1(s))
        h = self.fc2(torch.cat([s, y], dim=1))  # a host op: no activation of its own
        return self.r3(self.fc3(self.r2(h + s)))


class PyramidPad(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 8), nn.Linear(8, 8), nn.Linear(8, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        y = self.fc2(self.r1(self.fc1(s)))
        return self.r3(self.fc3(self.r2(y + F.pad(s, (0, 2)))))


class Projection(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2 = nn.Linear(8, 6), nn.Linear(6, 8), nn.Linear(8, 8)
        self.proj, self.fc3 = nn.Linear(6, 8), nn.Linear(8, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        y = self.fc2(self.r1(self.fc1(s)))
        return self.r3(self.fc3(self.r2(y + self.proj(s))))


class PoolJoin(nn.Module):
    shape = (3, 8, 8)

    def __init__(self, pool: str):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)
        self.c1, self.c2 = nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        p = self.pool(self.r0(self.stem(x)))
        z = self.r2(self.c2(self.r1(self.c1(p))) + p)
        return self.r3(self.fc(z.flatten(1)))


class PoolDownsample(nn.Module):
    shape = (3, 8, 8)

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.c1 = nn.Conv2d(4, 6, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(6, 6, 3, padding=1)
        self.down = nn.Conv2d(4, 6, 1, stride=2)
        self.fc = nn.Linear(6 * 2 * 2, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        p = self.pool(self.r0(self.stem(x)))
        z = self.r2(self.c2(self.r1(self.c1(p))) + self.down(p))
        return self.r3(self.fc(z.flatten(1)))


class AbsoluteConcat(nn.Module):
    """DenseNet-BC's mixed concatenation at minimum size: an ABSOLUTE host chain (a stem without activation, rooted at the
    input) concatenated with a WIRE core output, read by a core."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(12, 6)
        self.r1, self.r3 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.stem(x.flatten(1))
        y = self.r1(self.fc1(s))
        return self.r3(self.fc3(torch.cat([s, y], dim=1)))


class AbsoluteConcatHost(nn.Module):
    """The same mixed concatenation read by a host affine op before a core (DenseNet-BC's ``bn1``)."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(12, 6)
        self.bn = nn.BatchNorm1d(12)
        self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.stem(x.flatten(1))
        y = self.r1(self.fc1(s))
        return self.r3(self.fc3(self.r2(self.bn(torch.cat([s, y], dim=1)))))


class NestedMixedGather(nn.Module):
    """A host-side mixed concatenation (the absolute stem with a host op over a core output: no source carries spikes)
    nested inside a concatenation that also takes core outputs (so it runs inside a neural segment), read by a host
    affine op only."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(8, 6)
        self.fc1, self.fc2, self.fc4, self.fc3 = nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(18, 6)
        self.bn, self.bn2 = nn.BatchNorm1d(6), nn.BatchNorm1d(18)
        self.r1, self.r2, self.r3, self.r4 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def gathers(self, x):
        s = self.stem(x.flatten(1))
        inner = torch.cat([s, self.bn(self.r1(self.fc1(s)))], dim=1)
        return torch.cat([inner, self.r4(self.fc4(self.r2(self.fc2(s))))], dim=1)

    def forward(self, x):
        return self.r3(self.fc3(self.bn2(self.gathers(x))))


class NestedMixedGatherCore(NestedMixedGather):
    """The same nesting with the outer concatenation read by a core."""

    def forward(self, x):
        return self.r3(self.fc3(self.gathers(x)))


class WideEncodingRead(nn.Module):
    """A wide subsumed encoding (48 channels) read by a narrow core (2 channels): the encoding layer's boundary train
    is the walk's largest tensor (U-SC3)."""

    shape = (3, 8, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.c1 = nn.Conv2d(3, 48, 3, padding=1), nn.Conv2d(48, 2, 1)
        self.fc = nn.Linear(2 * 8 * 8, 6)
        self.r0, self.r1, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.r3(self.fc(self.r1(self.c1(self.r0(self.stem(x)))).flatten(1)))


class WideConcatHostRead(nn.Module):
    """DenseNet-BC's dense-unit read at minimum size, wide: an absolute stem concatenated with a core output, through a
    host BatchNorm and ReLU, read by a narrow core (S1's ``conv1`` reading ``relu1(bn1(cat))``): the boundary re-encode
    of a host value is the walk's largest tensor (U-SC3)."""

    shape = (3, 8, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.c1, self.c3 = nn.Conv2d(3, 32, 1), nn.Conv2d(32, 4, 1), nn.Conv2d(36, 2, 1)
        self.bn = nn.BatchNorm2d(36)
        self.fc = nn.Linear(2 * 8 * 8, 6)
        self.r1, self.r2, self.r3, self.r4 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.stem(x)
        y = self.r1(self.c1(s))
        z = self.r3(self.c3(self.r2(self.bn(torch.cat([s, y], dim=1)))))
        return self.r4(self.fc(z.flatten(1)))


class WideCoreOutput(nn.Module):
    """A narrow subsumed encoding read by a wide core (48 channels) whose output only a host op reads: no later node of
    its segment reads that core's events (U-SC3)."""

    shape = (3, 8, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.c1, self.c2 = nn.Conv2d(3, 2, 3, padding=1), nn.Conv2d(2, 48, 1), nn.Conv2d(48, 2, 1)
        self.bn = nn.BatchNorm2d(48)
        self.fc = nn.Linear(2 * 8 * 8, 6)
        self.r0, self.r1, self.r2, self.r3, self.r4 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        h = self.r1(self.c1(self.r0(self.stem(x))))
        z = self.r3(self.c2(self.r2(self.bn(h))))
        return self.r4(self.fc(z.flatten(1)))


class DeepChain(nn.Module):
    """One neural segment of ``depth`` cores after the subsumed encoding, then a narrow head: every hop's train is read
    by the next hop only (U-SC3)."""

    shape = (3, 8, 8)

    def __init__(self, depth: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.convs = nn.ModuleList(nn.Conv2d(16, 16, 3, padding=1) for _ in range(depth))
        self.relus = nn.ModuleList(nn.ReLU() for _ in range(depth + 1))
        self.head, self.rh = nn.Conv2d(16, 2, 1), nn.ReLU()
        self.fc, self.rf = nn.Linear(2 * 8 * 8, 6), nn.ReLU()

    def forward(self, x):
        h = self.relus[0](self.stem(x))
        for conv, relu in zip(self.convs, self.relus[1:]):
            h = relu(conv(h))
        return self.rf(self.fc(self.rh(self.head(h)).flatten(1)))


class TokenSkip(nn.Module):
    """A token-wise subsumed encoding (one host node per token, keyed 0..5 each) read by token-wise cores."""

    shape = (4, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(24, 6)
        self.r0, self.r1, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x))
        z = self.r1(self.fc1(s))
        return self.r3(self.fc3(z.flatten(1)))


class TokenResidual(nn.Module):
    """The token-wise encoding beside a host add over the whole token value (one host node, keyed 0..23), both read by
    token-wise cores."""

    shape = (4, 8)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.fc2, self.fc3 = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(24, 6)
        self.r0, self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x))
        y = self.r1(self.fc1(s))
        z = self.r2(self.fc2(y + s))
        return self.r3(self.fc3(z.flatten(1)))


class InputCoreAndHost(nn.Module):
    """The raw input read by a core and by a host affine op (a stem without activation, U-SC2 F-2's absolute entry)
    whose output a second core reads, joined by host adds read by a third core (U-IN): under ``offload`` the first core
    folds the input at the boundary gauge while the host op, armed at that gauge, decodes the raw buffer at one."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.fc1, self.host, self.fch, self.fc3 = nn.Linear(8, 6), nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, 6)
        self.r1, self.r2, self.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        h = self.host(x)  # read twice, so no conversion folds it into ``fch``
        return self.r3(self.fc3(self.r2(self.fch(h)) + h + self.r1(self.fc1(x))))


class InputConcat(nn.Module):
    """The raw input concatenated with a core's output and read by a core (U-IN): a MIXED layout gather whose raw-input
    elements the reading core folds at the boundary gauge, each element at its producer's currency."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.fc1, self.fc3 = nn.Linear(8, 6), nn.Linear(14, 6)
        self.r1, self.r3 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        return self.r3(self.fc3(torch.cat([x, self.r1(self.fc1(x))], dim=1)))


def tiny_vit():
    """t2_04's network family (``torch_vit``: torchvision's VisionTransformer) at its smallest size."""
    from torchvision.models.vision_transformer import VisionTransformer

    return VisionTransformer(image_size=8, patch_size=4, num_layers=1, num_heads=2, hidden_dim=8, mlp_dim=16,
                             num_classes=N_OUT)


def lenet5():
    """t0_51's network family (the registered ``lenet5`` builder) at the MNIST shape."""
    import mimarsinan.models.builders  # noqa: F401 - registers the builders
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    return ModelRegistry.get_builder_cls("lenet5")("cpu", (1, 28, 28), N_OUT, {}).build({})


def small_densenet_bc():
    from mimarsinan.models.families.densenet_bc import CifarDenseNetBC

    return CifarDenseNetBC(input_shape=(3, 8, 8), num_classes=N_OUT, depth=16, growth_rate=4, compression=0.5,
                           layer_order="published")


FORMS: Dict[str, Callable[[], nn.Module]] = {
    "encoding_skip": EncodingSkip,
    "dense_concat": DenseConcat,
    "dense_concat_host": DenseConcatHost,
    "pyramid_pad": PyramidPad,
    "projection": Projection,
    "pool_join_max": lambda: PoolJoin("max"),
    "pool_join_avg": lambda: PoolJoin("avg"),
    "pool_downsample": PoolDownsample,
}
#: The eight forms of SD-10c's probe.
SHAKEDOWN_FORMS = tuple(FORMS)
MIXED_FORMS: Dict[str, Callable[[], nn.Module]] = {
    "absolute_concat": AbsoluteConcat,
    "absolute_concat_host": AbsoluteConcatHost,
    "nested_gather_host_read": NestedMixedGather,
    "nested_gather_core_read": NestedMixedGatherCore,
}
#: The forms whose largest train is a boundary re-encode or a chain of hop trains (U-SC3's memory bounds).
MEMORY_FORMS: Dict[str, Callable[[], nn.Module]] = {
    "wide_encoding_read": WideEncodingRead,
    "wide_concat_host_read": WideConcatHostRead,
    "wide_core_output": WideCoreOutput,
    "chain_2": lambda: DeepChain(2),
    "chain_6": lambda: DeepChain(6),
}
#: The forms that map one mapper to several IR nodes per sample (U-DOOR: the dithered encode's keys follow the IR).
TOKEN_FORMS: Dict[str, Callable[[], nn.Module]] = {
    "token_skip": TokenSkip,
    "token_residual": TokenResidual,
    "vit_tiny": tiny_vit,
}
#: U-IN's form beside the others: the raw input read by a core and by an armed host op at once.
INPUT_FORMS: Dict[str, Callable[[], nn.Module]] = {"input_core_host": InputCoreAndHost, "input_concat": InputConcat}
SHAPES = {"densenet_bc": (3, 8, 8), "vit_tiny": (3, 8, 8), "lenet5": (1, 28, 28)}


def make_form(form: str) -> nn.Module:
    if form == "densenet_bc":
        return small_densenet_bc()
    if form == "lenet5":
        return lenet5()
    return {**FORMS, **MIXED_FORMS, **MEMORY_FORMS, **TOKEN_FORMS, **INPUT_FORMS}[form]()


def form_shape(form: str) -> tuple:
    return SHAPES.get(form) or tuple(make_form(form).shape)


@dataclass
class Program:
    form: str
    shape: tuple
    flow: Any
    repr_: Any
    ir: Any
    hybrid: Any
    perceptrons: List[Any]
    thetas: List[float]
    wq: bool


def lif_net(form: str, seed: int, theta_enc: float, gain: float = 2.0, *, placement: str = "subsume"):
    """The converted form at random weights (a live cascade) with a cycle-accurate LIF activation at each perceptron's
    own theta: ``theta_enc`` on the first (the subsumed encoding, when the form has one), seeded draws on the others.
    ``placement``: the encoding layer's (``offload`` maps the first perceptron on chip, a core reading the raw input)."""
    torch.manual_seed(seed)
    net = make_form(form)
    shape = form_shape(form)
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.mul_(gain)
                if m.bias is not None:
                    m.bias.copy_(0.05 + 0.25 * torch.rand(m.bias.shape, generator=gen))
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.running_mean.uniform_(-0.2, 0.2, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
    net.eval()
    flow = convert_torch_model(net, shape, N_OUT, device="cpu", encoding_layer_placement=placement).eval()
    perceptrons = list(flow.get_perceptrons())
    rng = np.random.default_rng(2000 + seed)
    thetas = [float(theta_enc)] + [float(rng.uniform(0.4, 1.6)) for _ in perceptrons[1:]]
    for p, theta in zip(perceptrons, thetas):
        p.set_activation_scale(torch.tensor(theta))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return shape, flow, perceptrons, thetas


def scm_sequence(flow, perceptrons, *, wq: bool, retimed: bool, establish: Callable | None = None,
                 wq_bits: int = WQ_BITS, kill: "Dict[int, List[int]] | Callable | None" = None, masks: bool = False,
                 prune: bool = False, geometry: dict | None = None, relays: bool = False):
    """The Soft Core Mapping sequence on ``flow`` in place (``establish`` replaces its two gauge calls when given;
    ``wq_bits``: the Weight Quantization step's bit width). [U-IN A-2] ``kill`` (``{perceptron index: channels}``, or a
    function of the perceptrons giving one) zeroes output channels once the gauge is established (``masks``: declared
    pruned on their layers too, the pipeline's model-mask path); ``prune`` runs IR
    pruning before the build (``prune_ir_graph``: zero detection, seeded by the model's masks with ``masks``);
    ``geometry``: the mapping's core limits (256 x 256 by default); ``relays``: depth-balancing relay cores on every
    unequal-depth intra-segment fan-in (the declared ``lif_depth_balancing_relays``, before quantization as Soft Core
    Mapping inserts them; a no-op on a gap-free program)."""
    repr_ = flow.get_mapper_repr()
    if establish is None:
        compute_per_source_scales(repr_, arm_wire_value_ops=True)
        propagate_boundary_input_scales(flow, input_data_scale=1.0)
    else:
        establish(flow)
    kill = kill(perceptrons) if callable(kill) else kill
    if kill:
        kill_channels(perceptrons, kill, masks=masks)
    q_max = 127.0
    if wq:
        napq = NormalizationAwarePerceptronQuantization(wq_bits, "cpu", rate=1.0, two_scale=False)
        for p in perceptrons:
            napq.transform(p)
        q_max = float(2 ** (wq_bits - 1) - 1)
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=q_max, firing_mode="Default", **(geometry or dict(max_axons=256, max_neurons=256))).map(repr_)
    if relays:
        insert_depth_balancing_relays(ir, thresholding_mode="<=", q_max=int(q_max))
    if wq:
        quantize_ir_graph(ir, wq_bits, weight_quantization=True)
        arm_integer_membrane_lattice(flow)
    if prune:
        ir = _pruned(flow, ir, masks=masks)
    declare_host_kernel_evaluation(ir, "per_sample")  # the records' own evaluation (module docstring)
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed)
    return repr_, ir, hybrid


def kill_channels(perceptrons, kill: Dict[int, List[int]], *, masks: bool) -> None:
    """Zero the ``kill`` output channels of each perceptron (weights and bias); with ``masks``, also declare them pruned
    on the layer (``prune_row_mask``, read by ``get_initial_pruning_masks_from_model``)."""
    with torch.no_grad():
        for index, rows in kill.items():
            layer = perceptrons[index].layer
            layer.weight[rows] = 0.0
            if layer.bias is not None:
                layer.bias[rows] = 0.0
            if masks:
                out_f = int(layer.weight.shape[0])
                row = torch.zeros(out_f, dtype=torch.bool)
                row[rows] = True
                layer.prune_row_mask = row
                layer.prune_col_mask = torch.zeros(int(layer.weight.reshape(out_f, -1).shape[1]), dtype=torch.bool)


def first_core_kill(perceptrons) -> Dict[int, List[int]]:
    """d_tok's kill: output channels 1 and ``out - 2`` of the first on-chip perceptron with four or more."""
    for index, p in enumerate(perceptrons):
        out = int(p.layer.weight.shape[0])
        if not getattr(p, "is_encoding_layer", False) and out >= 4:
            return {index: [1, out - 2]}
    return {}


def _pruned(flow, ir, *, masks: bool):
    from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
    from mimarsinan.mapping.pruning.ir_pruning_masks import get_initial_pruning_masks_from_model

    if not masks:
        return prune_ir_graph(ir, simulation_steps=T, spiking_mode="lif")
    per_node, per_bank = get_initial_pruning_masks_from_model(flow, ir)
    return prune_ir_graph(ir, initial_pruned_per_node=per_node or None, initial_pruned_per_bank=per_bank or None,
                          simulation_steps=T, spiking_mode="lif")


def build(form: str, seed: int, theta_enc: float, semantics: str, wq: bool, *, gain: float = 2.0,
          establish: Callable | None = None, wq_bits: int = WQ_BITS, placement: str = "subsume",
          **pruning) -> Program:
    """``pruning``: ``scm_sequence``'s ``kill``, ``masks``, ``prune``, ``geometry`` and ``relays`` (U-IN)."""
    retimed = SEMANTICS[semantics][0]
    shape, flow, perceptrons, thetas = lif_net(form, seed, theta_enc, gain, placement=placement)
    repr_, ir, hybrid = scm_sequence(flow, perceptrons, wq=wq, retimed=retimed, establish=establish, wq_bits=wq_bits,
                                     **pruning)
    return Program(form, shape, flow, repr_, ir, hybrid, perceptrons, thetas, wq)


def samples(program: Program, n: int, seed: int) -> torch.Tensor:
    torch.manual_seed(5000 + seed)
    return 2.0 * torch.rand(n, *program.shape)


def _nf_capture(program: Program, x: torch.Tensor, semantics: str) -> dict:
    rec: dict = {}
    with torch.no_grad(), measurement_plane():
        out = SegmentForwardDriver(program.repr_, T, SEMANTICS[semantics][2]())(x, node_value_recorder=rec)
    counts = {}
    for i, p in enumerate(program.perceptrons):
        if id(p) in rec:
            v = rec[id(p)].reshape(x.shape[0], -1) / float(p.activation_scale)
            counts[i] = torch.round(v * T).double().numpy()
    return {"out": out, "counts": counts}


def nf_window_counts(program: Program, x: torch.Tensor, semantics: str):
    """The NF output and every perceptron's window counts, read inside the measurement plane (the gate's reading), as
    the program evaluates its host kernels (``twin_as_program``)."""
    nf = twin_as_program(program.hybrid, lambda prog, xs: _nf_capture(prog, xs, semantics), program, x)
    return nf["out"], nf["counts"]


class _ExecutionStages:
    """The mapping as the record indexes it: one entry per executed unit."""

    def __init__(self, hybrid):
        from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages

        self.stages = {i: stage for i, _prog, stage in enumerate_execution_stages(hybrid.stages)}


def deployed_flow(program: Program, semantics: str) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(program.shape, program.hybrid, simulation_length=T, thresholding_mode="<=",
                                 membrane_integer_lattice=bool(program.wq), **SEMANTICS[semantics][1]).eval()


def scm_window_counts(program: Program, x: torch.Tensor, semantics: str):
    flow = deployed_flow(program, semantics)
    by_exec = _ExecutionStages(program.hybrid)
    per = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            _, record = flow.forward_with_recording(x[i:i + 1], sample_index=i)
            per.append(_group_record_by_perceptron(record, by_exec,
                                                   values_of=lambda c: c.output_spike_count[: c.n_out_used]))
        out = flow(x)
    return out, {pi: np.stack([s[pi] for s in per]).astype(np.float64) for pi in per[0]}


def kept_neurons(program: Program, widths: Dict[int, int]) -> Dict[int, np.ndarray]:
    """[U-LIVE r2] Per core perceptron, the NF neurons (of its record's ``widths``) the deployed program holds, by the
    program's own kept-column rule (``derive_deployed_neuron_survival``: IR pruning's and the soft cores' compaction of
    every core of an executed perceptron, keyed by column and channel, projected through ``record_layouts``; the NF and
    SCM parity gate's own projection). A removed core's neurons and a neuron marked unread are then held by neither
    side; a core per position reads its own position's neurons."""
    from mimarsinan.mapping.pruning.deployed_neuron_survival import derive_deployed_neuron_survival, record_layouts

    survival, layouts = derive_deployed_neuron_survival(program.ir), record_layouts(program.perceptrons)
    index = {pi: np.arange(int(w), dtype=np.float64)[None, :] for pi, w in widths.items()}
    return {pi: v[0].astype(np.int64) for pi, v in survival.project(index, layouts).items()}


def count_mismatches(nf: Dict[int, np.ndarray], scm: Dict[int, np.ndarray],
                     kept: Dict[int, np.ndarray] | None = None) -> Dict[int, int]:
    """Per core perceptron, the window counts that differ (rows compared as sorted multisets, the gate's convention):
    [U-LIVE r2] the NF's counts at the neurons the deployed program holds (``kept_neurons``) against the deployed
    counts."""
    assert scm and set(scm) <= set(nf), (sorted(nf), sorted(scm))
    kept = kept or {}
    return {pi: int((np.sort(nf[pi][:, kept[pi]] if pi in kept else nf[pi], 1) != np.sort(scm[pi], 1)).sum())
            for pi in sorted(scm)}


def _nf_output_gauge(program: Program, like: torch.Tensor) -> torch.Tensor:
    """The gauge the NF walk decodes its output node at (U-TGAUGE: a forward returns its output in the value domain),
    read by the walk's own decode on ones."""
    driver = SegmentForwardDriver(program.repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
    driver.policy.prepare(driver)
    try:
        return driver.policy.decode_output(driver, driver._output, torch.ones_like(like)).double()
    finally:
        driver.policy.finalize(driver)


def output_counts(program: Program, nf_out: torch.Tensor, scm_out: torch.Tensor, n: int):
    """Both twins' outputs in the deployed flow's units: a core's output on its count grid (the NF holds count / T times
    theta), a host op's output in its wire units times T (the flow's readout convention; the NF holds its value, the
    wire value times the gauge the walk decodes it at)."""
    out_node = program.repr_.output_layer_mapper
    perceptron = getattr(out_node, "perceptron", None)
    if perceptron is not None:
        factor = T / float(perceptron.activation_scale)
    else:
        factor = float(T) / _nf_output_gauge(program, nf_out)
    return (nf_out.double() * factor).reshape(n, -1), scm_out.double().reshape(n, -1)


def compare(program: Program, semantics: str, n: int = 32, seed: int = 0):
    """``(mismatches per core perceptron, counts compared, max |output difference| on the count grid)``."""
    x = samples(program, n, seed)
    nf_out, nf = nf_window_counts(program, x, semantics)
    scm_out, scm = scm_window_counts(program, x, semantics)
    mism = count_mismatches(nf, scm, kept_neurons(program, {pi: int(v.shape[1]) for pi, v in nf.items()}))
    ref, dep = output_counts(program, nf_out, scm_out, n)
    return mism, int(sum(c.size for c in scm.values())), float((ref - dep).abs().max())
