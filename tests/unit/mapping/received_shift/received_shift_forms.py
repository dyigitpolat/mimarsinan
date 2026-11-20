"""[U-RSHIFT] Converter-built forms of the received shift, and the read of a reader against the trained reader.

Each form is plain torch code converted by the toolchain's converter (the encoding layer subsumed), its perceptrons at
their calibrated maxima as cycle-accurate LIF nodes, the gauges established. A sequence is Soft Core Mapping's policy
alone (``policy``), the gauge seam then the policy (``seam``, a LIF row's), or the opt-in signed-seam install
(``install``).

The read is the reader's own linear identity at the effective level. On its input plus the field ``d`` each axon
receives, placed here by the form's own construction (which elements a source emits, a pool's window shares from the
pool's own module), never by the unit's propagation, a reader departs from the trained reader (its effective bias before
the sequence) by ``e_p = W~_p . d_p + B~_p - B~_before`` at every core position ``p``, a padding slot reading zero.
``e`` is zero exactly where the reader's correction is its own response to what it receives. Reads are in gauges (rate
units); ``raw`` is ``e`` times the reader's threshold.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from spiking_points import point_of_mode
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_point
from mimarsinan.mapping.support.negative_boundary import apply_negative_boundary_policy
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.transformations.perceptron import position_fold
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.orchestration.negative_boundary_seam import NegativeBoundarySeam

T = 15
#: The read's tolerance: float dust of max(1, |value|) (the generator's own, vgen's 1e-5).
RELATIVE = 1e-5


def init_(model: nn.Module, gen: torch.Generator, wscale: float = 1.0) -> None:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * wscale / (m.weight[0].numel() ** 0.5))
                if m.bias is not None:
                    m.bias.copy_(torch.randn(m.bias.shape, generator=gen) * 0.1)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.copy_(torch.randn(m.num_features, generator=gen) * 1.2)
                m.bias.copy_(torch.randn(m.num_features, generator=gen) * 0.5)
                m.running_mean.uniform_(0.1, 0.5, generator=gen)
                m.running_var.uniform_(0.4, 0.9, generator=gen)


class PadRead(nn.Module):
    """The padding record's forms: an encoder, a signed host normalization (after a strided core projection, or before
    a pool), read by a zero-padded convolution on a core."""

    def __init__(self, gen, pre: str, k: int, s: int):
        super().__init__()
        self.shape = (2, 14, 14)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(3, 3, 1, stride=2), nn.ReLU()) if pre == "proj" else nn.Identity()
        self.bn = nn.BatchNorm2d(3)
        self.pool = {"avg": nn.AvgPool2d(2), "max": nn.MaxPool2d(2)}.get(pre, nn.Identity())
        self.reader = nn.Sequential(nn.Conv2d(3, 4, k, stride=s, padding=k // 2), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        return self.reader(self.pool(self.bn(self.proj(self.enc(x))))).flatten(1)


class Join(nn.Module):
    """The check's form: a signed host normalization and an unsigned host max pool of one encoder, joined along the
    channels, read by a core convolution (``k`` 1 unpadded, 3 zero-padded)."""

    def __init__(self, gen, k: int):
        super().__init__()
        self.shape = (3, 7, 7)
        self.enc = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(4)
        self.mp = nn.MaxPool2d(3, stride=1, padding=1)
        self.reader = nn.Sequential(nn.Conv2d(8, 3, k, padding=k // 2), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        e = self.enc(x)
        return self.reader(torch.cat([self.bn(e), self.mp(e)], dim=1)).flatten(1)


class TwoShifted(nn.Module):
    """Two signed host values (two normalizations of one encoder, the second far more negative) joined along the
    channels, read by one core convolution: two sources, two shifts."""

    def __init__(self, gen, k: int):
        super().__init__()
        self.shape = (3, 7, 7)
        self.enc = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        self.bna = nn.BatchNorm2d(3)
        self.bnb = nn.BatchNorm2d(3)
        self.reader = nn.Sequential(nn.Conv2d(6, 3, k, padding=k // 2), nn.ReLU())
        init_(self, gen)
        with torch.no_grad():
            self.bnb.bias.sub_(1.5)

    def forward(self, x):
        e = self.enc(x)
        return self.reader(torch.cat([self.bna(e), self.bnb(e)], dim=1)).flatten(1)


class PoolLift(nn.Module):
    """A host normalization that goes slightly negative (its own boundary, read by a core) and a zero-padded average
    pool of it that counts its padding and never goes negative (no boundary of its own), the pool read by a core: the
    lift crosses the pool, whose border windows pass 4/9 and 6/9 of it."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 9, 9)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.reader = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.direct = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        init_(self, gen)
        with torch.no_grad():
            enc = self.enc[0]
            enc.weight.mul_(0.2)
            enc.bias.fill_(3.0)
            self.bn.running_mean.fill_(3.0)
            self.bn.running_var.fill_(0.04)
            self.bn.weight.fill_(0.1)
            self.bn.bias.fill_(0.35)

    def forward(self, x):
        v = self.bn(self.enc(x))
        return torch.cat([self.reader(self.pool(v)).flatten(1), self.direct(v).flatten(1)], dim=1)


class TwoPads(nn.Module):
    """One signed host normalization read by a 3x3 and a 5x5 zero-padded convolution: no one padding carries it."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 9, 9)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.r3 = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.r5 = nn.Sequential(nn.Conv2d(3, 4, 5, padding=2), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        v = self.bn(self.enc(x))
        return torch.cat([self.r3(v), self.r5(v)], dim=1).flatten(1)


class OneRead(nn.Module):
    """An encoder, a signed host normalization, a zero-padded convolution on a core."""

    def __init__(self, gen, k: int = 3):
        super().__init__()
        self.shape = (2, 9, 9)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.reader = nn.Sequential(nn.Conv2d(3, 4, k, padding=k // 2), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        return self.reader(self.bn(self.enc(x))).flatten(1)


class HostJoin(nn.Module):
    """The check's join flattened into a host Linear with a bias (``linear``) or a host LayerNorm (``layernorm``); the
    normalization is read by a core too, so it is a boundary the lift carries."""

    def __init__(self, gen, kind: str):
        super().__init__()
        self.shape = (2, 5, 5)
        c = 3
        self.enc = nn.Sequential(nn.Conv2d(2, c, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(c)
        self.mp = nn.MaxPool2d(3, stride=1, padding=1)
        self.direct = nn.Sequential(nn.Conv2d(c, 4, 1), nn.ReLU())
        n = 2 * c * 25
        self.host = nn.Linear(n, 5) if kind == "linear" else nn.LayerNorm(n)
        init_(self, gen)

    def forward(self, x):
        e = self.enc(x)
        v, q = self.bn(e), self.mp(e)
        j = torch.cat([v, q], dim=1).flatten(1)
        return torch.cat([self.host(j), self.direct(v).flatten(1)], dim=1)


class CoreChain(nn.Module):
    """Two zero-padded convolutions on cores, the second reading the first: an affine fold's producer and consumer."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 10, 10)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.a = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        return self.b(self.a(self.enc(x))).flatten(1)


class HostEncoder(nn.Module):
    """A signed host normalization of the raw input read by a zero-padded convolution (made an encoder on the host by
    hand) and by a 1x1 core."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 8, 8)
        self.bn = nn.BatchNorm2d(2)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.ReLU())
        self.core = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(6, 4, 1), nn.ReLU())
        init_(self, gen)

    def forward(self, x):
        v = self.bn(x)
        return self.head(torch.cat([self.enc(v), self.core(v)], dim=1)).flatten(1)


# ---- building and sequences ----------------------------------------------------------------------------------------
def convert(model: nn.Module):
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model.eval(), model.shape, 1, device="cpu", encoding_layer_placement="subsume").eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    repr_._ensure_exec_graph()
    return flow, repr_


def gauges(flow, repr_, x_cal: torch.Tensor, floor: float = 0.05) -> list:
    """Each perceptron at its calibrated float maximum as a cycle-accurate LIF node; the gauges established."""
    perceptrons = list(flow.get_perceptrons())
    peak: dict = {}
    hooks = [p.activation.register_forward_hook(
        lambda _m, _i, out, i=i: peak.__setitem__(i, max(peak.get(i, 0.0), float(out.detach().max()))))
        for i, p in enumerate(perceptrons)]
    with torch.no_grad():
        flow(x_cal)
    for h in hooks:
        h.remove()
    for i, p in enumerate(perceptrons):
        p.set_activation_scale(torch.tensor(max(peak.get(i, floor), floor)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    return perceptrons


def build(model: nn.Module, seed: int, n: int = 48):
    """``(flow, repr_, x_cal)``: the converted form at its gauges, with a seeded calibration batch."""
    flow, repr_ = convert(model)
    x_cal = torch.randn(n, *model.shape, generator=torch.Generator().manual_seed(700 + seed))
    gauges(flow, repr_, x_cal)
    return flow, repr_, x_cal


def policy(flow, x_cal) -> str:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        apply_negative_boundary_policy(flow, x_cal, T, shift_enabled=True,
                                       forward_fn=calibration_forward_for_point(point_of_mode("lif", soma_law=DEFAULT_SOMA_LAW)))
    return buf.getvalue()


def seam(flow, x_cal) -> str:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        NegativeBoundarySeam(calibration_x=x_cal, simulation_steps=T, input_data_scale=1.0, point=point_of_mode("lif", soma_law=DEFAULT_SOMA_LAW),
                             shift_enabled=True)(flow)
    return buf.getvalue()


class _Trainer:
    def __init__(self, x_cal):
        self.x_cal = x_cal

    def iter_validation_batches(self, n):
        half = self.x_cal.shape[0] // n
        for i in range(n):
            yield self.x_cal[i * half:(i + 1) * half], torch.zeros(half, dtype=torch.long)


def install(flow, x_cal) -> str:
    from mimarsinan.tuning.orchestration.signed_seam_install import install_signed_seam_offsets

    with contextlib.redirect_stdout(io.StringIO()) as buf:
        install_signed_seam_offsets(flow, _Trainer(x_cal), {"device": "cpu", "simulation_steps": T,
                                                             "input_data_scale": 1.0})
    return buf.getvalue()


def sequence(flow, x_cal, name: str) -> str:
    """``policy``; ``seam`` (the seam, then the policy, a LIF row's order); ``install`` (the opt-in install, then the
    policy)."""
    printed = seam(flow, x_cal) if name == "seam" else install(flow, x_cal) if name == "install" else ""
    return printed + policy(flow, x_cal)


# ---- the read ------------------------------------------------------------------------------------------------------
def conv_readers(repr_) -> list:
    return [n for n in repr_.execution_order() if isinstance(n, Conv2DPerceptronMapper)
            and not getattr(n.perceptron, "is_encoding_layer", False)]


def ops(repr_) -> list:
    return [n for n in repr_.execution_order() if isinstance(n, ComputeOpMapper)]


def op_named(repr_, attr: str):
    """The host op whose torch module is the model's ``attr`` (the converter names ops by their module's path)."""
    return next(o for o in ops(repr_) if getattr(o, "name", None) == attr)


def own_shift(op) -> float | None:
    """The shift ``op``'s readers receive, in their effective (wire) units: Soft Core Mapping's stamp (one value on a
    channel-first value), or a lift over its emitted gauge."""
    stamp = getattr(op, "_negative_shift", None)
    if stamp is not None:
        flat = np.asarray(stamp, dtype=np.float32).reshape(-1)
        return float(flat[0]) if flat.size == 1 else None
    if getattr(op, "output_value_offset", None) is not None:
        kappa = float(torch.as_tensor(op.output_scale).float().mean())
        return float(torch.tensor(float(op.output_value_offset) / max(kappa, 1e-12)))
    return None


def input_shape(flow, node, model) -> tuple:
    """The per-sample shape of ``node``'s input in the plain forward."""
    seen = {}
    original = node._forward_impl

    def grab(x, _o=original):
        seen["shape"] = tuple(x.shape[1:])
        return _o(x)
    node._forward_impl = grab
    try:
        with torch.no_grad():
            flow(torch.rand(1, *model.shape))
    finally:
        del node._forward_impl
    return seen["shape"]


def biases(readers) -> dict:
    """Each reader's effective bias before a sequence (its trained bias in the effective domain)."""
    return {id(n.perceptron): PerceptronTransformer().get_effective_bias(n.perceptron).detach().double().clone()
            for n in readers}


def slot_rows(node, d: torch.Tensor) -> torch.Tensor:
    """The field each core position's slots read, ``(positions, in_features)``: a convolution's patches by ``F.unfold``
    at its own kernel, stride, padding and dilation (a padding slot reads zero), a linear's leading positions."""
    if isinstance(node, Conv2DPerceptronMapper):
        return F.unfold(d.double().unsqueeze(0), node.kernel_size, dilation=node.dilation, padding=node.padding,
                        stride=node.stride)[0].t()
    return d.double().reshape(-1, int(node.perceptron.input_features))


def effective_rows(p):
    """``(weights (positions or 1, out, in), bias (positions or 1, out))`` in the effective domain."""
    if position_fold.holds_positions(p):
        rows, index = position_fold.effective_weight_rows(p)
        w = rows.detach().double()[index]
        b = PerceptronTransformer(positions=True).get_effective_bias(p).detach().double()
        return w, b
    tr = PerceptronTransformer()
    return tr.get_effective_weight(p).detach().double().unsqueeze(0), tr.get_effective_bias(p).detach().double()[None]


def reader_error(node, d: torch.Tensor, before: torch.Tensor) -> dict:
    """``e_p = W~_p . d_p + B~_p - B~_before`` per core position and output, in gauges, with its border split (the
    positions whose patch reaches a padding slot) and the raw maximum."""
    rows = slot_rows(node, d)
    w, b = effective_rows(node.perceptron)
    e = (w * rows.unsqueeze(1)).sum(-1) + b - before.reshape(1, -1)
    theta = float(torch.as_tensor(node.perceptron.activation_scale).max())
    padded = F.unfold(torch.ones(1, *d.shape, dtype=torch.float64), node.kernel_size, dilation=node.dilation,
                      padding=node.padding, stride=node.stride)[0].t() if isinstance(node, Conv2DPerceptronMapper) \
        else torch.ones_like(rows)
    border = (padded == 0).any(dim=1)
    scale = max(1.0, float((w * rows.unsqueeze(1)).sum(-1).abs().max()), float(b.abs().max()))
    return {"max": float(e.abs().max()), "raw": float(e.abs().max()) * theta,
            "border": float(e[border].abs().max()) if bool(border.any()) else 0.0,
            "inner": float(e[~border].abs().max()) if bool((~border).any()) else 0.0,
            "inner_min": float(e[~border].abs().amax(dim=1).min()) if bool((~border).any()) else 0.0,
            "tol": RELATIVE * scale, "positions": int(rows.shape[0])}


def field_full(shape: tuple, value: float) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=torch.float64)


def field_parts(shape: tuple, parts: list) -> torch.Tensor:
    """A channel-first join's field: ``parts`` is ``[(channels, value)]`` in the join's order."""
    d = torch.zeros(shape, dtype=torch.float64)
    start = 0
    for channels, value in parts:
        d[start:start + channels] = float(value)
        start += channels
    return d


def tuner(bits: int = 4):
    """The Weight Quantization step's projection on the law route (a float bias, so the compensation is isolated)."""
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner as Tuner,
    )

    t = Tuner.__new__(Tuner)
    t.quantization_bits = bits
    t.two_scale_projection = False
    t.fractional_scale, t.float_bias = False, True
    t.bias_rows_floor = None
    t.register_bias = False
    t.pipeline = SimpleNamespace(config={"device": "cpu"})
    return t


def project(perceptrons, bits: int = 4) -> None:
    transform = tuner(bits)._get_new_perceptron_transform(1.0)
    with torch.no_grad():
        for p in perceptrons:
            transform(p)
