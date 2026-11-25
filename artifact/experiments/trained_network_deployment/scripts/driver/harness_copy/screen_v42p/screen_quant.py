""

from __future__ import annotations

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.models.nn.activations.autograd import LIF_EXACT_QAT_THETA_FLOOR, _gated_lsq_backward
from mimarsinan.transformations.activation_scale_policy import CountQuantilePolicy

SIGNED_QUANTILE = 0.001
CALIBRATION_SAMPLE = 1 << 16
FOLD_SCALE_FLOOR = 1e-5


def quantize_weight_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    ""
    q_max = 2 ** (bits - 1) - 1
    scale = q_max / w.detach().abs().max().clamp_min(1e-12)
    return torch.round(w * scale) / scale


class WeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bits):
        return quantize_weight_tensor(w, int(bits))

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        return g, None


def staircase_forward(x, theta, shift, levels: int):
    ""
    dtype = torch.promote_types(x.dtype, torch.float32)
    safe = theta.to(dtype).clamp(min=LIF_EXACT_QAT_THETA_FLOOR)
    r = (x.to(dtype) + shift.to(dtype)) / safe
    q = torch.floor(levels * r).clamp(0.0, float(levels)) / levels
    return (safe * q - shift.to(dtype)).to(x.dtype)


def staircase_backward(g, x, theta, shift, levels: int):
    ""
    dtype = torch.promote_types(x.dtype, torch.float32)
    safe = theta.to(dtype).clamp(min=LIF_EXACT_QAT_THETA_FLOOR)
    r = (x.to(dtype) + shift.to(dtype)) / safe
    q = torch.floor(levels * r).clamp(0.0, float(levels)) / levels
    g = g.to(dtype)
    grad_z, grad_theta = _gated_lsq_backward(g, r, q, safe)
    return grad_z.to(x.dtype), grad_theta.to(theta.dtype), (grad_z.sum() - g.sum()).to(shift.dtype)


FUSE_ON_CUDA = True
_FUSED: dict = {}


def _kernels(x: torch.Tensor):
    if not (FUSE_ON_CUDA and x.is_cuda):
        return staircase_forward, staircase_backward
    if not _FUSED:
        _FUSED["fwd"] = torch.compile(staircase_forward, dynamic=True)
        _FUSED["bwd"] = torch.compile(staircase_backward, dynamic=True)
    return _FUSED["fwd"], _FUSED["bwd"]


class LeanCountStaircase(torch.autograd.Function):
    ""

    @staticmethod
    def forward(ctx, x, theta, shift, levels):
        ctx.save_for_backward(x, theta, shift)
        ctx.levels = levels
        return _kernels(x)[0](x, theta, shift, levels)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        x, theta, shift = ctx.saved_tensors
        grad_x, grad_theta, grad_shift = _kernels(x)[1](g, x, theta, shift, ctx.levels)
        return grad_x, grad_theta, (grad_shift if ctx.needs_input_grad[2] else None), None


def fused_kernel_check(device, n: int = 1 << 22) -> dict:
    ""
    out = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    for signed in (False, True):
        x = torch.randn(n, generator=g).to(device)
        x = (x if signed else torch.relu(x)).to(torch.bfloat16)
        theta = torch.tensor(2.3, device=device)
        shift = torch.tensor(1.7 if signed else 0.0, device=device)
        grad = torch.randn(n, generator=g).to(device).to(torch.bfloat16)
        fwd, bwd = _kernels(x)
        y_f, y_e = fwd(x, theta, shift, 15), staircase_forward(x, theta, shift, 15)
        b_f, b_e = bwd(grad, x, theta, shift, 15), staircase_backward(grad, x, theta, shift, 15)
        out["signed" if signed else "unsigned"] = {
            "forward_mismatches": int((y_f != y_e).sum()), "elements": n,
            "grad_x_mismatches": int((b_f[0] != b_e[0]).sum()),
            "grad_theta_rel_err": float((b_f[1] - b_e[1]).abs() / b_e[1].abs().clamp_min(1e-12)),
        }
    return out


class CrossbarInputQuantizer(nn.Module):
    ""

    def __init__(self, bits: int):
        super().__init__()
        self.bits = int(bits)
        self.levels = 2 ** self.bits - 1
        self.act_theta = nn.Parameter(torch.tensor(1.0))
        self.act_shift = nn.Parameter(torch.tensor(0.0))
        self.signed = False
        self.calibrated = False
        self.mode = "bypass"
        self._samples: list[torch.Tensor] = []

    def get_extra_state(self):
        return {"signed": self.signed, "calibrated": self.calibrated, "bits": self.bits}

    def set_extra_state(self, state):
        if int(state["bits"]) != self.bits:
            raise ValueError(f"checkpoint bits {state['bits']} != {self.bits}")
        self.signed, self.calibrated = bool(state["signed"]), bool(state["calibrated"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bypass":
            return x
        if self.mode == "observe":
            flat = x.detach().float().reshape(-1)
            if flat.numel() > CALIBRATION_SAMPLE:
                flat = flat[torch.randint(0, flat.numel(), (CALIBRATION_SAMPLE,), device=flat.device)]
            self._samples.append(flat)
            return x
        if not self.calibrated:
            raise RuntimeError("CrossbarInputQuantizer used before calibration")
        shift = self.act_shift if self.signed else torch.zeros((), device=x.device)
        return LeanCountStaircase.apply(x, self.act_theta, shift, self.levels)

    @torch.no_grad()
    def finalize_calibration(self) -> dict:
        if not self._samples:
            raise RuntimeError("no calibration samples observed")
        s = torch.cat(self._samples)
        self._samples = []
        low = float(safe_quantile(s, SIGNED_QUANTILE))
        self.signed = low < 0.0
        shift = -low if self.signed else 0.0
        theta = CountQuantilePolicy().scale(s + shift)
        self.act_shift.fill_(shift)
        self.act_theta.fill_(float(theta))
        self.calibrated = True
        return {"signed": self.signed, "shift": shift, "theta": float(theta)}


class _CrossbarMixin:
    ""

    w_bits: int
    in_q: CrossbarInputQuantizer
    _paired_bn: list

    def _effective_quantized_weight(self, weight: torch.Tensor):
        bn = self._paired_bn[0] if self._paired_bn else None
        if bn is None:
            return WeightSTE.apply(weight, self.w_bits), None
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        s = torch.where(s >= 0, s.clamp_min(FOLD_SCALE_FLOOR), s.clamp_max(-FOLD_SCALE_FLOOR))
        shape = (-1,) + (1,) * (weight.dim() - 1)
        return WeightSTE.apply(weight * s.reshape(shape), self.w_bits), s


class QConv2d(_CrossbarMixin, nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_q(x)
        if self.in_q.mode != "quantize":
            return super().forward(x)
        wq, s = self._effective_quantized_weight(self.weight)
        y = self._conv_forward(x, wq, None)
        if s is not None:
            y = y / s.reshape(1, -1, 1, 1).to(y.dtype)
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y


class QLinear(_CrossbarMixin, nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_q(x)
        if self.in_q.mode != "quantize":
            return super().forward(x)
        wq, s = self._effective_quantized_weight(self.weight)
        y = F.linear(x, wq, None)
        if s is not None:
            y = y / s.to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y


CROSSBAR_TYPES = (nn.Conv2d, nn.Linear)
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


def find_bn_pairs(model: nn.Module) -> dict[str, str]:
    ""
    gm = torch.fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    calls: dict[str, int] = {}
    pairs: dict[str, str] = {}
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        calls[node.target] = calls.get(node.target, 0) + 1
        if not isinstance(modules[node.target], CROSSBAR_TYPES):
            continue
        users = list(node.users)
        if (len(users) == 1 and users[0].op == "call_module"
                and isinstance(modules[users[0].target], BN_TYPES) and users[0].args[0] is node):
            pairs[node.target] = users[0].target
    reused = [n for n, c in calls.items() if c > 1 and isinstance(modules[n], CROSSBAR_TYPES + BN_TYPES)]
    if reused:
        raise ValueError(f"modules called more than once cannot carry per-site quantizers: {reused}")
    return pairs


def _make_crossbar(module: nn.Module, w_bits: int, a_bits: int) -> nn.Module:
    w = module.weight
    if isinstance(module, nn.Conv2d):
        q = QConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding,
                    module.dilation, module.groups, module.bias is not None, module.padding_mode,
                    device=w.device, dtype=w.dtype)
    else:
        q = QLinear(module.in_features, module.out_features, module.bias is not None, device=w.device, dtype=w.dtype)
    q.weight = module.weight
    q.bias = module.bias
    q.w_bits = int(w_bits)
    q.in_q = CrossbarInputQuantizer(a_bits).to(w.device)
    q._paired_bn = []
    return q


def quantize_model(model: nn.Module, w_bits: int, a_bits: int) -> dict:
    ""
    pairs = find_bn_pairs(model)
    named = dict(model.named_modules())
    crossbars = [n for n, m in named.items() if type(m) in CROSSBAR_TYPES]
    for name in crossbars:
        parent_name, _, attr = name.rpartition(".")
        parent = named[parent_name] if parent_name else model
        q = _make_crossbar(named[name], w_bits, a_bits)
        if name in pairs:
            q._paired_bn = [named[pairs[name]]]
        setattr(parent, attr, q)
    return {"crossbars": len(crossbars), "folded_bn": len(pairs), "unfolded": len(crossbars) - len(pairs),
            "w_bits": int(w_bits), "a_bits": int(a_bits)}


def quantizers(model: nn.Module) -> list[CrossbarInputQuantizer]:
    return [m for m in model.modules() if isinstance(m, CrossbarInputQuantizer)]


def set_quant_mode(model: nn.Module, mode: str) -> None:
    if mode not in ("quantize", "observe", "bypass"):
        raise ValueError(mode)
    for q in quantizers(model):
        q.mode = mode


@torch.no_grad()
def calibrate(model: nn.Module, batches) -> dict:
    ""
    was_training = model.training
    model.eval()
    set_quant_mode(model, "observe")
    for x in batches:
        model(x)
    stats = [q.finalize_calibration() for q in quantizers(model)]
    set_quant_mode(model, "quantize")
    model.train(was_training)
    return {"quantizers": len(stats), "signed": sum(s["signed"] for s in stats)}
