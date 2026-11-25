""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACC_LIMIT = 1 << 24
CALIB_KEEP = 1 << 15
CALIB_RATIOS = torch.linspace(0.02, 1.0, 50, dtype=torch.float64)
TOOLCHAIN_QUANTILE = 0.999
TOOLCHAIN_MIN_SCALE = 1e-3
QUANTILE_ELEMENT_LIMIT = 1 << 24
RULES = ("toolchain", "mse")


def levels(bits: int, signed: bool = True) -> int:
    ""
    return (1 << (int(bits) - 1)) - 1 if signed else (1 << int(bits)) - 1


def code_range(bits: int, signed: bool = True) -> tuple[int, int]:
    q = levels(bits, signed)
    return (-q, q) if signed else (0, q)


def subsample_to_limit(tensor: torch.Tensor, limit: int = QUANTILE_ELEMENT_LIMIT) -> torch.Tensor:
    ""
    flat = tensor.detach().reshape(-1)
    n = flat.numel()
    if n <= limit:
        return flat
    return flat[:: -(-n // limit)]


def safe_quantile(tensor: torch.Tensor, q: float, limit: int = QUANTILE_ELEMENT_LIMIT) -> torch.Tensor:
    ""
    flat = subsample_to_limit(tensor, limit)
    if not flat.is_floating_point():
        flat = flat.float()
    return torch.quantile(flat, q, interpolation="linear")


def quantize_weight_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    ""
    q_max = 2 ** (bits - 1) - 1
    scale = q_max / w.detach().abs().max().clamp_min(1e-12)
    return torch.round(w * scale) / scale


def weight_codes_and_step(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    ""
    q = levels(bits)
    m = w.detach().abs().max().clamp_min(1e-12)
    codes = torch.round(w.detach() * (q / m))
    return codes, m / q


class WeightSTE(torch.autograd.Function):
    ""

    @staticmethod
    def forward(ctx, w, bits):
        codes, step = weight_codes_and_step(w, int(bits))
        return codes * step

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        return g, None


def _precision(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def mse_step(samples: torch.Tensor, max_abs: float, bits: int, signed: bool = True) -> float:
    ""
    lo, q = code_range(bits, signed)
    if max_abs <= 0.0:
        return 1.0 / q
    x = samples.to(torch.float64)
    best, best_err = None, math.inf
    for r in CALIB_RATIOS.tolist():
        s = r * max_abs / q
        err = float(((torch.clamp(torch.round(x / s), lo, q) * s - x) ** 2).mean())
        if err < best_err:
            best, best_err = s, err
    return float(best)


class StaticGrid(nn.Module):
    ""

    def __init__(self, bits: int, groups: int = 1, group_dim: int = 1, name: str = "", signed: bool = True):
        super().__init__()
        self.bits = int(bits)
        self.signed = bool(signed)
        self.groups = int(groups)
        self.group_dim = int(group_dim)
        self.name = name
        self.register_buffer("step", torch.zeros(self.groups))
        self.mode = "off"
        self._gen = torch.Generator().manual_seed(0)
        self.reset_stats()


    @property
    def q(self) -> int:
        return levels(self.bits, self.signed)

    def set_bits(self, bits: int) -> None:
        self.bits = int(bits)

    def get_extra_state(self):
        return {"bits": self.bits, "signed": self.signed, "groups": self.groups, "group_dim": self.group_dim,
                "mode": self.mode}

    def set_extra_state(self, state):
        if int(state["groups"]) != self.groups:
            raise ValueError(f"{self.name}: checkpoint groups {state['groups']} != {self.groups}")
        self.bits, self.mode = int(state["bits"]), str(state["mode"])
        self.signed = bool(state.get("signed", True))

    def reset_stats(self) -> None:
        self._quantiles: list[list[float]] = [[] for _ in range(self.groups)]
        self._quantile_pooled: list[float] = []
        self._max = [0.0] * self.groups
        self._samples: list[list[torch.Tensor]] = [[] for _ in range(self.groups)]
        self._cache: dict = {}

    @property
    def observed(self) -> bool:
        return bool(self._quantile_pooled)


    def _parts(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.groups == 1:
            return [x]
        return [x.select(self.group_dim, g) for g in range(self.groups)]

    def observe(self, x: torch.Tensor) -> None:
        ""
        xd = x.detach().to(torch.float32)
        self._quantile_pooled.append(float(safe_quantile(xd.abs(), TOOLCHAIN_QUANTILE)))
        for g, part in enumerate(self._parts(xd)):
            flat = part.reshape(-1)
            if not flat.numel():
                continue
            a = flat.abs()
            self._quantiles[g].append(float(safe_quantile(a, TOOLCHAIN_QUANTILE)))
            self._max[g] = max(self._max[g], float(a.max()))
            if flat.numel() > CALIB_KEEP:
                idx = torch.randint(flat.numel(), (CALIB_KEEP,), generator=self._gen).to(flat.device)
                flat = flat[idx]
            self._samples[g].append(flat.cpu())
        self._cache = {}

    def rule_steps(self, rule: str, bits: int | None = None, signed: bool | None = None,
                   pooled: bool = False) -> list[float]:
        ""
        bits = self.bits if bits is None else int(bits)
        signed = self.signed if signed is None else bool(signed)
        if rule not in RULES:
            raise ValueError(f"unknown step rule {rule!r}; declared {RULES}")
        if not self.observed:
            raise RuntimeError(f"REFUSED: grid {self.name} observed nothing")
        key = (rule, bits, signed, bool(pooled))
        if key in self._cache:
            return list(self._cache[key])
        q = levels(bits, signed)
        if rule == "toolchain":
            if pooled:
                steps = [max(max(self._quantile_pooled), TOOLCHAIN_MIN_SCALE) / q] * self.groups
            else:
                steps = [max(max(qs) if qs else 0.0, TOOLCHAIN_MIN_SCALE) / q for qs in self._quantiles]
        else:
            if pooled:
                s = mse_step(torch.cat([torch.cat(v) for v in self._samples if v]), max(self._max), bits, signed)
                steps = [s] * self.groups
            else:
                steps = [mse_step(torch.cat(v), self._max[g], bits, signed) if v else 1.0 / q
                         for g, v in enumerate(self._samples)]
        self._cache[key] = list(steps)
        return steps

    def apply_rule(self, rule: str, bits: int | None = None, signed: bool | None = None, pooled: bool = False):
        ""
        steps = self.rule_steps(rule, bits, signed, pooled)
        if bits is not None:
            self.set_bits(bits)
        if signed is not None:
            self.signed = bool(signed)
        with torch.no_grad():
            self.step.copy_(torch.tensor(steps, dtype=self.step.dtype))
        return steps


    def _view(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return s.reshape(())
        shape = [1] * x.dim()
        shape[self.group_dim] = self.groups
        return s.reshape(shape)

    def step_view(self, x: torch.Tensor) -> torch.Tensor:
        return self._view(self.step.detach().to(_precision(x)), x)

    def _check(self) -> None:
        if not bool((self.step.detach() > 0).all()):
            raise RuntimeError(f"REFUSED: grid {self.name} has a non-positive step (not calibrated)")

    def codes(self, x: torch.Tensor) -> torch.Tensor:
        ""
        self._check()
        dt = _precision(x)
        lo, q = code_range(self.bits, self.signed)
        return torch.clamp(torch.round(x.to(dt) / self.step_view(x)), lo, q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.mode == "observe":
            self.observe(x)
            return x
        xf = x.to(_precision(x))
        values = self.codes(xf) * self.step_view(xf)
        if self.training:
            return xf + (values - xf).detach()
        return values


def exact_product(codes_a: torch.Tensor, codes_b: torch.Tensor, step_a: torch.Tensor, step_b: torch.Tensor) -> torch.Tensor:
    ""
    k = codes_a.shape[-1]
    bound = k * float(codes_a.detach().abs().max()) * float(codes_b.detach().abs().max()) if k else 0.0
    if bound >= ACC_LIMIT and codes_a.dtype != torch.float64:
        raise RuntimeError(f"REFUSED: accumulator bound {bound:.0f} >= 2^24 for k = {k}")
    with torch.autocast(codes_a.device.type, enabled=False):
        m = torch.matmul(codes_a, codes_b)
    return m * (step_a * step_b)


def accumulator_bound_ok(k: int, bits_a: int, bits_b: int, signed_a: bool = True) -> bool:
    ""
    return k * levels(bits_a, signed_a) * levels(bits_b) < ACC_LIMIT


class QuantLinear(nn.Module):
    ""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, act_bits: int, weight_bits: int,
                 name: str = ""):
        super().__init__()
        ref = nn.Linear(in_features, out_features, bias=bias)
        self.weight = ref.weight
        self.bias = ref.bias
        self.in_features, self.out_features = in_features, out_features
        self.weight_bits = int(weight_bits)
        self.in_grid = StaticGrid(act_bits, 1, name=f"{name}.in")
        self.quantized = False
        self.name = name
        if not accumulator_bound_ok(in_features, act_bits, weight_bits):
            raise ValueError(f"REFUSED: {name}: k {in_features} x q_a x q_w >= 2^24")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_grid.mode == "observe":
            self.in_grid.observe(x)
        if not self.quantized:
            return F.linear(x, self.weight, self.bias)
        if self.training:
            dt = _precision(x)
            with torch.autocast(x.device.type, enabled=False):
                w = WeightSTE.apply(self.weight.to(dt), self.weight_bits)
                return F.linear(self.in_grid(x.to(dt)), w, None if self.bias is None else self.bias.to(dt))
        ca = self.in_grid.codes(x)
        cw, sw = weight_codes_and_step(self.weight, self.weight_bits)
        y = exact_product(ca, cw.to(ca.dtype).t(), self.in_grid.step_view(ca).to(ca.dtype), sw.to(ca.dtype))
        return y if self.bias is None else y + self.bias.to(y.dtype)


class QuantPatchEmbed(nn.Module):
    ""

    def __init__(self, in_ch: int, dim: int, patch: int, *, act_bits: int, weight_bits: int):
        super().__init__()
        ref = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, padding=0, bias=True)
        nn.init.kaiming_normal_(ref.weight)
        self.weight, self.bias = ref.weight, ref.bias
        self.patch = patch
        self.weight_bits = int(weight_bits)
        self.in_grid = StaticGrid(act_bits, 1, name="tokenizer.in")
        self.quantized = False
        if not accumulator_bound_ok(in_ch * patch * patch, act_bits, weight_bits):
            raise ValueError("REFUSED: tokenizer accumulator bound")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_grid.mode == "observe":
            self.in_grid.observe(x)
        if not self.quantized:
            return F.conv2d(x, self.weight, self.bias, stride=self.patch).flatten(2).transpose(1, 2)
        patches = F.unfold(x, kernel_size=self.patch, stride=self.patch).transpose(1, 2)
        w2 = self.weight.reshape(self.weight.shape[0], -1)
        if self.training:
            dt = _precision(x)
            with torch.autocast(x.device.type, enabled=False):
                return F.linear(self.in_grid(patches.to(dt)), WeightSTE.apply(w2.to(dt), self.weight_bits),
                                self.bias.to(dt))
        ca = self.in_grid.codes(patches)
        cw, sw = weight_codes_and_step(w2, self.weight_bits)
        y = exact_product(ca, cw.to(ca.dtype).t(), self.in_grid.step_view(ca).to(ca.dtype), sw.to(ca.dtype))
        return y + self.bias.to(y.dtype)


class OperandContraction(nn.Module):
    ""

    def __init__(self, heads: int, *, act_bits: int, payload_bits: int, name: str):
        super().__init__()
        self.a_grid = StaticGrid(act_bits, heads, group_dim=1, name=f"{name}.streamed")
        self.b_grid = StaticGrid(payload_bits, heads, group_dim=1, name=f"{name}.stationary")
        self.on_chip = False
        self.a_on = self.b_on = True
        self.name = name

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.a_grid.mode == "observe":
            self.a_grid.observe(a)
            self.b_grid.observe(b)
        if not self.on_chip:
            return a @ b
        if self.training or not (self.a_on and self.b_on):
            dt = _precision(a)
            with torch.autocast(a.device.type, enabled=False):
                av = self.a_grid(a.to(dt)) if self.a_on else a.to(dt)
                bv = self.b_grid(b.to(dt)) if self.b_on else b.to(dt)
                return av @ bv
        ca, cb = self.a_grid.codes(a), self.b_grid.codes(b)
        dt = ca.dtype
        return exact_product(ca, cb.to(dt), self.a_grid.step_view(ca).to(dt), self.b_grid.step_view(cb).to(dt))
