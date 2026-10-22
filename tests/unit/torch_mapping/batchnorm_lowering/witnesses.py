"""Self-contained witness models and fp64 parity helpers for the BatchNorm lowering tests.

Every witness is built here from the layer types the converter accepts (declared local), as the
audit scripts build them (arith_models.py, net_models.py, dbg_bn_eps.py). Parity follows the E270
convention: fp64, eval mode, BatchNorm statistics randomized on a deep copy, four seeded inputs (two
N(0, 1), two U(0, 1)), relative error max |delta| / max(1, max |reference|).
"""

from __future__ import annotations

import contextlib
import copy
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program

NC = 10
PARITY_BOUND = 1e-12
CONTRACTS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


# ----------------------------------------------------------------------------- parity helpers
def randomize_batchnorm(model: nn.Module, seed: int = 1) -> nn.Module:
    """Non-trivial statistics and affine terms on every BatchNorm, so a dropped or misplaced BN shows."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None:
                n, dt = m.running_mean.numel(), m.running_mean.dtype
                m.running_mean.copy_((torch.rand(n, generator=g, dtype=torch.float64) * 0.4 - 0.2).to(dt))
                m.running_var.copy_((torch.rand(n, generator=g, dtype=torch.float64) * 1.5 + 0.5).to(dt))
                if m.weight is not None:
                    m.weight.copy_((torch.rand(n, generator=g, dtype=torch.float64) * 1.5 + 0.5).to(dt))
                    m.bias.copy_((torch.rand(n, generator=g, dtype=torch.float64) * 0.4 - 0.2).to(dt))
    return model


def fp64_eval_copy(model: nn.Module, seed: int = 1) -> nn.Module:
    """A deep fp64 eval copy with randomized BatchNorm statistics (the caller's model is untouched)."""
    return randomize_batchnorm(copy.deepcopy(model).double().eval(), seed)


def seeded_inputs(shape: tuple[int, ...], seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.cat([
        torch.randn(2, *shape, generator=g, dtype=torch.float64),
        torch.rand(2, *shape, generator=g, dtype=torch.float64),
    ])


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    got = got.reshape(want.shape)
    return float((got - want).abs().max()) / max(1.0, float(want.abs().max()))


def convert_quietly(model: nn.Module, shape: tuple[int, ...], contract: str):
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_torch_model(model, shape, NC, packaging=CONTRACTS[contract]).eval()


def converted_error(model: nn.Module, shape: tuple[int, ...], contract: str) -> float:
    """Converted model (the NF side) against the native model; ``model`` is an fp64 eval model."""
    flow = convert_quietly(model, shape, contract)
    x = seeded_inputs(shape)
    with torch.no_grad():
        return relative_error(flow(x), model(x))


def value_program_error(model: nn.Module, shape: tuple[int, ...]) -> float:
    """MVM identity value program on the IR (the SCM side) against the native model."""
    program = identity_value_program(convert_quietly(model, shape, "mvm"))
    x = seeded_inputs(shape)
    with torch.no_grad():
        return relative_error(program(x), model(x))


def batchnorm_node_names(model: nn.Module, shape: tuple[int, ...]) -> set[str]:
    """FX node names of every BatchNorm1d/2d call in the normalized traced graph."""
    gm = normalize_fx_graph(trace_model(model, shape, device="cpu"))
    mods = dict(gm.named_modules())
    return {n.name for n in gm.graph.nodes
            if n.op == "call_module" and isinstance(mods.get(str(n.target)), (nn.BatchNorm1d, nn.BatchNorm2d))}


# ----------------------------------------------------------------------------- D1: BatchNorm not after a weight layer
class ConvReluBnConv(nn.Module):
    """conv -> ReLU -> BN -> conv -> ReLU -> FC (the BN reads an activation)."""

    def __init__(self):
        super().__init__()
        self.c1, self.a1, self.bn = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8)
        self.c2, self.a2 = nn.Conv2d(8, 8, 3, padding=1, stride=2), nn.ReLU()
        self.fc = nn.Linear(8 * 4 * 4, NC)

    def forward(self, x):
        return self.fc(torch.flatten(self.a2(self.c2(self.bn(self.a1(self.c1(x))))), 1))


class BnOnInput(nn.Module):
    """BN on the graph input, then conv-ReLU-FC."""

    def __init__(self):
        super().__init__()
        self.bn, self.c, self.a = nn.BatchNorm2d(3), nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        return self.fc(torch.flatten(self.a(self.c(self.bn(x))), 1))


class BnOnSum(nn.Module):
    """PyramidNet-style: BN on a residual sum, then conv (3x3, padded: no forward fold)."""

    def __init__(self):
        super().__init__()
        self.c0, self.c1 = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(3, 8, 1)
        self.bn, self.c2, self.a = nn.BatchNorm2d(8), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        return self.fc(torch.flatten(self.a(self.c2(self.bn(self.c0(x) + self.c1(x)))), 1))


class BnOnConcat(nn.Module):
    """DenseNet-style: BN-ReLU-conv on a channel concatenation."""

    def __init__(self):
        super().__init__()
        self.c0, self.a0 = nn.Conv2d(3, 6, 3, padding=1), nn.ReLU()
        self.bn, self.a1, self.c1 = nn.BatchNorm2d(9), nn.ReLU(), nn.Conv2d(9, 4, 1)
        self.fc = nn.Linear(4 * 8 * 8, NC)

    def forward(self, x):
        h = torch.cat([x, self.a0(self.c0(x))], 1)
        return self.fc(torch.flatten(self.c1(self.a1(self.bn(h))), 1))


class DoubleBn(nn.Module):
    """conv -> BN -> BN -> ReLU: the second BN has no fold slot left."""

    def __init__(self):
        super().__init__()
        self.c, self.bn1, self.bn2, self.a = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.BatchNorm2d(8), nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        return self.fc(torch.flatten(self.a(self.bn2(self.bn1(self.c(x)))), 1))


class ActThenAct(nn.Module):
    """conv -> ReLU -> GELU: under spiking the second activation has no slot left."""

    def __init__(self):
        super().__init__()
        self.c, self.a1, self.a2 = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.GELU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        return self.fc(torch.flatten(self.a2(self.a1(self.c(x))), 1))


class LinearTokensBn(nn.Module):
    """Linear on (B, L, C) then BatchNorm1d(L): the BN normalizes the token axis, not the Linear's features."""

    def __init__(self, tokens: int = 8, features: int = 8):
        super().__init__()
        self.tokens = tokens
        self.l1, self.bn, self.a = nn.Linear(features, features), nn.BatchNorm1d(tokens), nn.ReLU()
        self.fc = nn.Linear(tokens * features, NC)

    def forward(self, x):
        h = x.reshape(x.shape[0], self.tokens, -1)
        return self.fc(torch.flatten(self.a(self.bn(self.l1(h))), 1))


class LinearTokensChannelsLastBn(nn.Module):
    """Linear on (B, L, C) then a channels-last BN over C: folds into the Linear (the Mixer form)."""

    def __init__(self, tokens: int = 8, features: int = 8):
        super().__init__()
        self.tokens = tokens
        self.l1, self.bn, self.a = nn.Linear(features, features), ChannelsLastBatchNorm1d(features), nn.ReLU()
        self.fc = nn.Linear(tokens * features, NC)

    def forward(self, x):
        h = x.reshape(x.shape[0], self.tokens, -1)
        return self.fc(torch.flatten(self.a(self.bn(self.l1(h))), 1))


# ----------------------------------------------------------------------------- D8: fold into a weight layer with a second consumer
class FanoutBn(nn.Module):
    """The conv output feeds BN-ReLU-conv and, raw, the skip of an add."""

    def __init__(self):
        super().__init__()
        self.c0, self.bn, self.r = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        self.c1, self.r1 = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        h = self.c0(x)
        return self.fc(torch.flatten(self.r1(self.c1(self.r(self.bn(h))) + h), 1))


class FanoutAct(nn.Module):
    """The conv output feeds ReLU-conv and, raw, the skip of an add (the spiking activation fold)."""

    def __init__(self):
        super().__init__()
        self.c0, self.r = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        self.c1, self.r1 = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, NC)

    def forward(self, x):
        h = self.c0(x)
        return self.fc(torch.flatten(self.r1(self.c1(self.r(h)) + h), 1))


class StemTwoConsumers(nn.Module):
    """DenseNet stem: the stem conv feeds the first layer's BN-ReLU-conv and, raw, the concatenation."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 3, padding=1, bias=False)
        self.bn1, self.r1, self.c1 = nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 4, 3, padding=1, bias=False)
        self.bn2, self.r2 = nn.BatchNorm2d(10), nn.ReLU()
        self.fc = nn.Linear(10 * 8 * 8, NC)

    def forward(self, x):
        s = self.stem(x)
        h = torch.cat([s, self.c1(self.r1(self.bn1(s)))], 1)
        return self.fc(torch.flatten(self.r2(self.bn2(h)), 1))


class LinearFanoutBn(nn.Module):
    """Linear output feeds BN-ReLU-Linear and, raw, an add."""

    def __init__(self):
        super().__init__()
        self.l0, self.bn, self.r, self.l1 = nn.Linear(48, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 16)
        self.fc = nn.Linear(16, NC)

    def forward(self, x):
        h = self.l0(torch.flatten(x, 1)[:, :48])
        return self.fc(self.l1(self.r(self.bn(h))) + h)


# ----------------------------------------------------------------------------- families (arith_models.py forms, small)
class PreActBottleneck(nn.Module):
    def __init__(self, cin, planes, stride):
        super().__init__()
        out = 4 * planes
        self.bn1, self.relu1 = nn.BatchNorm2d(cin), nn.ReLU()
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn2, self.relu2 = nn.BatchNorm2d(planes), nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn3, self.relu3 = nn.BatchNorm2d(planes), nn.ReLU()
        self.conv3 = nn.Conv2d(planes, out, 1, bias=False)
        self.shortcut = nn.Conv2d(cin, out, 1, stride=stride, bias=False) if (stride != 1 or cin != out) else None

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        sc = self.shortcut(o) if self.shortcut is not None else x
        o = self.conv2(self.relu2(self.bn2(self.conv1(o))))
        return self.conv3(self.relu3(self.bn3(o))) + sc


class PreActResNetBottleneck(nn.Module):
    """ResNet-(9n+2) pre-activation bottleneck (He et al. 2016, identity mappings)."""

    def __init__(self, n: int = 1, base: int = 8):
        super().__init__()
        self.conv0 = nn.Conv2d(3, base, 3, padding=1, bias=False)
        blocks, cin = [], base
        for s, p in enumerate((base, 2 * base, 4 * base)):
            for b in range(n):
                blocks.append(PreActBottleneck(cin, p, 2 if (s > 0 and b == 0) else 1))
                cin = 4 * p
        self.blocks = nn.Sequential(*blocks)
        self.bn, self.relu = nn.BatchNorm2d(cin), nn.ReLU()
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, NC)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.relu(self.bn(self.blocks(self.conv0(x)))))))


class PreActBasic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1, self.relu1 = nn.BatchNorm2d(cin), nn.ReLU()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2, self.relu2 = nn.BatchNorm2d(cout), nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride, bias=False) if (stride != 1 or cin != cout) else None

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        sc = self.shortcut(o) if self.shortcut is not None else x
        return self.conv2(self.relu2(self.bn2(self.conv1(o)))) + sc


class WRNPreAct(nn.Module):
    """WRN-(6n+4)-k pre-activation; at k = 1 the stem width equals the first stage (identity skip on the stem)."""

    def __init__(self, n: int = 1, k: int = 1):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks, cin = [], 16
        for s, c in enumerate((16 * k, 32 * k, 64 * k)):
            for b in range(n):
                blocks.append(PreActBasic(cin, c, 2 if (s > 0 and b == 0) else 1))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.bn, self.relu = nn.BatchNorm2d(cin), nn.ReLU()
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, NC)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.relu(self.bn(self.blocks(self.conv0(x)))))))


class ConvMixerReLU(nn.Module):
    """ConvMixer-h/d with ReLU; ``kernel`` 1 is the dense shape twin of the depthwise mixer, ``kernel`` 9 the dense k x k twin."""

    def __init__(self, h: int = 32, depth: int = 2, kernel: int = 1, patch: int = 2):
        super().__init__()
        self.embed = nn.Conv2d(3, h, patch, stride=patch)
        self.act0, self.bn0 = nn.ReLU(), nn.BatchNorm2d(h)
        self.dw = nn.ModuleList([nn.Conv2d(h, h, kernel, padding=kernel // 2, bias=False) for _ in range(depth)])
        self.a1 = nn.ModuleList([nn.ReLU() for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.BatchNorm2d(h) for _ in range(depth)])
        self.pw = nn.ModuleList([nn.Conv2d(h, h, 1) for _ in range(depth)])
        self.a2 = nn.ModuleList([nn.ReLU() for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.BatchNorm2d(h) for _ in range(depth)])
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(h, NC)
        self.depth = depth

    def forward(self, x):
        x = self.bn0(self.act0(self.embed(x)))
        for i in range(self.depth):
            x = self.n1[i](self.a1[i](self.dw[i](x))) + x
            x = self.n2[i](self.a2[i](self.pw[i](x)))
        return self.fc(self.flatten(self.pool(x)))


class PyramidUnit(nn.Module):
    """BN-conv-BN-ReLU-conv-BN plus a zero-padded identity shortcut: BN on the unit input reads a sum."""

    def __init__(self, cin, cout):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn2, self.relu = nn.BatchNorm2d(cout), nn.ReLU()
        self.conv2, self.bn3 = nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.pad = cout - cin

    def forward(self, x):
        branch = self.bn3(self.conv2(self.relu(self.bn2(self.conv1(self.bn1(x))))))
        return branch + F.pad(x, (0, 0, 0, 0, 0, self.pad))


# ----------------------------------------------------------------------------- D10: BatchNorm eps after a weight layer
class WeightBnEps(nn.Module):
    """``kind`` in {conv2d, conv1d, linear}: weight -> BN(eps) -> ReLU -> FC."""

    def __init__(self, kind: str, eps: float, affine: bool = True):
        super().__init__()
        self.kind = kind
        if kind == "conv2d":
            self.w, self.bn, flat = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8, eps=eps, affine=affine), 8 * 8 * 8
        elif kind == "conv1d":
            self.w, self.bn, flat = nn.Conv1d(3, 8, 3, padding=1), nn.BatchNorm1d(8, eps=eps, affine=affine), 8 * 64
        else:
            self.w, self.bn, flat = nn.Linear(48, 32), nn.BatchNorm1d(32, eps=eps, affine=affine), 32
        self.r, self.fc = nn.ReLU(), nn.Linear(flat, NC)

    def forward(self, x):
        if self.kind == "conv1d":
            x = x.reshape(x.shape[0], 3, -1)
        elif self.kind == "linear":
            x = torch.flatten(x, 1)[:, :48]
        return self.fc(torch.flatten(self.r(self.bn(self.w(x))), 1))


SHAPE = (3, 8, 8)

D1_WITNESSES = {
    "conv_relu_bn_conv": (ConvReluBnConv, SHAPE),
    "bn_on_input": (BnOnInput, SHAPE),
    "bn_on_sum": (BnOnSum, SHAPE),
    "bn_on_concat": (BnOnConcat, SHAPE),
    "double_bn": (DoubleBn, SHAPE),
    "linear_tokens_bn": (LinearTokensBn, (1, 8, 8)),
    "preact_bottleneck_n1": (lambda: PreActResNetBottleneck(n=1), (3, 16, 16)),
    "convmixer_h32_d2_twin": (lambda: ConvMixerReLU(kernel=1), (3, 16, 16)),
    "convmixer_h32_d2_kxk_twin": (lambda: ConvMixerReLU(kernel=9), (3, 16, 16)),
    "pyramid_unit": (lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                                           PyramidUnit(8, 12), PyramidUnit(12, 16), nn.BatchNorm2d(16), nn.ReLU(),
                                           nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, NC)), SHAPE),
}

D8_WITNESSES = {
    "fanout_bn": (FanoutBn, SHAPE),
    "fanout_act": (FanoutAct, SHAPE),
    "stem_two_consumers": (StemTwoConsumers, SHAPE),
    "linear_fanout_bn": (LinearFanoutBn, SHAPE),
    "wrn_preact_n1_k1": (lambda: WRNPreAct(n=1, k=1), (3, 16, 16)),
}

SPIKING_ONLY_WITNESSES = {
    "act_then_act": (ActThenAct, SHAPE),
}

EPS_VALUES = (1e-5, 1e-3, 0.1)
EPS_KINDS = ("conv2d", "conv1d", "linear")
