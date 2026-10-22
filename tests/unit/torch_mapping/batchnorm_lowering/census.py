"""The census oracle of the absorption plan and the defect-free census members (arith_census.py, arith_models.py).

``census_defects`` is the predicate ``absorption_defects`` of the arith audit, ported: D1 lists every BatchNorm the
plan does not fold directly after a weight layer (or an Identity), D8 every folded op whose chain fans out before it.
The members are the arith section-5 forms that need no family builder, at reduced widths (declared local).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.cifar_models import CifarBasicBlock, CifarResNet
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.models.vit_leaf import LeafVisionTransformer
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityAnalyzer
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .witnesses import NC

_WEIGHT = (nn.Linear, nn.Conv1d, nn.Conv2d)


def traced_plan(model: nn.Module, shape, packaging=MVM_PACKAGING):
    gm = normalize_fx_graph(trace_model(model, shape, device="cpu"))
    return gm, RepresentabilityAnalyzer(gm, packaging=packaging).analyze()


def census_defects(gm, plan) -> dict[str, list[str]]:
    mods = dict(gm.named_modules())
    d1, d8 = [], []
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        mod = mods.get(str(node.target))
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            inp = node.args[0] if node.args else None
            direct = (isinstance(inp, type(node)) and inp.op == "call_module"
                      and isinstance(mods.get(str(inp.target)), (*_WEIGHT, nn.Identity)))
            if node.name not in plan or not direct:
                d1.append(str(node.target))
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(mods.get(str(node.target)), _WEIGHT):
            continue
        cur = node
        while True:
            absorbed_users = [u for u in cur.users if u.name in plan]
            if not absorbed_users:
                break
            nxt = absorbed_users[0]
            if len(cur.users) > 1:
                d8.append(str(nxt.target))
            cur = nxt
    return {"D1": sorted(set(d1)), "D8": sorted(set(d8))}


# ----------------------------------------------------------------------------- members (arith_models.py, reduced widths)
def resnet18_cifar(w: float = 0.25) -> nn.Module:
    widths = [max(8, int(c * w / 8 + 0.5) * 8) for c in (64, 128, 256, 512)]

    class R18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
            self.stem_bn, self.stem_relu = nn.BatchNorm2d(widths[0]), nn.ReLU()
            blocks, cin = [], widths[0]
            for s, c in enumerate(widths):
                for b in range(2):
                    blocks.append(CifarBasicBlock(cin, c, stride=2 if (s > 0 and b == 0) else 1))
                    cin = c
            self.blocks = nn.Sequential(*blocks)
            self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, NC)

        def forward(self, x):
            x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
            return self.fc(self.flatten(self.pool(self.blocks(x))))

    return R18()


def resnet110_cifar() -> nn.Module:
    return CifarResNet((3, 32, 32), NC, blocks_per_stage=18, base_width=8)


class InvRes(nn.Module):
    """Inverted residual, shape twin: the depthwise 3x3 becomes a dense 1x1 (stride 1) or 2x2 (stride 2)."""

    def __init__(self, cin, cout, stride, t):
        super().__init__()
        h = cin * t
        layers = [nn.Conv2d(cin, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6()] if t != 1 else []
        dw = nn.Conv2d(h, h, 1, bias=False) if stride == 1 else nn.Conv2d(h, h, 2, stride=2, bias=False)
        layers += [dw, nn.BatchNorm2d(h), nn.ReLU6(), nn.Conv2d(h, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.body = nn.Sequential(*layers)
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_res else y


class MobileNetV2Twin(nn.Module):
    def __init__(self, cfg=((1, 8, 1, 1), (6, 8, 2, 1), (6, 16, 2, 2))):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU6())
        blocks, cin = [], 8
        for t, c, n, s in cfg:
            for i in range(n):
                blocks.append(InvRes(cin, c, s if i == 0 else 1, t))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6())
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, NC)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.head(self.blocks(self.stem(x))))))


class ShuffleUnitTwin(nn.Module):
    def __init__(self, cin, cout, stride, hw_out):
        super().__init__()
        self.stride, self.half, self.hw, self.cout = stride, cout // 2, hw_out, cout
        b = cout // 2

        def dw(c):
            return nn.Conv2d(c, c, 1, bias=False) if stride == 1 else nn.Conv2d(c, c, 2, stride=2, bias=False)

        if stride == 2:
            self.branch1 = nn.Sequential(dw(cin), nn.BatchNorm2d(cin), nn.Conv2d(cin, b, 1, bias=False),
                                         nn.BatchNorm2d(b), nn.ReLU())
            c2in = cin
        else:
            self.branch1 = None
            c2in = b
        self.branch2 = nn.Sequential(nn.Conv2d(c2in, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                                     dw(b), nn.BatchNorm2d(b), nn.Conv2d(b, b, 1, bias=False), nn.BatchNorm2d(b),
                                     nn.ReLU())

    def forward(self, x):
        if self.stride == 1:
            out = torch.cat([x[:, :self.half], self.branch2(x[:, self.half:])], 1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], 1)
        h = self.hw
        return out.view(-1, 2, self.half, h, h).transpose(1, 2).contiguous().view(-1, self.cout, h, h)


class ShuffleNetV2Twin(nn.Module):
    def __init__(self, widths=(16, 32), repeats=(2, 2), stem=8, last=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem), nn.ReLU())
        units, cin, hw = [], stem, 32
        for c, r in zip(widths, repeats):
            hw //= 2
            for i in range(r):
                units.append(ShuffleUnitTwin(cin, c, 2 if i == 0 else 1, hw))
                cin = c
        self.units = nn.Sequential(*units)
        self.conv5 = nn.Sequential(nn.Conv2d(cin, last, 1, bias=False), nn.BatchNorm2d(last), nn.ReLU())
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(last, NC)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.conv5(self.units(self.stem(x))))))


class ResNeXtBlockTwin(nn.Module):
    def __init__(self, cin, inner, cout, stride):
        super().__init__()
        c2 = nn.Conv2d(inner, inner, 1, bias=False) if stride == 1 else nn.Conv2d(inner, inner, 2, stride=2, bias=False)
        self.c1, self.b1, self.r1 = nn.Conv2d(cin, inner, 1, bias=False), nn.BatchNorm2d(inner), nn.ReLU()
        self.c2, self.b2, self.r2 = c2, nn.BatchNorm2d(inner), nn.ReLU()
        self.c3, self.b3 = nn.Conv2d(inner, cout, 1, bias=False), nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
                   if (stride != 1 or cin != cout) else None)
        self.r3 = nn.ReLU()

    def forward(self, x):
        o = self.b3(self.c3(self.r2(self.b2(self.c2(self.r1(self.b1(self.c1(x))))))))
        return self.r3(o + (x if self.sc is None else self.sc(x)))


class ResNeXt29Twin(nn.Module):
    def __init__(self, inner0=16, out0=32, stem=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem), nn.ReLU())
        blocks, cin = [], stem
        for s in range(3):
            inner, cout = inner0 * (2 ** s), out0 * (2 ** s)
            blocks.append(ResNeXtBlockTwin(cin, inner, cout, 2 if s > 0 else 1))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, NC)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.blocks(self.stem(x)))))


class ResMixerBlock(nn.Module):
    def __init__(self, s, d, ds, dc):
        super().__init__()
        self.tok1, self.tok_bn, self.tok_act, self.tok2 = nn.Linear(s, ds), ChannelsLastBatchNorm1d(ds), nn.ReLU(), nn.Linear(ds, s)
        self.ch1, self.ch_bn, self.ch_act, self.ch2 = nn.Linear(d, dc), ChannelsLastBatchNorm1d(dc), nn.ReLU(), nn.Linear(dc, d)

    def forward(self, x):
        y = x.transpose(1, 2)
        x = x + self.tok2(self.tok_act(self.tok_bn(self.tok1(y)))).transpose(1, 2)
        return x + self.ch2(self.ch_act(self.ch_bn(self.ch1(x))))


class ResMixer(nn.Module):
    def __init__(self, patch=8, d=32, ds=16, dc=64, depth=2):
        super().__init__()
        s = (32 // patch) ** 2
        self.embed, self.embed_bn = nn.Conv2d(3, d, patch, stride=patch), nn.BatchNorm2d(d)
        self.blocks = nn.Sequential(*[ResMixerBlock(s, d, ds, dc) for _ in range(depth)])
        self.head = nn.Linear(d, NC)

    def forward(self, x):
        x = self.embed_bn(self.embed(x)).flatten(2).transpose(1, 2)
        return self.head(self.blocks(x).mean(dim=1))


def vit_lite() -> nn.Module:
    return LeafVisionTransformer(image_size=32, patch_size=8, in_channels=3, embed_dim=32, num_heads=2,
                                 depth=2, mlp_ratio=2.0, num_classes=NC)


CENSUS_SHAPE = (3, 32, 32)
CENSUS_MEMBERS = {
    "resnet18_w1_4": resnet18_cifar,
    "resnet110": resnet110_cifar,
    "mobilenetv2_twin": MobileNetV2Twin,
    "shufflenetv2_twin": ShuffleNetV2Twin,
    "resnext29_twin": ResNeXt29Twin,
    "residual_mixer": ResMixer,
    "vit_lite": vit_lite,
}
