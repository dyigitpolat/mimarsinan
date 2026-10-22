"""E276: the mapper walk is iterative, so mapping depth is not bounded by the recursion limit.

Spec `07_engineering_refinements.md` E276 (V42F00 item 1): a 111-block
pre-activation bottleneck chain (the block count of ResNet-1001) and a 100-layer
concatenation chain, built here from the layer types the toolchain converts at a
declared small width (the test measures depth, not width), convert, map and build
their shape-only layout with the interpreter's default recursion limit in force;
the recursive reference walk raises ``RecursionError`` on the bottleneck chain at
that limit (so the chain is deep enough to test the walk); and the IR of CIFAR
ResNet-18 equals the recursive walk's byte for byte, node ids and order included.
The chains are D1 witnesses (pre-activation), whose values are not this test's
subject, so they convert with ``strict`` off and the assertions are structural.
Both chains are order witnesses as well as depth witnesses: the bottleneck chain
changes width at each of its three stages, so the first block of a stage adds a
projection shortcut to its main path (two neural sources), and each concatenation
layer is DenseNet-BC's (BN-ReLU-conv1x1-BN-ReLU-conv3x3) with its new features
first, so the recursive reference walk reaches the previous layer through the
layer's own convolutions and overflows at the default limit on this chain too.
"""

from __future__ import annotations

import ast
import resource
import sys
import time
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.mappers.module_mapper import ModuleMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.verification.verifier.mapping_verifier_soft import verify_soft_core_mapping
from mimarsinan.models.cifar_models import CifarBasicBlock
from mimarsinan.torch_mapping.converter import convert_torch_model

from unit.mapping.recursive_walk_reference import (
    DEFAULT_RECURSION_LIMIT,
    ir_bytes,
    layout_bytes,
    recursive_walk,
)

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
SMALL = (3, 4, 4)
CORE = {"max_axons": 256, "max_neurons": 256, "count": 100_000}


class _PreActBottleneck(nn.Module):
    """BN-ReLU-conv1x1-BN-ReLU-conv3x3-BN-ReLU-conv1x1 plus an identity or projection skip."""

    def __init__(self, cin: int, planes: int):
        super().__init__()
        out = 4 * planes
        self.bn1, self.relu1 = nn.BatchNorm2d(cin), nn.ReLU()
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn2, self.relu2 = nn.BatchNorm2d(planes), nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn3, self.relu3 = nn.BatchNorm2d(planes), nn.ReLU()
        self.conv3 = nn.Conv2d(planes, out, 1, bias=False)
        self.shortcut = nn.Conv2d(cin, out, 1, bias=False) if cin != out else None

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        skip = self.shortcut(o) if self.shortcut is not None else x
        o = self.conv2(self.relu2(self.bn2(self.conv1(o))))
        return self.conv3(self.relu3(self.bn3(o))) + skip


class _BottleneckChain(nn.Module):
    """Three stages of 37 blocks at planes 1, 2 and 3: the first block of stages 2 and 3 projects its skip."""

    def __init__(self, blocks: int = 111, stage_planes: tuple = (1, 2, 3)):
        super().__init__()
        per_stage = blocks // len(stage_planes)
        assert per_stage * len(stage_planes) == blocks
        self.stem = nn.Conv2d(3, 4 * stage_planes[0], 3, padding=1, bias=False)
        layers, cin = [], 4 * stage_planes[0]
        for planes in stage_planes:
            for _ in range(per_stage):
                layers.append(_PreActBottleneck(cin, planes))
                cin = 4 * planes
        self.blocks = nn.Sequential(*layers)
        self.bn, self.relu = nn.BatchNorm2d(cin), nn.ReLU()
        self.pool, self.flatten = nn.AvgPool2d(SMALL[1]), nn.Flatten()
        self.fc = nn.Linear(cin, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.blocks(self.stem(x))))
        return self.fc(self.flatten(self.pool(x)))


class _ConcatLayer(nn.Module):
    """DenseNet-BC's layer (BN-ReLU-conv1x1-BN-ReLU-conv3x3), its new features concatenated first."""

    def __init__(self, cin: int, growth: int):
        super().__init__()
        inner = 2 * growth
        self.bn1, self.relu1 = nn.BatchNorm2d(cin), nn.ReLU()
        self.conv1 = nn.Conv2d(cin, inner, 1, bias=False)
        self.bn2, self.relu2 = nn.BatchNorm2d(inner), nn.ReLU()
        self.conv2 = nn.Conv2d(inner, growth, 3, padding=1, bias=False)

    def forward(self, x):
        new = self.conv2(self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x))))))
        return torch.cat([new, x], 1)


class _ConcatChain(nn.Module):
    def __init__(self, layers: int = 100, growth: int = 1, stem: int = 2):
        super().__init__()
        width = stem + layers * growth
        self.stem = nn.Conv2d(3, stem, 3, padding=1, bias=False)
        self.layers = nn.Sequential(*[_ConcatLayer(stem + i * growth, growth) for i in range(layers)])
        self.bn, self.relu = nn.BatchNorm2d(width), nn.ReLU()
        self.pool, self.flatten = nn.AvgPool2d(SMALL[1]), nn.Flatten()
        self.fc = nn.Linear(width, 4)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.relu(self.bn(self.layers(self.stem(x)))))))


def _resnet18_cifar(width: float) -> nn.Module:
    """CIFAR ResNet-18 from ``CifarBasicBlock`` (3x3 stem, no max-pool, [2, 2, 2, 2]),
    as `RA/06_build/audit_v42/mapping/family_probe.py:60-85` builds it."""
    widths = [max(8, int(c * width / 8 + 0.5) * 8) for c in (64, 128, 256, 512)]

    class _R18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
            self.stem_bn, self.stem_relu = nn.BatchNorm2d(widths[0]), nn.ReLU()
            blocks, cin = [], widths[0]
            for stage, planes in enumerate(widths):
                for b in range(2):
                    blocks.append(CifarBasicBlock(cin, planes, stride=2 if (stage > 0 and b == 0) else 1))
                    cin = planes
            self.blocks = nn.Sequential(*blocks)
            self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 10)

        def forward(self, x):
            x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
            return self.fc(self.flatten(self.pool(self.blocks(x))))

    return _R18()


CHAINS = {"bottleneck_111": _BottleneckChain, "concat_100": _ConcatChain}
PACKAGINGS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


def _convert(model: nn.Module, shape, packaging):
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return convert_torch_model(model.eval(), shape, 4, packaging=packaging, strict=False)


def _identity_ir(flow):
    return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None).map(
        flow.get_mapper_repr()
    )


def _layout_softcores(flow):
    return LayoutIRMapping(max_axons=CORE["max_axons"], max_neurons=CORE["max_neurons"]).collect_layout_softcores(
        flow.get_mapper_repr()
    )


@pytest.mark.parametrize("packaging", sorted(PACKAGINGS))
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_deep_chain_converts_maps_and_lays_out_at_the_default_limit(chain, packaging):
    assert sys.getrecursionlimit() == DEFAULT_RECURSION_LIMIT
    torch.manual_seed(0)
    start = time.time()
    flow = _convert(CHAINS[chain](), SMALL, PACKAGINGS[packaging])
    ir = _identity_ir(flow)
    assert ir.get_neural_cores()
    soft = verify_soft_core_mapping(flow.get_mapper_repr(), CORE["max_axons"], CORE["max_neurons"])
    assert soft.feasible and soft.error is None, soft.error
    plan = build_layout_plan(soft, [dict(CORE)])
    assert plan.feasible, plan.errors
    assert sys.getrecursionlimit() == DEFAULT_RECURSION_LIMIT
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[E276] {chain}/{packaging}: {len(ir.nodes)} IR nodes, {len(soft.softcores)} softcores, "
          f"wall {time.time() - start:.1f} s, process peak RSS {peak_gib:.2f} GiB (a measurement)")


@pytest.mark.parametrize("packaging", sorted(PACKAGINGS))
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_recursive_reference_overflows_on_the_deep_chain(chain, packaging):
    flow = _convert(CHAINS[chain](), SMALL, PACKAGINGS[packaging])
    assert sys.getrecursionlimit() == DEFAULT_RECURSION_LIMIT
    with recursive_walk(), pytest.raises(RecursionError):
        _identity_ir(flow)


def _reference_equality(flow, limit):
    """(iterative IR, recursive IR, iterative layout, recursive layout) as bytes."""
    repr_ = flow.get_mapper_repr()
    walked = ir_bytes(_identity_ir(flow)), layout_bytes(_layout_softcores(flow))
    repr_.clear_ir_caches()
    with recursive_walk(limit):
        reference = ir_bytes(_identity_ir(flow)), layout_bytes(_layout_softcores(flow))
    repr_.clear_ir_caches()
    return walked, reference


@pytest.mark.parametrize("packaging", sorted(PACKAGINGS))
def test_resnet18_ir_equals_the_recursive_walk_byte_for_byte(packaging):
    torch.manual_seed(0)
    flow = _convert(_resnet18_cifar(1 / 8), (3, 32, 32), PACKAGINGS[packaging])
    walked, reference = _reference_equality(flow, limit=20_000)
    assert walked[0] == reference[0]
    assert walked[1] == reference[1]


@pytest.mark.parametrize("packaging", sorted(PACKAGINGS))
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_deep_chain_ir_equals_the_recursive_walk_under_a_raised_limit(chain, packaging):
    torch.manual_seed(0)
    flow = _convert(CHAINS[chain](), SMALL, PACKAGINGS[packaging])
    walked, reference = _reference_equality(flow, limit=20_000)
    assert walked == reference


def test_a_direct_caller_maps_a_deep_chain_without_recursion():
    """``Mapper.map_to_ir`` itself walks iteratively, not only through ``ModelRepresentation``."""
    node = InputMapper((1, 4))
    for _ in range(5 * DEFAULT_RECURSION_LIMIT):
        node = ModuleMapper(node, nn.Identity())
    sources = node.map_to_ir(IRMapping())
    assert sources.shape == (1, 4)
    assert [s.node_id for s in sources.flat] == [-2] * 4


def test_only_the_base_mapper_defines_the_walk():
    """No mapper overrides ``map_to_ir``, so every mapper is reached by the one iterative walk."""
    owners = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if any(isinstance(f, ast.FunctionDef) and f.name == "map_to_ir" for f in cls.body):
                owners.append(f"{path.relative_to(SRC_ROOT).as_posix()}::{cls.name}")
    assert sorted(owners) == [
        "mapping/mappers/base.py::Mapper",
        "mapping/model_representation.py::ModelRepresentation",
    ]
