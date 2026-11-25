""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
RA = HERE.parents[2]

G7_ROSTER = RA / "10_wave1" / "g7_requirement_census" / "runs" / "census" / "zoo_roster.json"
G7_ROSTER_SHA16 = "32e4382b5b29de6d"
G1_COVERAGE = RA / "10_wave1" / "g1_lowering" / "coverage_table.json"
G1_COVERAGE_SHA16 = "72363e8342c02226"
NET_MODELS = RA / "09_wave0" / "runs" / "net_models.py"
NET_MODELS_SHA16 = "a538dbb191e833f5"

TORCHVISION = "torchvision"
TEST_MODEL = "test model"
SELFTEST = "selftest"


TV_BUILDER_KWARGS = {"googlenet": {"aux_logits": False, "init_weights": True},
                     "inception_v3": {"aux_logits": False, "init_weights": True}}
TV_CLASSES = 1000
TV_CHANNELS = 3
NET_CLASSES = 10
VIT_LITE_CLASSES = 100
VIT_LITE_SHAPE = (3, 32, 32)
DECOMPOSED = "[decomposed]"


class MemberSourceError(RuntimeError):
    ""


def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _pinned(path: Path, expected: str) -> Path:
    found = sha16(path)
    if found != expected:
        raise MemberSourceError(f"{path} has sha256 {found}, the census pins {expected}")
    return path


def g7_roster() -> dict:
    return json.loads(_pinned(G7_ROSTER, G7_ROSTER_SHA16).read_text())


def _g1_coverage() -> dict:
    return json.loads(_pinned(G1_COVERAGE, G1_COVERAGE_SHA16).read_text())


def _member(mid: str, population: str, source: str, name: str, classes: int, decompose: bool,
            declared: dict, kwargs: Optional[dict] = None, note: str = "") -> dict:
    return {"id": mid, "population": population, "source": source, "name": name, "num_classes": classes,
            "decompose_leaf_modules": decompose, "builder_kwargs": kwargs or {}, "declared_input": declared,
            "note": note}


def torchvision_members() -> List[dict]:
    ""
    roster = g7_roster()
    names = [m["name"] for m in roster["members"]]
    if len(names) != roster["listed"] or len(set(names)) != len(names):
        raise MemberSourceError(f"the roster lists {roster['listed']} builders, names {len(set(names))} distinct")
    declared = {"kind": "image", "channels": TV_CHANNELS, "resizable": True,
                "minimum_side_from": "torchvision DEFAULT weights meta['min_size']"}
    return [_member(n, TORCHVISION, "torchvision.models", n, TV_CLASSES, False, dict(declared),
                    TV_BUILDER_KWARGS.get(n), note=f"G7 roster {G7_ROSTER.name} (sha {G7_ROSTER_SHA16})")
            for n in names]


def test_members() -> List[dict]:
    ""
    table = _g1_coverage()
    out: List[dict] = []
    nets = [r["member"] for r in table["rows"] if r["family"] == "net_models"]
    shapes = _net_shapes()
    for key in nets:
        for decompose in (False, True):
            out.append(_member(f"net_models/{key}{DECOMPOSED if decompose else ''}", TEST_MODEL, "net_models", key,
                               NET_CLASSES, decompose,
                               {"kind": "fixed", "shape": list(shapes[key]),
                                "shape_from": f"net_models.instances() (sha {NET_MODELS_SHA16})"},
                               note=f"G1 coverage table row {key}"))
    forms = sorted({(r["form"], bool(r["decompose_key"])) for r in table["vit_lite_7_4"] if r["contract"] == "mvm"},
                   key=lambda fk: (fk[0] != "softmax", fk[1]))
    for form, decompose in forms:
        out.append(_member(f"vit_lite_7_4_{form}{DECOMPOSED if decompose else ''}", TEST_MODEL,
                           "mimarsinan.models.families.vit_lite", "ViTLiteCCT", VIT_LITE_CLASSES, decompose,
                           {"kind": "fixed", "shape": list(VIT_LITE_SHAPE),
                            "shape_from": "G1 coverage_rows_r1_g1.py:81, shape = (3, 32, 32)"},
                           {"num_classes": VIT_LITE_CLASSES, "attention": form},
                           note=f"G1 coverage table ViT-Lite-7/4 ({form})"))
    if any(r["decomposed"] for r in table["vit_b_16"]):
        out.append(_member(f"vit_b_16{DECOMPOSED}", TEST_MODEL, "torchvision.models", "vit_b_16", TV_CLASSES, True,
                           {"kind": "image", "channels": TV_CHANNELS, "resizable": True,
                            "minimum_side_from": "torchvision DEFAULT weights meta['min_size']"},
                           note="G1 coverage table ViT-B/16, decomposed"))
    return out


def selftest_members() -> List[dict]:
    ""
    declared = {"kind": "fixed", "shape": [3, 16, 16], "shape_from": "selftest"}
    return [_member(f"selftest/{a}", SELFTEST, "selftest", a, 10, False, dict(declared),
                    note="harness self-test, never a census member")
            for a in ("convert", "loud", "alloc", "slow", "kill", "near_kill", "fast_oom")]


def census_members() -> List[dict]:
    return torchvision_members() + test_members()


def by_id(members: List[dict]) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for m in members:
        if m["id"] in found:
            raise MemberSourceError(f"member id {m['id']} twice")
        found[m["id"]] = m
    return found


def _net_models():
    spec = importlib.util.spec_from_file_location("cl_cov_net_models", _pinned(NET_MODELS, NET_MODELS_SHA16))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _net_shapes() -> Dict[str, tuple]:
    return {key: tuple(shape) for key, _family, _factory, shape, _mode in _net_models().instances()}


def _net_factory(key: str) -> Callable:
    table = {k: f for k, _family, f, _shape, _mode in _net_models().instances()}
    return table[key]


class _SelftestNet:
    ""

    @staticmethod
    def build():
        import torch.nn as nn

        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 16 * 16, 10))


def build_member(member: dict):
    ""
    import torch

    torch.manual_seed(0)
    source = member["source"]
    with contextlib.redirect_stdout(io.StringIO()):
        if source == "torchvision.models":
            import torchvision.models as tvm

            model = tvm.get_model(member["name"], **member["builder_kwargs"])
        elif source == "net_models":
            model = _net_factory(member["name"])()
        elif source == "mimarsinan.models.families.vit_lite":
            from mimarsinan.models.families.vit_lite import ViTLiteCCT

            model = ViTLiteCCT(**member["builder_kwargs"])
        elif source == "selftest":
            model = _SelftestNet.build()
        else:
            raise MemberSourceError(f"{member['id']}: unknown source {source!r}")
    return model.eval()


def listed_builders() -> List[str]:
    ""
    import torchvision.models as tvm

    return sorted(tvm.list_models(module=tvm))


def roster_problems(listed: List[str], counts: Dict[str, int], roster: dict) -> List[str]:
    ""
    pinned = {m["name"]: m for m in roster["members"]}
    problems = []
    if sorted(listed) != sorted(pinned):
        problems.append(f"torchvision lists {len(listed)} builders, the roster {len(pinned)}: "
                        f"{sorted(set(listed) ^ set(pinned))}")
    for name, count in sorted(counts.items()):
        want = pinned.get(name)
        if want is not None and count != want["params"]:
            problems.append(f"{name}: {count} parameters now, the roster {want['params']}")
    return problems


def listed_check() -> List[str]:
    ""
    return roster_problems(listed_builders(), {}, g7_roster())


def enumeration_check() -> dict:
    ""
    import torch
    import torchvision
    import torchvision.models as tvm

    roster = g7_roster()
    listed = listed_builders()
    counts = {}
    for name in listed:
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            model = tvm.get_model(name, **TV_BUILDER_KWARGS.get(name, {}))
        counts[name] = int(sum(p.numel() for p in model.parameters()))
        del model
    return {"torchvision": torchvision.__version__, "roster_torchvision": roster.get("torchvision"),
            "listed": len(listed), "roster_listed": roster["listed"], "params": counts,
            "problems": roster_problems(listed, counts, roster)}
