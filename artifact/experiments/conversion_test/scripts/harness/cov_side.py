""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cov_members as CM

FLOOR = 112
STEP = 16
CEILING = 1024
SUPPLEMENT_SIDE = 224
CHECKS = ("verdict", "supplement")
RULE_TEXT = ("s(m) = max(declared minimum side, smallest side >= F = 112, in steps of 16, that the native model "
             "accepts with every weight tap (every convolution kernel tap) reading input data); the declared minimum "
             "side is the DEFAULT weights' min_size, and none when the native model as built refuses every side from "
             "it to the ceiling (those weights build the network for another input); a member built for one input "
             "shape: that shape")
WEIGHTS_ONLY_TEXT = ("the native model as built (no weights) refuses every candidate from this minimum to the ceiling: "
                     "the DEFAULT weights build the network for another input (a builder argument they set), so the "
                     "minimum is theirs only and the candidates run from F")
NO_MINIMUM_TEXT = ("none: the DEFAULT weights' minimum belongs to those weights only (weights_only_minimum); the "
                   "network as built declares no minimum, so the candidates run from F")
SUPPLEMENT_TEXT = "stage 1 only at max(224, declared minimum side); a member built for one input shape: that shape"
SCHEMA = "cl_cov.side_table/1"


class SideRuleError(RuntimeError):
    ""


class SideTableExists(RuntimeError):
    ""


def _tuple(value, n: int) -> Tuple[int, ...]:
    return tuple(int(v) for v in value) if isinstance(value, (tuple, list)) else (int(value),) * n


def _padding(module, kernel: Tuple[int, ...], dilation: Tuple[int, ...]) -> Tuple[int, ...]:
    pad = module.padding
    if isinstance(pad, str):
        return tuple(0 if pad == "valid" else (d * (k - 1)) // 2 for k, d in zip(kernel, dilation))
    return _tuple(pad, len(kernel))


def taps_read(kernel: int, stride: int, padding: int, dilation: int, extent: int, out_extent: int) -> bool:
    ""
    return all(any(0 <= i * stride + r * dilation - padding < extent for i in range(out_extent))
               for r in range(kernel))


def scan(model, shape: Sequence[int]) -> dict:
    ""
    import torch
    import torch.nn as nn

    tapped = (nn.modules.conv._ConvNd,)
    adaptive = (nn.modules.pooling._AdaptiveAvgPoolNd, nn.modules.pooling._AdaptiveMaxPoolNd)
    unread: List[list] = []
    replicating: List[list] = []
    counted = {"tap_layers": 0, "adaptive_pools": 0}

    def hook(name):
        def run(mod, inputs, output):
            if isinstance(mod, tapped):
                if getattr(mod, "transposed", False):
                    return
                n = len(mod.kernel_size) if isinstance(mod.kernel_size, (tuple, list)) else inputs[0].dim() - 2
                kernel = _tuple(mod.kernel_size, n)
                stride = _tuple(mod.stride if mod.stride is not None else mod.kernel_size, n)
                dilation = _tuple(getattr(mod, "dilation", 1), n)
                padding = _padding(mod, kernel, dilation)
                extents, outs = inputs[0].shape[-n:], output.shape[-n:]
                counted["tap_layers"] += 1
                for ax in range(n):
                    if not taps_read(kernel[ax], stride[ax], padding[ax], dilation[ax], int(extents[ax]),
                                     int(outs[ax])):
                        unread.append([name, type(mod).__name__, ax, kernel[ax], int(extents[ax])])
                        break
            else:
                n = output.dim() - 2 if output.dim() > 2 else 1
                extents, outs = inputs[0].shape[-n:], output.shape[-n:]
                counted["adaptive_pools"] += 1
                if any(int(m) < int(o) for m, o in zip(extents, outs)):
                    replicating.append([name, type(mod).__name__, list(map(int, extents)), list(map(int, outs))])
        return run

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, tapped + adaptive)]
    try:
        with torch.no_grad():
            model(torch.zeros(1, *shape))
        return {"input_shape": list(shape), "accepted": True, "n_layers_with_unread_taps": len(unread),
                "unread_taps": unread[:5], "adaptive_replicating": replicating[:5], **counted}
    except Exception as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        return {"input_shape": list(shape), "accepted": False, "refusal": f"{type(exc).__name__}: {first}"}
    finally:
        for h in handles:
            h.remove()


def candidates(min_side: int) -> List[int]:
    start = max(min_side, FLOOR)
    return [start] + [s for s in range(STEP * (start // STEP + 1), CEILING + 1, STEP)]


def declared_minimum(name: str) -> Tuple[int, str]:
    import torchvision.models as tvm

    weights = tvm.get_model_weights(name).DEFAULT
    size = tuple(int(v) for v in weights.meta["min_size"])
    if len(size) != 2 or size[0] != size[1]:
        raise SideRuleError(f"{name}: declared minimum size {size} is not one square side")
    return size[0], f"{weights}.meta['min_size'] = {list(size)}"


def first_side(model, channels: int, start: int) -> Tuple[Optional[int], List[dict]]:
    ""
    scans: List[dict] = []
    for s in candidates(start):
        at = scan(model, [channels, s, s])
        scans.append(at)
        if at["accepted"] and at["n_layers_with_unread_taps"] == 0:
            return s, scans
    return None, scans


def resizable_rule(model, channels: int, min_side: int, min_from: str, label: str = "member") -> dict:
    ""
    side, scans = first_side(model, channels, min_side)
    inputs = {"kind": "declared minimum side, resizable", "min_side": min_side, "min_side_from": min_from,
              "floor": FLOOR, "step": STEP}
    if side is None and not any(a["accepted"] for a in scans):
        inputs["weights_only_minimum"] = {"min_side": min_side, "min_side_from": min_from, "reason": WEIGHTS_ONLY_TEXT,
                                          "candidates_scanned": scans}
        inputs["min_side"], inputs["min_side_from"] = None, NO_MINIMUM_TEXT
        side, scans = first_side(model, channels, FLOOR)
    if side is None:
        raise SideRuleError(f"{label}: no candidate side from {scans[0]['input_shape'][-1]} to {CEILING} is accepted "
                            f"with every tap read")
    sup_side = max(SUPPLEMENT_SIDE, inputs["min_side"] or 0)
    sup_scan = next((a for a in scans if a["input_shape"][-1] == sup_side), None) or scan(
        model, [channels, sup_side, sup_side])
    inputs["candidates_scanned"], inputs["supplement_scan"] = scans, sup_scan
    if not sup_scan["accepted"]:
        inputs["supplement_refused"] = sup_scan["refusal"]
    return {"inputs": inputs, "side": side, "supplement_side": sup_side, "supplement_accepted": sup_scan["accepted"]}


def member_entry(member: dict) -> dict:
    ""
    declared = member["declared_input"]
    model = CM.build_member(member)
    params = int(sum(p.numel() for p in model.parameters()))
    if declared["kind"] == "fixed":
        shape = list(declared["shape"])
        at = scan(model, shape)
        if not at["accepted"]:
            raise SideRuleError(f"{member['id']}: the native model refuses its own declared shape {shape}: "
                                f"{at['refusal']}")
        inputs = {"kind": "fixed input shape", "declared_shape": shape, "declared_from": declared["shape_from"],
                  "scan": [at]}
        verdict, supplement = shape, shape
        side = sup_side = (shape[-1] if len(shape) == 3 and shape[-1] == shape[-2] else None)
    else:
        min_side, min_from = declared_minimum(member["name"])
        channels = declared["channels"]
        found = resizable_rule(model, channels, min_side, min_from, label=member["id"])
        inputs, side, sup_side = found["inputs"], found["side"], found["supplement_side"]
        verdict = [channels, side, side]
        supplement = [channels, sup_side, sup_side] if found["supplement_accepted"] else None
    del model
    checks = {"verdict": {"input_shape": verdict, "side": side, "stages": "both (converted model, identity value "
                                                                         "program)"}}
    if supplement is not None:
        checks["supplement"] = {"input_shape": supplement, "side": sup_side, "stages": "stage 1 only (converted model)"}
    return {"member": member["id"], "population": member["population"], "params": params, "rule_inputs": inputs,
            "checks": checks}


def build(out: Path, population: str) -> dict:
    if out.exists():
        raise SideTableExists(f"{out} exists: the side table is fixed before any run and never rebuilt over")
    import torch
    import torchvision

    warnings.filterwarnings("ignore")
    members = CM.selftest_members() if population == "selftest" else CM.census_members()
    started = time.time()
    entries: Dict[str, dict] = {}
    for member in members:
        t0 = time.time()
        entries[member["id"]] = member_entry(member)
        v = entries[member["id"]]["checks"]["verdict"]
        print(f"{member['id']}: verdict {v['input_shape']}, supplement "
              f"{entries[member['id']]['checks'].get('supplement', {}).get('input_shape')} ({time.time() - t0:.1f} s)",
              flush=True)
    enumeration = CM.enumeration_check() if population == "census" else None
    if enumeration is not None and enumeration["problems"]:
        raise CM.MemberSourceError("; ".join(enumeration["problems"]))
    table = {"schema": SCHEMA, "population": population, "rule": RULE_TEXT, "supplement_rule": SUPPLEMENT_TEXT,
             "floor": FLOOR, "step": STEP, "ceiling": CEILING, "supplement_side": SUPPLEMENT_SIDE,
             "torch": torch.__version__, "torchvision": torchvision.__version__,
             "builder_sha16": {p.name: CM.sha16(p) for p in (Path(__file__), Path(CM.__file__))},
             "sources": {"g7_roster": [str(CM.G7_ROSTER), CM.G7_ROSTER_SHA16],
                         "g1_coverage": [str(CM.G1_COVERAGE), CM.G1_COVERAGE_SHA16],
                         "net_models": [str(CM.NET_MODELS), CM.NET_MODELS_SHA16]},
             "enumeration_check": None if enumeration is None else {k: v for k, v in enumeration.items()
                                                                     if k != "params"},
             "built_s": round(time.time() - started, 1), "members": entries}
    text = json.dumps(table, indent=1, sort_keys=True) + "\n"
    with open(out, "x") as handle:
        handle.write(text)
    return table


def load(path: Path) -> Tuple[dict, str]:
    ""
    data = Path(path).read_bytes()
    table = json.loads(data)
    if table.get("schema") != SCHEMA:
        raise SideRuleError(f"{path}: schema {table.get('schema')!r}, expected {SCHEMA}")
    return table, hashlib.sha256(data).hexdigest()[:16]


def expected(table: dict, member: str, check: str) -> Optional[dict]:
    entry = table["members"].get(member)
    return None if entry is None else entry["checks"].get(check)


def side_groups(table: dict, member_ids: Sequence[str], check: str = "verdict") -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for mid in member_ids:
        shape = table["members"][mid]["checks"][check]["input_shape"]
        key = " x ".join(map(str, shape[1:]))
        groups[key] = groups.get(key, 0) + 1
    return groups


def compare(old: dict, new: dict) -> dict:
    ""
    shared = sorted(set(old["members"]) & set(new["members"]))
    differing = [m for m in shared if old["members"][m] != new["members"][m]]
    return {"shared": len(shared), "identical": len(shared) - len(differing), "differing": differing,
            "added": sorted(set(new["members"]) - set(old["members"])),
            "removed": sorted(set(old["members"]) - set(new["members"]))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "show", "compare"))
    parser.add_argument("path")
    parser.add_argument("new", nargs="?")
    parser.add_argument("--population", choices=("census", "selftest"), default="census")
    args = parser.parse_args(argv)
    if args.command == "compare":
        (old, old_sha), (new, new_sha) = load(Path(args.path)), load(Path(args.new))
        found = compare(old, new)
        print(json.dumps({"old": [args.path, old_sha], "new": [args.new, new_sha], **found,
                          "new_entries": {m: {c: e["input_shape"] for c, e in new["members"][m]["checks"].items()}
                                          for m in found["added"]}}, indent=1))
        return 0 if not found["differing"] and not found["removed"] else 1
    if args.command == "build":
        table = build(Path(args.path), args.population)
        _, sha = load(Path(args.path))
    else:
        table, sha = load(Path(args.path))
    ids = [m for m, e in table["members"].items() if e["population"] == CM.TORCHVISION]
    print(f"side table {args.path} sha {sha}: {len(table['members'])} members")
    print("torchvision verdict sides:", side_groups(table, ids))
    print("torchvision supplement sides:", side_groups(table, [m for m in ids if "supplement" in
                                                               table["members"][m]["checks"]], "supplement"))
    tests = [m for m, e in table["members"].items() if e["population"] != CM.TORCHVISION]
    print("other members:", {m: table["members"][m]["checks"]["verdict"]["input_shape"] for m in tests})
    return 0


if __name__ == "__main__":
    sys.exit(main())
