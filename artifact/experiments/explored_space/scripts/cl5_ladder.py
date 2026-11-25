""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parents[3] / "12_tcad_readiness" / "e2e" / "driver"
CIFAR = (3, 32, 32)


K0_FIELDS = ("checkpoint", "sha256", "sha256_equals_inventory", "missing", "unexpected", "shape_mismatched",
             "module_keys", "lanes", "lane_sites_matched", "grids")


MEMBERS_REGISTERED: Dict[str, Dict[str, Any]] = {}
_LADDER: Dict[str, Any] = {}


def load_ladder(wt) -> Any:
    ""
    if "module" in _LADDER:
        return _LADDER["module"]
    scripts = str(Path(wt) / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import scaling_ladder as ladder
    _LADDER.update(module=ladder, resolved_platform=ladder.resolved_platform, builder=ladder._builder)
    ladder.resolved_platform = resolved_platform
    ladder._builder = _builder
    return ladder


def resolved_platform(target: str, fraction_or_declared, packed: Optional[int],
                      declared: Optional[Dict[str, Any]] = None, core_fields: Optional[Dict[str, Any]] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ""
    ladder, original = _LADDER["module"], _LADDER["resolved_platform"]
    bits = (declared or {}).get("weight_bits")
    if bits is None:
        return original(target, fraction_or_declared, packed, declared, core_fields, overrides)
    spec_of = ladder.target_spec
    ladder.target_spec = lambda name, _spec=spec_of, _bits=int(bits): {**_spec(name), "weight_bits": _bits}
    try:
        return original(target, fraction_or_declared, packed, declared, core_fields, overrides)
    finally:
        ladder.target_spec = spec_of


def _driver():
    for p in (DRIVER, DRIVER / "members"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import inventory
    import loader
    return inventory, loader


def member_checkpoints(member) -> Dict[str, Any]:
    ""
    return {c.path: c for c in (member.float_ckpt, *member.qat.values()) if c is not None}


def k0_record(k0: Dict[str, Any]) -> Dict[str, Any]:
    ""
    return {k: v for k, v in k0.items() if k != "lane_sites"} | {"lane_sites_matched": len(k0.get("lane_sites", {}))}


def row_checkpoint(member, result: Dict[str, Any], item: str):
    ""
    k0, inputs = result.get("K0") or {}, result.get("inputs") or {}
    path = k0.get("checkpoint") or inputs.get("checkpoint")
    digest = k0.get("sha256") or inputs.get("checkpoint_sha256")
    own = member_checkpoints(member)
    if path not in own:
        raise SystemExit(f"--member {item}: the row's checkpoint {path!r} is no inventory checkpoint of {member.id} "
                         f"({', '.join(sorted(own))})")
    if digest != own[path].sha256:
        raise SystemExit(f"--member {item}: the row's checkpoint sha256 {digest} is not the inventory's "
                         f"{own[path].sha256} ({path})")
    return own[path]


def _builder(key: str):
    ""
    if key.startswith("member:"):
        name = key.split(":", 1)[1]
        spec = MEMBERS_REGISTERED[name]
        inventory, loader = _driver()
        member = inventory.member(spec["member"])
        ckpt = member_checkpoints(member)[spec["checkpoint"]]

        def build(shape, classes, config):
            model, _split, k0 = loader.load_checkpoint(member, ckpt)
            record = k0_record(k0)
            spec["k0"] = {f: record.get(f) for f in K0_FIELDS}
            spec["weights"] = f"checkpoint sha256 {record['sha256']}"
            print(f"member {name}: K0 {record['checkpoint']} sha256 {record['sha256']} (equals the inventory: "
                  f"{record['sha256_equals_inventory']}), {record['module_keys']} module keys loaded strictly, "
                  f"{record['lane_sites_matched']} of {record['lanes']} lanes matched, {record['grids']} grids",
                  file=sys.stderr, flush=True)
            return model
        return build
    return _LADDER["builder"](key)


def register_member(item: str) -> str:
    ""
    key, _, row_dir = item.partition("=")
    row = Path(row_dir)
    if not key or not (row / "result.json").exists() or not (row / "config.json").exists():
        raise SystemExit(f"--member {item}: KEY=ROW_DIR names no row with config.json and result.json")
    if key in MEMBERS_REGISTERED:
        return key
    result = json.loads((row / "result.json").read_text())
    config = json.loads((row / "config.json").read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"--member {item}: the row's result is {result.get('status')!r}, not complete")
    inventory, _loader = _driver()
    try:
        member = inventory.member(result.get("member"))
    except Exception as exc:
        raise SystemExit(f"--member {item}: member {result.get('member')!r} is no route (b) inventory member ({exc})")
    ckpt = row_checkpoint(member, result, item)
    MEMBERS_REGISTERED[key] = {
        "member": member.id, "row": row.name, "label": member.label, "dataset": str(config.get("data_provider_name")),
        "checkpoint": ckpt.path, "classes": 100}
    ladder = _LADDER["module"]
    ladder.ROSTER[key] = (f"member:{key}", {}, 100, CIFAR)
    return key


def split_argv(argv: Sequence[str]):
    ""
    if not argv:
        raise SystemExit(__doc__)
    wt, rest, members = argv[0], [], []
    it = iter(argv[1:])
    for a in it:
        if a == "--member":
            members.append(next(it))
        else:
            rest.append(a)
    return wt, members, rest


def main(argv: Optional[Sequence[str]] = None) -> int:
    wt, members, rest = split_argv(list(sys.argv[1:] if argv is None else argv))
    ladder = load_ladder(wt)
    for item in members:
        register_member(item)
    return int(ladder.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
