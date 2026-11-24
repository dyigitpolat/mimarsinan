"""The residual-sum branch's identity test, one tree run: a ResNet fixture planned, built and priced on the tree at
``--tree``, printed as four digests (the canonical plan, the IR, the program, the priced record) and a few plain facts.

The form is the throughput branch's identity test (tcad/throughput-replication 1ca189c3,
``tests/identity/identity_digests.py``): the standing test (``test_rsum_identity.py``) runs this file twice per
fixture, once on the export of record (the paper version, read only) and once on this branch, each in its own
interpreter whose ``mimarsinan`` and test helpers are that tree's own (``<tree>/src``, ``<tree>/tests``), and asserts
the digests equal. The fixtures are networks whose residual additions already round once on chip (the residual
block's lane program, single and multi pass, the pipelined lane block at 8, 3 and 1 cores, the lane planes'
residual program at 8 and 3 cores): the rule of this branch (a constant-zero pad over an on-chip value is wiring) must
leave every one of them byte for byte as the paper version builds it.

- **Plan**: every plan the one pass planner (``pass_planner.plan_segment_passes``) returns while the fixture is
  planned, built and priced, in call order, with the shape-only plane's own answer where the fixture has one.
- **IR**: the fixture's IR graph after Soft Core Mapping's law hook (lanes fused, laws stamped, the fusion report).
- **Program**: the built hybrid program.
- **Priced**: the sealed record of the built program (``executed_record``), its quantities (``from_record``) and its
  absolute pricing at a declared physics.

The canonical form walks every object generically (the throughput branch's ``Canon``, unchanged): mappings,
sequences, arrays and tensors by content, dataclasses by their fields, other objects by their attributes; an attribute
holding None is left out; wall-clock readings and per-run tokens are left out by name (``WALL_KEYS``).

usage: python -s -B identity_digests.py --tree ROOT --fixture NAME
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: Readings of the wall clock and per-run tokens (an IR build's random cache-trust token, which its mapping echoes),
#: never part of a plan, an IR, a program or a price.
WALL_KEYS = frozenset({"wall_s", "created_at", "build_token", "source_ir_build_token"})
#: The physics a vehicle's sealed record is priced with (a shipped profile of every tree) and the batch it declares.
PHYSICS_PROFILE = "generic_estimated_22nm"
BATCH_K = 16
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


# --------------------------------------------------------------------------------------------- the canonical form
class Canon:
    """One canonical walk: shared objects by their first visit's ordinal, cycles closed by reference."""

    def __init__(self, drop: frozenset) -> None:
        self.drop = drop
        self.seen: Dict[int, int] = {}
        self.keep: List[Any] = []  # holds every visited object, so no id is reused during the walk

    def __call__(self, obj: Any) -> Any:
        import numpy as np
        import torch

        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, int) and not isinstance(obj, enum.Enum):
            return obj if type(obj) is int else {"int": type(obj).__qualname__, "v": int(obj)}
        if isinstance(obj, float):
            return {"f": obj.hex()}
        if isinstance(obj, complex):
            return {"c": [obj.real.hex(), obj.imag.hex()]}
        if isinstance(obj, bytes):
            return {"b": hashlib.sha256(obj).hexdigest()}
        if isinstance(obj, np.generic):
            return {"np": type(obj).__name__, "v": self(obj.item())}
        if isinstance(obj, enum.Enum):
            return {"enum": _name(type(obj)), "v": self(obj.value)}
        if isinstance(obj, type):
            return {"type": _name(obj)}
        if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType,
                            functools.partial)):
            return {"callable": _callable_name(obj)}
        if isinstance(obj, Path):
            return {"path": str(obj)}
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return {"nd": "object", "shape": list(obj.shape), "items": [self(x) for x in obj.ravel().tolist()]}
            a = np.ascontiguousarray(obj)
            return {"nd": str(a.dtype), "shape": list(a.shape), "sha": hashlib.sha256(a.tobytes()).hexdigest()}
        if torch.is_tensor(obj):
            t = obj.detach().cpu().contiguous()
            raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
            return {"tensor": str(t.dtype), "shape": list(t.shape), "sha": hashlib.sha256(raw).hexdigest()}
        key = id(obj)
        if key in self.seen:
            return {"ref": self.seen[key]}
        self.seen[key] = len(self.seen)
        self.keep.append(obj)
        if isinstance(obj, dict) or _is_mapping(obj):
            return {"map": _name(type(obj)), "items": [[self(k), self(v)] for k, v in obj.items()
                                                       if not (isinstance(k, str) and k in self.drop)]}
        if isinstance(obj, (list, tuple)):
            return {"seq": _name(type(obj)), "items": [self(x) for x in obj]}
        if isinstance(obj, (set, frozenset)):
            items = sorted(json.dumps(Canon(self.drop)(x), sort_keys=True) for x in obj)
            return {"set": _name(type(obj)), "items": items}
        if dataclasses.is_dataclass(obj):
            fields = [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]
            extra = [(k, v) for k, v in vars(obj).items() if k not in {f.name for f in dataclasses.fields(obj)}] \
                if hasattr(obj, "__dict__") else []
            return {"dc": _name(type(obj)), "fields": self._attrs(fields + extra)}
        state = _attributes(obj)
        if state is not None:
            return {"obj": _name(type(obj)), "attrs": self._attrs(state)}
        return {"repr": _name(type(obj)), "v": _ADDRESS.sub("", repr(obj))}

    def _attrs(self, pairs) -> list:
        return [[k, self(v)] for k, v in pairs if v is not None and k not in self.drop]


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _callable_name(obj: Any) -> str:
    func = getattr(obj, "func", None) or obj
    return f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', getattr(func, '__name__', type(func)))}"


def _is_mapping(obj: Any) -> bool:
    from collections.abc import Mapping
    return isinstance(obj, Mapping)


def _attributes(obj: Any) -> Optional[list]:
    pairs: list = []
    if hasattr(obj, "__dict__"):
        pairs += list(vars(obj).items())
    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()) or ():
            if slot not in ("__dict__", "__weakref__") and hasattr(obj, slot):
                pairs.append((slot, getattr(obj, slot)))
    return pairs if (pairs or hasattr(obj, "__dict__")) else None


def digest(obj: Any, drop: frozenset) -> str:
    text = json.dumps(Canon(drop)(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------------------------- the planner's plans
def record_the_planner() -> List[Any]:
    """Every plan the one pass planner returns from here on, in call order (both planes call it)."""
    from mimarsinan.mapping.support.schedule import pass_planner as PP

    plans: List[Any] = []
    original = PP.plan_segment_passes

    def planned(*args, **kwargs):
        plan = original(*args, **kwargs)
        plans.append(plan)
        return plan

    for module in list(sys.modules.values()):
        if getattr(module, "plan_segment_passes", None) is original:
            setattr(module, "plan_segment_passes", planned)
    return plans


# --------------------------------------------------------------------------------------------- pricing helpers
def sealed_and_priced(hybrid: Any, platform: Dict[str, Any], *, weight_bits: int, label: str,
                      timesteps: Optional[int] = None, pass_transfer: Optional[str] = None) -> Dict[str, Any]:
    """The sealed record of ``hybrid`` on ``platform``, its quantities and its absolute pricing at the declared physics
    (``PHYSICS_PROFILE``, batch ``BATCH_K``)."""
    from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
    from mimarsinan.deployment_record.design_point.pricing import executed_record
    from mimarsinan.deployment_record.platform_physics import get_platform_physics
    from mimarsinan.deployment_record.quantities.from_record import from_record

    record = executed_record(hybrid, platform, weight_bits=weight_bits, label=label, timesteps=timesteps,
                             pass_transfer=pass_transfer)
    physics = get_platform_physics(PHYSICS_PROFILE).to_dict()
    priced_platform = {**record.identity.platform, "platform_physics_resolved": physics, "inference_batch_K": BATCH_K}
    priced = dataclasses.replace(record, identity=dataclasses.replace(record.identity, platform=priced_platform))
    return {"record": record.to_dict(), "quantities": from_record(record), "pricing": absolute_pricing_for_record(priced)}


def _pricing_fact(pricing: Any) -> str:
    """The absolute pricing's kind and, where it refuses, its refusal."""
    refusal = getattr(pricing, "refusal", None)
    return _name(type(pricing)) + ("" if refusal is None else f": {str(refusal)[:160]}")


def passes(hybrid: Any) -> int:
    return sum(1 for s in hybrid.stages if s.kind == "neural")


# --------------------------------------------------------------------------------------------- the fixtures
def _lane_residual(packed: Dict[str, Any]):
    """The residual block's lane program (lanes L, E and P, 4 bits), packed on ``packed``'s cores, as
    ``unit.mapping.lanes.test_lane_programs`` builds it (single pass, or multi pass with ``passes``)."""
    def build():
        from unit.chip_simulation.value_run.integer_programs import residual_block_net
        from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform

        lane = lane_platform()
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane)
        hybrid = program.packed_mapping(**packed)
        core = {"max_axons": packed["axons"], "max_neurons": packed["neurons"], "count": packed.get("count", 999)}
        platform = {**lane, "cores": [core], "core_semantics": "mvm", "allow_neuron_splitting": True,
                    "allow_scheduling": "passes" in packed, "max_schedule_passes": packed.get("passes", 8)}
        return {}, program.ir, hybrid, sealed_and_priced(hybrid, platform, weight_bits=4, label="lane residual")
    return build


def _pipe_block(count):
    def build():
        from unit.mapping.pipelined_multiplexing import vehicles as V

        block = V.lane_program("block")
        hybrid = V.build(block.ir, V.cores(count), True)
        platform = {"cores": V.cores(count), "core_semantics": "mvm", "activation_bits": 4, "allow_scheduling": True,
                    "max_schedule_passes": V.PASSES, "position_multiplexing": True, "pipelined_multiplexing": True,
                    "allow_coalescing": True, "allow_neuron_splitting": True}
        return {}, block.ir, hybrid, sealed_and_priced(hybrid, platform, weight_bits=4, label=f"lane block {count}")
    return build


def _planes(fixture, count):
    def build():
        from mimarsinan.deployment_record.introspection import scaling_ladder as LD
        from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
        from unit.mapping.lanes.lane_planes import STEPS, TRANSFER, ladder, network_of
        from unit.mapping.lanes.test_lane_shape_walk import FIXTURES, _fixture

        platform, program = _fixture(fixture, ("L", "E"), multiplexing=True, axons=256, neurons=64, count=count,
                                     passes=1 << 16, coalescing=True, hardware_bias=True)
        platform.update({"pipelined_multiplexing": True})
        network = network_of(program, FIXTURES[fixture][2])
        census = LD.shape_census(network, platform)
        scheduled = LD.schedule_program(network, census, platform, ladder().program_facts,
                                        tiles=LD.packed_tiles(census, platform))
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, platform)
        return ({"census": census, "scheduled": scheduled}, program.ir, hybrid,
                sealed_and_priced(hybrid, platform, weight_bits=4, label=f"{fixture} {count}", timesteps=STEPS,
                                  pass_transfer=TRANSFER))
    return build


FIXTURES: Dict[str, Callable] = {
    "lane_residual_block": _lane_residual(dict(axons=256, neurons=32)),
    "lane_residual_block_multi_pass": _lane_residual(dict(axons=256, neurons=32, count=24, passes=128)),
    "pipe_block_8": _pipe_block(8),
    "pipe_block_3": _pipe_block(3),
    "pipe_block_1": _pipe_block(1),
    "planes_residual_8": _planes("residual", 8),
    "planes_residual_3": _planes("residual", 3),
}


def run(fixture: str) -> Dict[str, Any]:
    drop = WALL_KEYS
    plans = record_the_planner()
    shape_only, ir, hybrid, priced = FIXTURES[fixture]()
    return {"plan": digest({"planner": plans, **shape_only}, drop), "ir": digest(ir, drop),
            "program": digest(hybrid, drop), "priced": digest(priced, drop),
            "facts": {"plans": len(plans), "passes": passes(hybrid), "pricing": _pricing_fact(priced["pricing"]),
                      "lanes": getattr(ir, "lane_fusion_report", {}).get("plan")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
    args = ap.parse_args(argv)
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", here)]
    tree = args.tree.resolve()
    sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
    import torch
    torch.set_num_threads(1)
    import warnings
    warnings.simplefilter("ignore")
    print(json.dumps(run(args.fixture), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
