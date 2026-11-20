"""The identity test's one tree run: a fixture planned, built and priced on the tree at ``--tree``, printed as three
digests (the canonical plan, the program, the priced record) and a few plain facts.

The standing identity test (``test_thru_identity.py``) runs this file twice per fixture, once on the export of record
(the paper version, read only) and once on this branch, each in its own interpreter whose ``mimarsinan`` and test
helpers are that tree's own (``<tree>/src``, ``<tree>/tests``, ``<tree>/scripts``), and asserts the digests equal.

- **Plan**: every plan the one pass planner (``pass_planner.plan_segment_passes``, which both planes call) returns while
  the fixture is planned, built and priced, in call order, with the shape-only plane's own answer where the fixture has
  one.
- **Program**: the built hybrid program (C1: the candidate plane's pass programs, the program it prices).
- **Priced**: the sealed record of the built program (``executed_record``), its quantities (``from_record``) and its
  absolute pricing at a declared physics; C1: the candidate plane's priced axes as W48's driver prices them.

The canonical form walks every object generically: mappings, sequences, arrays and tensors by content, dataclasses by
their fields, other objects by their attributes. An attribute or a dataclass field holding None is left out, so a
declared-only field (absent when undeclared) leaves the form unchanged. Wall-clock readings and per-run tokens are
left out by name (``WALL_KEYS``). Under ``--declare false`` the new keys' own declarations (``NEW_KEYS``), wherever a
declaration is echoed, are left out with the config digest that hashes them, so the run is compared with the paper
version's run of the same fixture without them.

usage: python -s -B identity_digests.py --tree ROOT --fixture NAME [--declare absent|false]
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: The branch's new keys of the layer copies (PL/thru/design.md (b)): declared false, each is the absent program.
NEW_KEYS = ("layer_replication", "wavefront_group_cores")
DECLARATIONS = {"absent": {}, "false": {"layer_replication": False}}
#: Readings of the wall clock and per-run tokens (an IR build's random cache-trust token, which its mapping echoes),
#: never part of a plan, a program or a price.
WALL_KEYS = frozenset({"wall_s", "created_at", "build_token", "source_ir_build_token"})
#: The digest of a declared platform the sealed record carries (it hashes the declarations, the new keys included).
DECLARATION_DIGESTS = frozenset({"config_digest"})
#: W48's pipelined driver (read only): the candidate plane's pricing of a route (b) member on a study chip.
W48DIR = Path("/home/yigit/repos/research_stuff/papers/mimarsinan_tcad_2026/research_artifacts/12_tcad_readiness/"
              "e2e/w48")
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
def _w02_value(extra):
    from unit.mapping.position_multiplexing import vehicles as W02
    from unit.mapping.position_multiplexing.test_programs import VALUE_CORES

    program = W02.integer_resnet()
    hybrid = W02.build(program.ir, VALUE_CORES, True, **extra)
    shape_only = W02.fragments(program.ir, VALUE_CORES, True, **extra)
    platform = {"cores": VALUE_CORES, "core_semantics": "mvm", "activation_bits": 8, "allow_scheduling": True,
                "max_schedule_passes": W02.PASSES, "position_multiplexing": True, **extra}
    return {"shape_only": shape_only}, hybrid, sealed_and_priced(hybrid, platform, weight_bits=8, label="w02 value")


def _w02_lif(extra):
    from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
    from unit.mapping.position_multiplexing import vehicles as W02
    from unit.mapping.position_multiplexing.test_programs import LIF_CORES

    ir = W02.lif_ir(W02.lif_conv_chain(), 64, 16)
    hybrid = W02.build(ir, LIF_CORES, True, allow_neuron_splitting=True, **extra)
    shape_only = W02.fragments(ir, LIF_CORES, True, allow_neuron_splitting=True, **extra)
    platform = {"cores": LIF_CORES, "allow_scheduling": True, "max_schedule_passes": W02.PASSES,
                "position_multiplexing": True, "allow_neuron_splitting": True, **extra}
    return {"shape_only": shape_only}, hybrid, sealed_and_priced(
        hybrid, platform, weight_bits=8, label="w02 lif", timesteps=W02.T, pass_transfer=transfer_for(streamed=True))


def _pipe_block(count):
    def build(extra):
        from unit.mapping.pipelined_multiplexing import vehicles as V

        block = V.lane_program("block")
        hybrid = V.build(block.ir, V.cores(count), True, **extra)
        platform = {"cores": V.cores(count), "core_semantics": "mvm", "activation_bits": 4, "allow_scheduling": True,
                    "max_schedule_passes": V.PASSES, "position_multiplexing": True, "pipelined_multiplexing": True,
                    "allow_coalescing": True, "allow_neuron_splitting": True, **extra}
        return {}, hybrid, sealed_and_priced(hybrid, platform, weight_bits=4, label=f"lane block {count}")
    return build


def _pipe_lif(extra):
    from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
    from unit.mapping.pipelined_multiplexing import vehicles as V
    from unit.mapping.position_multiplexing import vehicles as W02

    ir = W02.lif_ir(W02.lif_conv_chain(), 64, 16)
    hybrid = V.build(ir, V.LIF_CORES, True, allow_coalescing=False, **extra)
    platform = {"cores": V.LIF_CORES, "allow_scheduling": True, "max_schedule_passes": V.PASSES,
                "position_multiplexing": True, "pipelined_multiplexing": True, "allow_neuron_splitting": True, **extra}
    return {}, hybrid, sealed_and_priced(hybrid, platform, weight_bits=8, label="pipelined lif",
                                         timesteps=W02.T, pass_transfer=transfer_for(streamed=True))


def _planes(fixture, count):
    def build(extra):
        from mimarsinan.deployment_record.introspection import scaling_ladder as LD
        from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
        from unit.mapping.lanes.lane_planes import STEPS, TRANSFER, ladder, network_of
        from unit.mapping.lanes.test_lane_shape_walk import FIXTURES, _fixture

        platform, program = _fixture(fixture, ("L", "E"), multiplexing=True, axons=256, neurons=64, count=count,
                                     passes=1 << 16, coalescing=True, hardware_bias=True)
        platform.update({"pipelined_multiplexing": True, **extra})
        network = network_of(program, FIXTURES[fixture][2])
        census = LD.shape_census(network, platform)
        scheduled = LD.schedule_program(network, census, platform, ladder().program_facts,
                                        tiles=LD.packed_tiles(census, platform))
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, platform)
        return ({"census": census, "scheduled": scheduled}, hybrid,
                sealed_and_priced(hybrid, platform, weight_bits=4, label=f"{fixture} {count}", timesteps=STEPS,
                                  pass_transfer=TRANSFER))
    return build


def _c1_compass_k16(extra, tree: Path):
    """Route (b)'s member C1 (ResNet-18 at 32 x 32, its B5m row's declarations) on the COMPASS-size study chip at
    batch 16, the one-copy arm, PIPE and PIPE with its 1 MiB buffer, priced on the candidate plane by W48's driver (the
    plane the comparison prices)."""
    spec = importlib.util.spec_from_file_location("w48_pipelined_identity", W48DIR / "w48_pipelined.py")
    WP = importlib.util.module_from_spec(spec)
    sys.modules["w48_pipelined_identity"] = WP
    spec.loader.exec_module(WP)
    W48 = WP.W48
    tc = WP.PipelinedToolchain(tree)
    member = dict(WP.with_pipelined_arm(W48.route_members())["C1"])
    arms = (W48.ONE_COPY, WP.PIPE, WP.PIPE_BUFFER)
    pipelined = (WP.PIPE, WP.PIPE_BUFFER)  # the arms the new keys are relevant on
    member["arms"] = {a: member["arms"][a] for a in arms}
    member["declare"] = {a: ({**(member["declare"].get(a) or {}), **extra} if extra and a in pipelined
                             else member["declare"].get(a)) for a in arms}
    programs: List[Any] = []  # the candidate plane's pass programs, per scheduled arm
    schedule = tc.schedule

    def scheduled_program(*args, **kwargs):
        out = schedule(*args, **kwargs)
        programs.append(out[1] if out[0] is None else (out[0].refusal, out[0].noc))
        return out

    tc.schedule = scheduled_program
    plans = record_the_planner()
    priced = W48.price_members(tc, {"C1": member}, ["C1"], arms, (BATCH_K,), ("candidate",), ("study:shipped",),
                               W48.DEFAULT_RECORD_ROOT, 3600.0, lambda line: None, chips={"C1": "study_compass"})
    return plans, programs, priced


FIXTURES: Dict[str, Callable] = {
    "w02_value": _w02_value,
    "w02_lif": _w02_lif,
    "pipe_block_8": _pipe_block(8),
    "pipe_block_3": _pipe_block(3),
    "pipe_block_1": _pipe_block(1),
    "pipe_lif": _pipe_lif,
    "planes_residual_8": _planes("residual", 8),
    "planes_residual_3": _planes("residual", 3),
}
#: Fixtures priced on the candidate plane by W48's driver (no builder program: the plane states its pass programs).
CANDIDATE_FIXTURES = ("c1_compass_k16",)


def run(tree: Path, fixture: str, declare: str) -> Dict[str, Any]:
    extra = DECLARATIONS[declare]
    drop = WALL_KEYS | (frozenset(NEW_KEYS) | DECLARATION_DIGESTS if extra else frozenset())
    if fixture in CANDIDATE_FIXTURES:
        plans, programs, priced = _c1_compass_k16(extra, tree)
        facts = {arm: {k: (v.get("passes") if isinstance(v, dict) else v) for k, v in out["candidate"].items()}
                 for arm, out in priced["C1"]["arms"].items()}
        return {"plan": digest(plans, drop), "program": digest(programs, drop), "priced": digest(priced, drop),
                "facts": {"plans": len(plans), "passes": facts}}
    plans = record_the_planner()
    shape_only, hybrid, priced = FIXTURES[fixture](extra)
    return {"plan": digest({"planner": plans, **shape_only}, drop), "program": digest(hybrid, drop),
            "priced": digest(priced, drop), "facts": {"plans": len(plans), "passes": passes(hybrid),
                                                      "pricing": _pricing_fact(priced["pricing"])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--fixture", required=True, choices=sorted(FIXTURES) + list(CANDIDATE_FIXTURES))
    ap.add_argument("--declare", default="absent", choices=sorted(DECLARATIONS))
    args = ap.parse_args(argv)
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", here)]
    tree = args.tree.resolve()
    sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
    import torch
    torch.set_num_threads(1)
    import warnings
    warnings.simplefilter("ignore")
    print(json.dumps(run(tree, args.fixture, args.declare), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
