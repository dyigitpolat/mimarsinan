""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

import common

LANE_PREFIX = "lane_quant."
_LANE_FIELDS = ("act_theta", "act_shift", "_extra_state")
_GRID_KEY = re.compile(r"^(?P<site>.+)\.(?P<grid>[a-z]+_grid)\.(?P<field>step|_extra_state)$")


class LoadRefusal(SystemExit):
    ""


@dataclass
class SplitState:
    module: dict
    lanes: dict = field(default_factory=dict)
    grids: dict = field(default_factory=dict)


def split_state(state: dict) -> SplitState:
    ""
    module, lane_raw, grid_raw = {}, {}, {}
    for key, value in state.items():
        if key.startswith(LANE_PREFIX):
            node, _, fld = key[len(LANE_PREFIX):].rpartition(".")
            if fld not in _LANE_FIELDS or not node:
                raise LoadRefusal(f"REFUSED: lane key {key!r} outside the lane layout {_LANE_FIELDS}")
            lane_raw.setdefault(node, {})[fld] = value
            continue
        m = _GRID_KEY.match(key)
        if m:
            grid_raw.setdefault(f"{m['site']}.{m['grid']}", {})[m["field"]] = value
            continue
        module[key] = value
    lanes = {}
    for node, fields in lane_raw.items():
        missing = [f for f in _LANE_FIELDS if f not in fields]
        if missing:
            raise LoadRefusal(f"REFUSED: lane {node!r} lacks {missing}")
        extra = dict(fields["_extra_state"]) if isinstance(fields["_extra_state"], dict) else {"raw": fields["_extra_state"]}
        lanes[node] = {"theta": float(fields["act_theta"]), "shift": float(fields["act_shift"]),
                       **{k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in extra.items()}}
    grids = {}
    for site, fields in grid_raw.items():
        if "step" not in fields:
            raise LoadRefusal(f"REFUSED: grid {site!r} has no step")
        step = fields["step"]
        grids[site] = {"step": [float(v) for v in torch.as_tensor(step).flatten().tolist()],
                       "extra": fields.get("_extra_state") if isinstance(fields.get("_extra_state"), (dict, type(None)))
                       else str(fields.get("_extra_state"))}
    return SplitState(module, lanes, grids)


def load_strict(model: torch.nn.Module, module_state: dict) -> dict:
    ""
    own = model.state_dict()
    shape = sorted(k for k, v in module_state.items() if k in own and tuple(own[k].shape) != tuple(v.shape))
    missing = sorted(set(own) - set(module_state))
    unexpected = sorted(set(module_state) - set(own))
    if shape or missing or unexpected:
        raise LoadRefusal(f"REFUSED: strict load: {len(missing)} missing {missing[:4]}, {len(unexpected)} unexpected "
                          f"{unexpected[:4]}, {len(shape)} shape-mismatched {shape[:4]}")
    model.load_state_dict(module_state, strict=True)
    return {"missing": 0, "unexpected": 0, "shape_mismatched": 0, "module_keys": len(module_state)}


def lane_sites(model: torch.nn.Module, lanes: dict, input_shape=(3, 32, 32)) -> dict:
    ""
    import torch.fx

    from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

    plain = {n.name: n for n in torch.fx.symbolic_trace(model).graph.nodes}
    toolchain = {n.name: n for n in trace_model(model, tuple(input_shape)).graph.nodes}
    sites, problems = {}, []
    for name in sorted(lanes):
        a, b = plain.get(name), toolchain.get(name)
        if a is None or b is None:
            problems.append(f"{name}: harness trace {'has' if a is not None else 'lacks'} it, toolchain tracer "
                            f"{'has' if b is not None else 'lacks'} it")
            continue
        if (a.op, str(a.target)) != (b.op, str(b.target)):
            problems.append(f"{name}: harness {a.op} {a.target} != toolchain {b.op} {b.target}")
            continue
        sites[name] = {"op": a.op, "target": str(a.target)}
    if problems:
        raise LoadRefusal("REFUSED: lane sites do not match one toolchain site each: " + "; ".join(problems[:6]))
    return sites


def grid_sites(model: torch.nn.Module, grids: dict) -> dict:
    ""
    modules = dict(model.named_modules())
    return {g: (type(modules[g.rsplit(".", 1)[0]]).__name__ if g.rsplit(".", 1)[0] in modules else None)
            for g in sorted(grids)}


def build_member(member, num_classes: int = 100) -> torch.nn.Module:
    ""
    import e2e_members
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    builder = ModelRegistry.get_builder_cls(member.model_type)("cpu", (3, 32, 32), num_classes, {})
    model = builder.build(dict(member.model_config))
    params = sum(p.numel() for p in model.parameters())
    if params != member.params:
        raise LoadRefusal(f"REFUSED: {member.id} builds {params} parameters, declared {member.params}")
    return model


def load_checkpoint(member, ckpt, *, sites: bool = True) -> tuple[torch.nn.Module, SplitState, dict]:
    ""
    digest = common.sha256_file(ckpt.file)
    if digest != ckpt.sha256:
        raise LoadRefusal(f"REFUSED: {ckpt.path} sha256 {digest} != inventory {ckpt.sha256}")
    state = torch.load(ckpt.file, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    split = split_state(state)
    model = build_member(member)
    k0 = {"checkpoint": ckpt.path, "sha256": digest, "sha256_equals_inventory": True, **load_strict(model, split.module),
          "lanes": len(split.lanes), "grids": len(split.grids)}
    if sites and split.lanes:
        k0["lane_sites"] = lane_sites(model, split.lanes)
    if split.grids:
        owners = grid_sites(model, split.grids)
        k0["grid_owner_types"] = owners
        k0["grids_without_builder_module"] = sorted(g for g, t in owners.items() if t is None)
    model.eval()
    return model, split, k0
