""

from __future__ import annotations

import torch.fx
import torch.nn as nn

from screen_quant import QConv2d, QLinear


def crossbars_in_order(gm: torch.fx.GraphModule) -> list[tuple[str, nn.Module]]:
    ""
    modules = dict(gm.named_modules())
    out = []
    for node in gm.graph.nodes:
        if node.op == "call_module" and isinstance(modules[node.target], (QConv2d, QLinear)):
            out.append((node.target, modules[node.target]))
    return out


def _shape(m: nn.Module) -> tuple[int, int, int]:
    if isinstance(m, nn.Conv2d):
        return int(m.in_channels), int(m.out_channels), int(m.kernel_size[0])
    return int(m.in_features), int(m.out_features), 1


def assign_stage_bits(gm: torch.fx.GraphModule, units: list[dict], bits) -> dict:
    ""
    bits = tuple(int(b) for b in bits)
    xbars = crossbars_in_order(gm)
    if len(xbars) != len(units):
        raise ValueError(f"{len(xbars)} crossbars in the graph, {len(units)} census units")
    stages: dict[str, dict] = {}
    for (target, m), u in zip(xbars, units):
        if _shape(m) != (int(u["cin"]), int(u["cout"]), int(u["k"])):
            raise ValueError(f"crossbar {target} {_shape(m)} does not match census unit {u['name']} "
                             f"({u['cin']}, {u['cout']}, {u['k']})")
        if int(m.weight.numel()) != int(u["cin"]) * int(u["cout"]) * int(u["k"]) ** 2:
            raise ValueError(f"crossbar {target} has {m.weight.numel()} weights, census unit {u['name']} counts "
                             f"{int(u['cin']) * int(u['cout']) * int(u['k']) ** 2}")
        s = int(u["stage"])
        m.w_bits = bits[s]
        row = stages.setdefault(str(s + 1), {"bits": bits[s], "crossbars": [], "units": [], "params": 0})
        row["crossbars"].append(target)
        row["units"].append(u["name"])
        row["params"] += int(m.weight.numel())
    return {"bits": list(bits), "stages": stages, "crossbars": len(xbars)}


def bits_by_stage(gm: torch.fx.GraphModule, units: list[dict]) -> dict[int, set]:
    ""
    out: dict[int, set] = {}
    for (_, m), u in zip(crossbars_in_order(gm), units):
        out.setdefault(int(u["stage"]) + 1, set()).add(int(m.w_bits))
    return out
