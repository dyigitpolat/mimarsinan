"""Lane call sites (WI-5e): the IR op of each host op the lane plan absorbs.

The plan names its host ops by call site: a node of the mapper graph, one per call of a module in the network. One
module object can be called at several sites (an activation a block calls after each of its sums), so its payload
reaches several ops of the IR, of which the plan may absorb some and leave others on the host. A role is therefore
keyed by call site, never by payload alone: a site's op carries the site's payload (``params['module']``) under the
site's name, the name its mapper gives the op it emits. A site without a name is told by its payload alone, which
must then reach one op only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from mimarsinan.mapping.ir import ComputeOp, IRGraph
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from mimarsinan.mapping.lanes.plan import LanePlan

#: A planned host op's role in fusion: ``(kind, argument)``, the argument an add's operand index or a pool's owner key.
Role = Tuple[str, object]


def planned_call_sites(plan: LanePlan, first_owner: int) -> List[Tuple[Any, Role]]:
    """Every host op ``plan`` absorbs, as its call site and its role: each perceptron lane's add, reduction and
    clamp, then each pool lane (owner keys from ``first_owner`` on)."""
    sites: List[Tuple[Any, Role]] = []
    for lane in plan.perceptron_lanes:
        if lane.add is not None:
            sites.append((lane.add, ("add", lane.operand_index)))
        if lane.reduce is not None:
            sites.append((lane.reduce, ("reduce", None)))
        if lane.clamp is not None:
            sites.append((lane.clamp, ("clamp", None)))
    for k, lane in enumerate(plan.pool_lanes):
        sites.append((lane.pool, ("pool", first_owner + k)))
    return sites


def call_site_roles(ir_graph: IRGraph, sites: Sequence[Tuple[Any, Role]]) -> Tuple[Dict[int, Role], int]:
    """``({IR op id: role}, number of sites the IR carries)``: each site's op is the one carrying its payload under
    its name (its payload alone for an unnamed site). A payload reaching several such ops is refused by name; a site
    the IR does not carry is not counted (fusion refuses the count)."""
    carriers: Dict[int, List[ComputeOp]] = {}
    for node in ir_graph.nodes:
        if isinstance(node, ComputeOp):
            carriers.setdefault(id(node.params.get("module")), []).append(node)
    roles: Dict[int, Role] = {}
    for site, role in sites:
        name = getattr(site, "name", None)
        ops = carriers.get(id(site.module), [])
        if name:
            ops = [op for op in ops if op.name == name]
        if len(ops) > 1:
            raise LaneFusionError(f"lane {role[0]} {name!r}: its module runs at {len(ops)} ops of the IR "
                                  f"({sorted(str(op.name) for op in ops)[:3]}) and its call site has no name that "
                                  f"tells them apart")
        if not ops:
            continue
        if int(ops[0].id) in roles:
            raise LaneFusionError(f"lane {role[0]} {name!r}: IR op {ops[0].name!r} is already another planned call "
                                  f"site's")
        roles[int(ops[0].id)] = role
    return roles, len(roles)
