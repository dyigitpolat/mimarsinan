"""[E1 fix round] The reference given as the IR at unbounded geometry is one (rule ``reference_geometry``): the caller
supplies it, and the verdict then names it, so the predicate holds it to what an untiled IR is before comparing.

Every neural node of the reference has a perceptron identity and holds its perceptron's whole output for its column
(``perceptron_output_slice`` from 0, as wide as the node's columns before IR pruning), exactly one node per
(perceptron, column), in no coalescing group and with no partial-sum role; every neuron and every host-op name is held
once. A tiled IR, an IR at another declared geometry, or an IR holding a relay or an inserted core fails by name."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from mimarsinan.certification.program_check.unbounded.keys import KeyedIR, node_width
from mimarsinan.mapping.ir import NeuralCore

RULE = "reference_geometry"
#: The node attributes that place a node in a coalescing or partial-sum group (none at unbounded geometry).
GROUP_ATTRIBUTES = ("coalescing_group_id", "coalescing_role", "psum_group_id", "psum_role")


def check_reference(u: KeyedIR, fail: Callable[[str, str, str], None]) -> int:
    """Fail ``reference_geometry`` once per defect of the reference ``u``; returns the nodes checked."""
    where = "the reference"
    for nid in u.unkeyed:
        fail(RULE, f"node {nid} of {where}", "a neural node with no perceptron identity (a relay, an inserted core) "
             "or a pre-pruning mask that is not its columns: the IR at unbounded geometry holds none")
    for code, held, other in u.duplicated:
        what = u.space.text(code) if code >= 0 else "one host-op name"
        fail(RULE, f"nodes {held[0]} and {other[0]} of {where}", f"both hold {what}")
    owners: Dict[Tuple[int, int], List[int]] = {}
    checked = 0
    for nid, node in u.by_id.items():
        if not isinstance(node, NeuralCore) or nid in u.unkeyed:
            continue
        checked += 1
        groups = {name: getattr(node, name, None) for name in GROUP_ATTRIBUTES}
        if any(value is not None for value in groups.values()):
            fail(RULE, f"node {nid} of {where}", f"in a coalescing or partial-sum group ({groups}): a tiling of a "
                 "fan-in across cores")
        slice_ = node.perceptron_output_slice
        perceptron = node.perceptron_index
        assert slice_ is not None and perceptron is not None    # keyed: ``perceptron_channels`` found both
        start, end = int(slice_[0]), int(slice_[1])
        mask = getattr(node, "pre_pruning_col_mask", None)
        width = len(mask) if mask is not None else node_width(node, u.graph)
        if start != 0 or end - start != width:
            fail(RULE, f"node {nid} of {where}", f"holds channels [{start}, {end}) of perceptron "
                 f"{perceptron} over {width} column(s): a tile, not its perceptron's whole output")
        column = -1 if node.perceptron_output_column is None else int(node.perceptron_output_column)
        owners.setdefault((int(perceptron), column), []).append(nid)
    for (perceptron, column), nodes in sorted(owners.items()):
        if len(nodes) > 1:
            fail(RULE, f"nodes {nodes} of {where}", f"all hold perceptron {perceptron}"
                 f"{'' if column < 0 else f' column {column}'}: one node per perceptron and column at unbounded "
                 "geometry")
    return checked
