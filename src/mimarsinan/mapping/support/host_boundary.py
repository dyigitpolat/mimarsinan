"""[U-J] The host-chip boundary of a program's shape-only walk: what crosses between the host and the chip per
inference beside the pass carry, and the host's memory-bound work.

Counted by the carry census's rule (``schedule/pass_carry.carried_softcore_spans``: a producer's whole published
slice, once per direction, however many consumers read it): the network input when an on-chip core reads it, each
host operator's output an on-chip core reads (host to chip); each on-chip slice a host operator or the network
output reads (chip to host). One transfer per direction at each such boundary: the input in, the readout out, and per
host operator its on-chip inputs out and its output in. The MAC-free host operators (element-wise, pooling: no rule
of ``module_macs`` counts a MAC) read and write their elements in host memory, at the host executor's float32
(``HOST_VALUE_BYTES``). The walk records each host operator's sizes as it walks (``LayoutIRMapping._host_io``); a
walk without its wire census states no census (it cannot tell which cores read the network input)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set

from mimarsinan.mapping.layout.layout_source_view_ops import node_ids_of
from mimarsinan.mapping.support.module_macs import module_macs

#: The host executor evaluates its operators in float32 (``chip_simulation.hybrid_run.host_compute``).
HOST_VALUE_BYTES = 4


@dataclass(frozen=True)
class HostBoundaryCensus:
    """Elements crossing per direction, host-chip transfers, and the MAC-free host operators' element traffic."""

    to_chip_elements: int
    from_chip_elements: int
    transfers: int
    memory_bound_elements: int


def _is_mac_free(params: Any, in_shape: Any, out_shape: Any) -> bool:
    module = params.get("module") if hasattr(params, "get") else None
    return module is None or not module_macs(module, in_shape, out_shape)


def host_boundary_census(layout: Any, *, input_elements: int) -> Optional[HostBoundaryCensus]:
    """The boundary census of a finished walk (``LayoutIRMapping``), or None without its wire census."""
    wires = getattr(layout, "_node_wire_census", None)
    if not wires:
        return None
    neural: Set[int] = {int(n) for n, is_neural in layout._node_is_neural.items() if is_neural}
    host = {int(n): io for n, io in getattr(layout, "_host_io", {}).items()}
    producers = {int(n): {int(p) for p in ids} for n, ids in layout._node_input_node_ids.items()}

    def slice_of(node: int) -> int:
        index = layout._node_id_to_softcore_idx.get(node)
        return 0 if index is None else int(layout.layout_softcores[index].output_count)

    read_back = {h: producers.get(h, set()) & neural for h in host}
    readout = {int(p) for p in node_ids_of(layout.output_sources)} & neural
    read_on_chip = {p for n in neural for p in producers.get(n, set()) if p in host}
    entry = int(input_elements) if any(int(cells) for _pairs, cells, _on in (wires.get(n, ({}, 0, 0))
                                                                              for n in neural)) else 0
    back = set().union(*read_back.values()) if read_back else set()
    return HostBoundaryCensus(
        to_chip_elements=entry + sum(int(host[h][1]) for h in read_on_chip),
        from_chip_elements=sum(slice_of(p) for p in back | readout),
        transfers=int(bool(entry)) + int(bool(readout - back)) + sum(1 for h in host if read_back[h])
        + len(read_on_chip),
        memory_bound_elements=sum(int(io[0]) + int(io[1]) for io in host.values() if _is_mac_free(*io[2:])),
    )
