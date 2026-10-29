"""[U-L] Layout records of the cores the lane fusion inserts after the layout walk (a pool lane's 0/1 reduction
cores, a producer-side reduction's reduced cores).

The layout walk records every core it emits (``IRGraph.layout_softcores``, ``NeuralCore.layout_softcore_index``),
and the pass planner reads a core's spec off its record; the position-multiplexing planner orders passes by the
records' dependency levels and refuses an IR whose cores carry none. The fusion, which inserts cores after the walk, gives
each of them a record of its own here: the spec the walk would state for the core in the program the pass leaves
(its compacted extent, its dependency level there, the IR's latency, and no bank; segment 0, the one frame of order
facts a segment holding such a core shares), appended to the IR's records, its index on the core. The core is
marked (``lane_law.INSERTED_RECORD``) because the walk's own records predate the insertion: a consumer of the walk's facts
(the planner's one source of order facts per segment, the residency class the finalizer keyed on a core's values)
tells the two apart, and a consumer that needs a record finds one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from mimarsinan.mapping.ir.lane_law import INSERTED_RECORD, operand_axons
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_neural_core
from mimarsinan.mapping.platform.core_residency import ungrouped_fallback_id


def record_inserted_cores(ir_graph: Any, cores: Iterable[Any]) -> int:
    """Append a layout record per inserted core of ``ir_graph`` (in the order given) and mark each core; a core that
    carries a record already keeps it, and an IR without the walk's records gets none (every core of it is
    reconstructed alike). A core's bias row (an always-on axon, the IR mapping's bias without an on-core register)
    counts among its axons, as the walk counts it; the IR's latency must be computed (the level the record states).
    Returns the number of records written."""
    records = list(getattr(ir_graph, "layout_softcores", None) or [])
    if not records:
        return 0
    written = 0
    for core in cores:
        if getattr(core, "layout_softcore_index", None) is not None:
            continue
        if getattr(core, "latency", None) is None:
            raise ValueError(f"core {core.name!r}: an inserted core's layout record states its dependency level, and "
                             "the IR's latency was not computed")
        index = len(records)
        spec = spec_from_neural_core(core, hardware_bias=getattr(core, "hardware_bias", None) is not None,
                                     fallback_residency_class_id=ungrouped_fallback_id(index))
        records.append(replace(spec, operand_axons=operand_axons(core)))
        core.layout_softcore_index = index
        setattr(core, INSERTED_RECORD, True)
        written += 1
    if written:
        ir_graph.layout_softcores = records
    return written
