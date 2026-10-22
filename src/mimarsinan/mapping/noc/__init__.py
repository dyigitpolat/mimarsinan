"""Shape-only NoC fragments of a candidate layout (placements + wire census)."""

from mimarsinan.mapping.noc.wire_census import (
    LayoutWireCensus,
    census_of_walk,
    record_emission_census,
)
from mimarsinan.mapping.noc.fragments import (
    LayoutNocFragments,
    collect_noc_fragments,
    execution_stage_latencies, execution_stage_placements,
)
from mimarsinan.mapping.noc.value_traffic import (
    ValueNocCensusError, value_noc_bytes, value_noc_wire_hops,
)

__all__ = [
    "LayoutNocFragments",
    "LayoutWireCensus",
    "census_of_walk",
    "collect_noc_fragments",
    "execution_stage_latencies",
    "execution_stage_placements",
    "record_emission_census",
    "ValueNocCensusError",
    "value_noc_bytes",
    "value_noc_wire_hops",
]
