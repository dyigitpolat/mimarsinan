"""Shape-only NoC fragments of a candidate layout (placements + wire census)."""

from mimarsinan.mapping.noc.wire_census import (
    LayoutWireCensus,
    census_of_walk,
    record_emission_census,
)
from mimarsinan.mapping.noc.fragments import (
    LayoutNocFragments,
    collect_noc_fragments,
    execution_stage_latencies,
)

__all__ = [
    "LayoutNocFragments",
    "LayoutWireCensus",
    "census_of_walk",
    "collect_noc_fragments",
    "execution_stage_latencies",
    "record_emission_census",
]
