"""The monotone conversion frontier: one concept, walked under the D-hat gate.

A conversion is walked 0→1 as a monotone frontier of discrete units — spike
segments (the P4 prefix ramp), cascade hops (hop-staged AQ, the hop-prefix
ramp) — one unit per gated fast-ladder rung, with a keep-best DFQ repair at
each rung (`reaffine`) and the bounded P1'' stage at the terminal position
(`endpoint_recovery`). `geometry` is the shared rate↔position/ladder SSOT;
the strategies (`hop_staging`, the TTFS prefix ramp) stay thin over it.

Only the leaf geometry is re-exported here: `models` imports it, and the
package init must not pull `spiking`/measurement modules into that path.
"""

from mimarsinan.tuning.orchestration.frontier.geometry import (
    frontier_ladder,
    frontier_position,
)

__all__ = ["frontier_ladder", "frontier_position"]
