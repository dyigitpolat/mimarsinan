"""Wall-capped SANA-FE chip simulation (fresh-chip retry, loud second expiry)."""

from __future__ import annotations

from typing import Any, Tuple

from mimarsinan.chip_simulation.execution_bounds import (
    retry_once_on_timeout,
    run_bounded,
)
from mimarsinan.chip_simulation.sanafe.runner.segment_release import CompactSpikeTrace

#: [U-SFR] ``spike_trace=SPIKE_TRACE_COMPACT``: the spike trace taken as text (``segment_release.CompactSpikeTrace``),
#: read back per timestep as the strings of SANA-FE's own address objects, at a few bytes per spike.
SPIKE_TRACE_COMPACT = "compact"


def simulate_chip_bounded(
    sanafe_module: Any,
    arch: Any,
    net: Any,
    timesteps: int,
    *,
    spike_trace: Any,
    potential_trace: bool,
    message_trace: bool,
    timeout_s: float,
) -> Tuple[Any, Any]:
    """Build a fresh ``SpikingChip``, load ``net``, and run ``sim`` under the wall cap.

    A hung native ``sim`` cannot be killed in-process: the watchdog abandons its
    thread, one retry runs on a fresh chip, and a second expiry raises loud.
    """

    def attempt(_attempt_index: int) -> Tuple[Any, Any]:
        chip = sanafe_module.SpikingChip(arch)
        chip.load(net)
        if spike_trace == SPIKE_TRACE_COMPACT:
            text = CompactSpikeTrace()  # a fresh one per attempt; no header, so one write per timestep
            results = run_bounded(
                lambda: chip.sim(timesteps, spike_trace=text, potential_trace=potential_trace,
                                 message_trace=message_trace, write_trace_headers=False),
                timeout_s=timeout_s,
                description=f"SANA-FE chip.sim({timesteps} timesteps)",
            )
            if results.get("spike_trace") is None:
                results["spike_trace"] = text.trace()
            return chip, results
        results = run_bounded(
            lambda: chip.sim(
                timesteps,
                spike_trace=spike_trace,
                potential_trace=potential_trace,
                message_trace=message_trace,
            ),
            timeout_s=timeout_s,
            description=f"SANA-FE chip.sim({timesteps} timesteps)",
        )
        return chip, results

    return retry_once_on_timeout(attempt, description="SANA-FE chip simulation")
