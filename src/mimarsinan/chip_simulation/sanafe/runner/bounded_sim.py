"""Wall-capped SANA-FE chip simulation (fresh-chip retry, loud second expiry)."""

from __future__ import annotations

from typing import Any, Tuple

from mimarsinan.chip_simulation.execution_bounds import (
    retry_once_on_timeout,
    run_bounded,
)


def simulate_chip_bounded(
    sanafe_module: Any,
    arch: Any,
    net: Any,
    timesteps: int,
    *,
    spike_trace: bool,
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
