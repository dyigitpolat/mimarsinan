"""[U-HOST2] A deployment's host work and whether a plane prices it: judged once, for both cost planes.

Host work is what the host executes per inference: its MACs (the partition census, ``host_macs``) and its ComputeOp
dispatches (``compute_op_count``). A plane prices it from the identified host's rows: the rate for the MACs
(``host_macs_per_s``), the dispatch time for the operators (``t_host_op_overhead``). Three states follow, and both
planes read them from ``host_work_state`` and name them with the same words:
- blocked: the host share is unknown (no partition census), so a host-inclusive number cannot be certified;
- missing: host MACs exist and the rate that prices them is not declared (the refusal, ``host_work_refusal``):
  without a host price, moving work host-side would look free (the subsume rig);
- unpriced: operators are dispatched without MACs and no dispatch time is declared; the headline names it beside.
The absolute plane judges it with the physics' declared rows; the measured plane, which prices no host row, judges it
with none declared, so a record holding host work refuses its host-inclusive total by the absolute plane's refusal.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

#: The rows that price host work: the rate (MACs) and the dispatch time (operators).
RATE, DISPATCH = "host_macs_per_s", "t_host_op_overhead"
HOST_SHARE_UNKNOWN = ("the host share is unknown (no partition census), so a host-inclusive number cannot be "
                      "certified")
DISPATCH_UNPRICED = f"{DISPATCH}(per-invocation dispatch)"


class HostWorkState(NamedTuple):
    """Why host work blocks a host-inclusive number (None: it does not), the rows it misses, the work left unpriced."""

    blocked: Optional[str] = None
    missing: Tuple[str, ...] = ()
    unpriced: Tuple[str, ...] = ()


def host_work_state(quantities: Any, declared: Callable[[str], bool]) -> HostWorkState:
    """The host work of ``quantities`` (``has``/``get``) against the host rows ``declared`` says are declared."""
    if not quantities.has("host_macs"):
        return HostWorkState(blocked=HOST_SHARE_UNKNOWN)
    if quantities.get("host_macs").value != 0:
        return HostWorkState() if declared(RATE) else HostWorkState(missing=(RATE,))
    ops = quantities.get("compute_op_count").value if quantities.has("compute_op_count") else 0
    if ops == 0 or declared(DISPATCH):
        return HostWorkState()
    return HostWorkState(unpriced=(DISPATCH_UNPRICED,))


class _RecordCounts:
    """A sealed record's host work as its quantities state it (``quantities.from_record``: the partition census's
    ``host_macs``, the schedule's ``compute_op_count``), read without the quantity builder."""

    def __init__(self, record: Any) -> None:
        partition = record.utilization.partition
        self._values = {"compute_op_count": int(record.schedule.compute_op_count)}
        if partition is not None:
            self._values["host_macs"] = int(partition.host_macs)

    def has(self, key: str) -> bool:
        return key in self._values

    def get(self, key: str) -> Any:
        return _Value(self._values[key])


class _Value(NamedTuple):
    value: int


def record_host_work(record: Any, declared: Callable[[str], bool] = lambda key: False) -> HostWorkState:
    """The host work a sealed record holds, judged against ``declared`` (none: the measured plane, which prices no
    host row); a record without a host ComputeOp or host MACs holds none."""
    counts = _RecordCounts(record)
    macs = counts.get("host_macs").value if counts.has("host_macs") else 0
    if counts.get("compute_op_count").value == 0 and macs == 0:
        return HostWorkState()
    return host_work_state(counts, declared)


def host_work_refusal(state: HostWorkState) -> str:
    """Why a host-inclusive headline cannot be certified, in the one wording both planes print."""
    if state.blocked is not None:
        return state.blocked
    return (
        "host work exists (host_macs > 0) and pricing it needs "
        f"{', '.join(state.missing)} from an identified host description (hosts/<name>.json, the run's host block; "
        "a measured wall is a diagnostic, never a price) — without a host price, moving work "
        "host-side would look free (the subsume rig)"
    )
