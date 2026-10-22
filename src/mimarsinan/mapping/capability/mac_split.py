"""What a level row may state about MACs: both counters, the cores' MAC floor, and a named reason per withheld field.

Every MAC number comes from the MAC SSOT (``count_host_ops``, ``estimate_onchip_fraction(metric='macs')``) or from
the IR's own cells. A number the counters cannot back is absent with a named reason, never wrong (plan §11.1).
Three generic tests decide it:

- the estimator's own on-chip MACs fall below the cores' MAC floor (core cells fed by a value): its total
  undercounts (today every on-chip convolution is counted at one output position), so the total, the on-chip MACs
  and every share are withheld;
- the two host counters disagree (a multi-input op only the forward estimator counts): host MACs are withheld;
- a MAC-carrying op the MAC SSOT counts 0 (no rule for it yet): the MAC sum of the side that runs it is withheld.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.capability.census import LevelCensus
from mimarsinan.mapping.capability.requirements import requirement_of

#: The estimator total is below the cores' MAC floor (the reviewed-delta cause: one position per on-chip conv).
ESTIMATOR_BELOW_CORE_FLOOR = "estimator_conv_one_position"
MULTI_INPUT_REASON = "multi-input op counted by the forward estimator only until G1's input_shapes rule"
#: Host counters that disagree on an IR with no multi-input op (the acceptance checker fails it).
COUNTERS_DISAGREE = "mac_counters_disagree"
UNCOUNTED_PRODUCT = "uncounted_product"


@dataclass(frozen=True)
class MacCounters:
    """Host MACs by the IR counter (``count_host_ops``) and by the forward estimator, and the estimator's total."""

    ir_host_macs: int
    estimator_host_macs: int
    estimator_total_macs: int
    has_multi_input_op: bool

    @property
    def agree(self) -> bool:
        return self.ir_host_macs == self.estimator_host_macs

    @property
    def reason(self) -> Optional[str]:
        return MULTI_INPUT_REASON if self.has_multi_input_op else None

    @property
    def estimator_onchip_macs(self) -> int:
        return self.estimator_total_macs - self.estimator_host_macs

    def total_withheld(self, core_mac_floor: int) -> Optional[str]:
        """The estimator counts fewer on-chip MACs than the IR's cores carry, so its total undercounts."""
        return ESTIMATOR_BELOW_CORE_FLOOR if self.estimator_onchip_macs < int(core_mac_floor) else None

    @property
    def host_withheld(self) -> Optional[str]:
        if self.agree:
            return None
        return MULTI_INPUT_REASON if self.has_multi_input_op else COUNTERS_DISAGREE


@dataclass(frozen=True)
class StatedBoundary:
    """A stated host boundary (§16 decision 4) with its host MACs, or the reason they are withheld."""

    boundary: str
    host_macs: Optional[int]
    withheld: Optional[str] = None


@dataclass(frozen=True)
class MacStatement:
    """The MAC numbers one level row states (None where withheld) and the named reason for each withheld field."""

    host_macs: Optional[int]
    onchip_macs: Optional[int]
    total_macs: Optional[int]
    synaptic_events: Optional[int]
    stated_boundaries: Tuple[StatedBoundary, ...] = ()
    withheld: Mapping[str, str] = field(default_factory=dict)
    total_withheld: Optional[str] = None
    #: ``count_host_ops`` per host kind, leaving out a kind that holds an uncounted product.
    host_macs_by_kind: Mapping[str, int] = field(default_factory=dict)


def _uncounted(products: Mapping[str, Tuple[str, ...]], kinds: Optional[Sequence[str]] = None) -> Optional[str]:
    types = sorted({t for kind, found in products.items() if kinds is None or kind in kinds for t in found})
    return f"{UNCOUNTED_PRODUCT}: {', '.join(types)}" if types else None


def joined_reasons(*reasons: Optional[str]) -> Optional[str]:
    """The named reasons in order, joined by "; " (None when none is named). A field derived from withheld inputs
    (a share of withheld host MACs over a withheld total) names the reason of every one of them, numerator first."""
    named = [r for r in reasons if r]
    return "; ".join(named) if named else None


def _stated_boundaries(row: LevelCensus) -> Tuple[StatedBoundary, ...]:
    stated = []
    for boundary, macs in row.stated_boundaries.items():
        kinds = [k for k in row.uncounted_host_products if requirement_of(k).host_boundary == boundary]
        reason = _uncounted(row.uncounted_host_products, kinds)
        stated.append(StatedBoundary(boundary, None if reason else int(macs), reason))
    return tuple(stated)


def mac_statement(row: LevelCensus, counters: Optional[MacCounters]) -> MacStatement:
    """The row's MAC fields: stated when the counters back them, else absent with the reason by field name."""
    withheld: Dict[str, str] = {}
    chip_gap = _uncounted(row.uncounted_chip_products)
    if chip_gap:
        withheld["synaptic_events"] = chip_gap
    synaptic = None if chip_gap else row.cells_committed + row.onchip_op_macs
    by_kind = {k: int(v) for k, v in row.host_macs_by_kind.items() if k not in row.uncounted_host_products}
    if len(by_kind) < len(row.host_macs_by_kind):
        withheld["host_macs_by_kind"] = str(_uncounted(row.uncounted_host_products))
    if counters is None:
        return MacStatement(None, None, None, synaptic, withheld=withheld, host_macs_by_kind=by_kind)
    host_reason = joined_reasons(counters.host_withheld, _uncounted(row.uncounted_host_products))
    total_reason = counters.total_withheld(row.core_mac_floor)
    onchip_reason = joined_reasons(host_reason, total_reason)
    for name, reason in (("host_macs", host_reason), ("total_macs", total_reason), ("onchip_macs", onchip_reason)):
        if reason:
            withheld[name] = reason
    host = None if host_reason else row.host_macs
    total = None if total_reason else counters.estimator_total_macs
    return MacStatement(
        host_macs=host, onchip_macs=None if host is None or total is None else total - host, total_macs=total,
        synaptic_events=synaptic, stated_boundaries=_stated_boundaries(row), withheld=withheld,
        total_withheld=total_reason, host_macs_by_kind=by_kind,
    )
