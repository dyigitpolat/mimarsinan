"""Frontier E1 — the hypervolume coverage ledger: genericity as a MEASURED fraction.

The program plan's review says genericity is *asserted, not measured*: coverage is
per-run, the hypervolume is unmodeled. This module makes it explicit and measured.

Three pieces:

* the HYPERVOLUME AXIS MODEL (:class:`HypervolumeAxis`, :data:`AXES`) — the typed
  product of deployment axes (``firing × sync × encoding_placement × quantization ×
  pruning × backend × mapping_strategy × S × vehicle × dataset × regime``), each
  classified ORTHOGONAL (covered MARGINALLY — vary one axis with the rest fixed) vs
  INTERACTING (tested JOINTLY). The classification is justified by cheap screening:
  ``encoding_placement`` offload≡subsume under signed-IF is one ORTHOGONAL result the
  checkpoint already found, so that axis COLLAPSES to a single representative value;
  ``firing × sync`` (the death-cascade law) and ``quantization × firing`` are
  INTERACTING — they must be tested jointly.

* the FULL-TUPLE :class:`HypervolumeCell` — extends the
  ``(firing × sync × backend)`` :class:`CertificationCell` (E6) to the full config
  tuple so each science-valid ledger row maps to exactly one hypervolume cell. Its
  ``cert_cell`` is the embedded (firing × sync × backend) sub-coordinate.

* :func:`coverage_report` — the GROUP BY over a ledger by cell-key + the
  ``deployment_validity`` tier, reporting per claimed cell a status in {VALID,
  VALID_FLAGGED, INVALID, UNTESTED}, the measured COVERAGE FRACTION (covered /
  claimed), the named UNTESTED frontier (claimed cells with no row), and the
  RESEARCH-GAP frontier (the union of ``research_gap_ops`` over VALID_FLAGGED cells
  — the future-conversion targets).

Pure data + a reader: it runs nothing, mutates no sim behavior; it reads ledger rows
(``runs/campaign/ledger.jsonl``) and tallies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.mapping.verification.onchip_fraction import (
    TIER_INVALID,
    TIER_VALID,
    TIER_VALID_FLAGGED,
)


__all__ = [
    "AxisKind",
    "HypervolumeAxis",
    "AXES",
    "collapse_orthogonal_axes",
    "active_axes",
    "HypervolumeCell",
    "CoverageStatus",
    "classify_validity_tier",
    "claimed_subproduct",
    "row_to_cell",
    "row_to_cells",
    "CoverageReport",
    "coverage_report",
]


class AxisKind(Enum):
    """How an axis is covered.

    ``ORTHOGONAL`` axes are covered MARGINALLY — vary one value with the rest of the
    tuple fixed (a cheap screen establishes the axis does not interact). ``INTERACTING``
    axes must be covered JOINTLY — their cross-product is the real surface (the
    death-cascade is a (firing × sync) joint result, quantization interacts with the
    firing law).
    """

    ORTHOGONAL = "orthogonal"
    INTERACTING = "interacting"


@dataclass(frozen=True)
class HypervolumeAxis:
    """One typed deployment axis of the hypervolume, with its coverage classification.

    ``name`` is the axis (a ``HypervolumeCell`` field); ``kind`` is ORTHOGONAL vs
    INTERACTING; ``interacts_with`` names the axes it is tested jointly with (empty
    for orthogonal); ``values`` is the (small) screened domain; ``collapsed`` marks an
    ORTHOGONAL axis whose values were cheap-screened EQUIVALENT (so the product drops
    it to one representative); ``representative`` is that value; ``justification`` is
    the cheap-screening reason the classification is sound.
    """

    name: str
    kind: AxisKind
    values: Tuple[str, ...]
    interacts_with: Tuple[str, ...] = ()
    collapsed: bool = False
    representative: Optional[str] = None
    justification: str = ""

    @classmethod
    def get(cls, name: str) -> "HypervolumeAxis":
        for axis in AXES:
            if axis.name == name:
                return axis
        raise KeyError(f"no hypervolume axis named {name!r}; known: {[a.name for a in AXES]}")


# The hypervolume axis model. ``values`` are the cheaply-screened domains (NOT the full
# combinatorial space — that is what a campaign fills); they anchor the claimed product
# and the untested frontier. The classification + justification are the load-bearing
# E1 content:
#
#   * firing × sync are INTERACTING — the death-cascade is a (firing × sync) joint law
#     (cascaded ttfs_cycle collapses where synchronized does not), so they are tested
#     jointly (the CertificationCell already keys on the pair).
#   * quantization × firing are INTERACTING — weight/activation quantization interacts
#     with the firing law (DFQ-for-LIF hurts; q=0.99 optimal for LIF; the cascade ramp
#     is quantization-sensitive), so quantization is screened per-firing.
#   * encoding_placement is ORTHOGONAL and COLLAPSED — the checkpoint's cheap screen
#     found offload≡subsume under signed integrate-and-fire (the segment-boundary
#     encode/decode is value-preserving either way), so it is dropped to one
#     representative ("subsume") and is NOT a coordinate of the cell key.
#   * pruning / mapping_strategy / backend are ORTHOGONAL (a pruned/packed/backend
#     variant is screened marginally against the dense/identity/nevresim reference);
#     they remain in the product (not yet proven collapsible) but are covered
#     marginally, not as a full cross-product.
#   * S / vehicle / dataset / regime are the campaign breadth axes — covered as the
#     claimed product demands (S and vehicle interact with the firing law's d_max(S)
#     budget, but are enumerated, not collapsed).
AXES: Tuple[HypervolumeAxis, ...] = (
    HypervolumeAxis(
        name="firing",
        kind=AxisKind.INTERACTING,
        values=("lif", "rate", "ttfs", "ttfs_quantized", "ttfs_cycle_based"),
        interacts_with=("sync", "quantization", "S"),
        justification=(
            "firing interacts with sync (the death-cascade is a firing×sync law) "
            "and with quantization and S (the d_max(S) firing-gain budget)"
        ),
    ),
    HypervolumeAxis(
        name="sync",
        kind=AxisKind.INTERACTING,
        values=("none", "cascaded", "synchronized"),
        interacts_with=("firing",),
        justification=(
            "the cascaded↔synchronized gap is a (firing×sync) joint result; "
            "tested jointly with firing (the CertificationCell keys on the pair)"
        ),
    ),
    HypervolumeAxis(
        name="encoding_placement",
        kind=AxisKind.ORTHOGONAL,
        values=("subsume", "offload"),
        collapsed=True,
        representative="subsume",
        justification=(
            "CHEAP-SCREEN RESULT (checkpoint): offload≡subsume under signed "
            "integrate-and-fire — the segment-boundary encode/decode is "
            "value-preserving either way, so the deployed numbers do not move with "
            "placement. Collapsed to a single representative; not a cell coordinate."
        ),
    ),
    HypervolumeAxis(
        name="quantization",
        kind=AxisKind.INTERACTING,
        values=("none", "wq", "aq", "wq_aq"),
        interacts_with=("firing",),
        justification=(
            "weight/activation quantization interacts with the firing law "
            "(DFQ-for-LIF hurts; the cascade ramp is quantization-sensitive); "
            "screened per-firing, not as a standalone axis"
        ),
    ),
    HypervolumeAxis(
        name="pruning",
        kind=AxisKind.ORTHOGONAL,
        values=("dense", "pruned"),
        justification=(
            "pruning is screened marginally against the dense reference (a pruned "
            "variant shares the firing×sync recipe cell); covered marginally"
        ),
    ),
    HypervolumeAxis(
        name="backend",
        kind=AxisKind.ORTHOGONAL,
        values=("nevresim", "sanafe", "hcm", "lava"),
        justification=(
            "backends are parity-locked to the reference (NF↔SCM↔SANA-FE bit-exact "
            "in the validated corner), so a backend is screened marginally"
        ),
    ),
    HypervolumeAxis(
        name="mapping_strategy",
        kind=AxisKind.ORTHOGONAL,
        values=("packed", "identity", "neuron_split", "coalesced"),
        justification=(
            "a mapping strategy (packed/neuron-split/coalesced) is screened "
            "marginally against the identity-mapping reference (the torch↔sim "
            "fidelity lock holds per-strategy); covered marginally"
        ),
    ),
    HypervolumeAxis(
        name="S",
        kind=AxisKind.INTERACTING,
        values=("4", "8", "16", "32"),
        interacts_with=("firing",),
        justification=(
            "S interacts with the firing law (d_max(S)≈0.56√S firing-gain budget); "
            "enumerated, not collapsed"
        ),
    ),
    HypervolumeAxis(
        name="vehicle",
        kind=AxisKind.ORTHOGONAL,
        values=("deep_mlp", "deep_cnn", "lenet5", "mlp_mixer_core", "vit_b"),
        justification=(
            "the model architecture is a breadth axis — genericity is the claim that "
            "the mechanisms transfer across vehicles; enumerated by the claimed product"
        ),
    ),
    HypervolumeAxis(
        name="dataset",
        kind=AxisKind.ORTHOGONAL,
        values=("mnist", "fmnist", "kmnist", "svhn", "cifar10"),
        justification=(
            "the dataset is a breadth axis — the synchronized off-MNIST gap is a "
            "training problem, not a deployment one; enumerated by the claimed product"
        ),
    ),
    HypervolumeAxis(
        name="regime",
        kind=AxisKind.ORTHOGONAL,
        values=("from_scratch", "pretrained"),
        justification=(
            "the training regime (from-scratch vs pretrained-bridge) is a breadth "
            "axis for the dual-regime certification; enumerated by the claimed product"
        ),
    ),
)


def collapse_orthogonal_axes(
    axes: Sequence[HypervolumeAxis],
) -> Tuple[HypervolumeAxis, ...]:
    """Drop every COLLAPSED axis — the active product the cell key is built over.

    A collapsed orthogonal axis (one whose values cheap-screened equivalent, e.g.
    ``encoding_placement`` offload≡subsume) contributes a single representative and is
    therefore NOT a coordinate of the hypervolume cell; this returns the axes that
    remain (the active product).
    """
    return tuple(a for a in axes if not a.collapsed)


def active_axes() -> Tuple[HypervolumeAxis, ...]:
    """The active (non-collapsed) hypervolume axes — the cell key's coordinates."""
    return collapse_orthogonal_axes(AXES)


# The cell-key coordinate order (collapsed axes excluded). The (firing × sync ×
# backend) prefix is the embedded CertificationCell; the rest extend it.
_CELL_AXES: Tuple[str, ...] = tuple(a.name for a in collapse_orthogonal_axes(AXES))


@dataclass(frozen=True)
class HypervolumeCell:
    """The FULL-TUPLE coverage cell — extends ``CertificationCell`` to the config tuple.

    The (firing × sync × backend) sub-coordinate is exactly an E6
    :class:`CertificationCell` (``cert_cell``); the remaining fields (vehicle,
    dataset, regime, quantization, pruning, mapping_strategy, S) extend it so each
    science-valid ledger row maps to exactly one hypervolume cell. The collapsed
    ``encoding_placement`` axis is NOT a field (offload≡subsume).
    """

    firing: str
    sync: Optional[str]
    backend: str
    vehicle: str
    dataset: str
    regime: str
    quantization: str
    pruning: str
    mapping_strategy: str
    s: str

    @property
    def cert_cell(self) -> CertificationCell:
        """The embedded (firing × sync × backend) E6 cell."""
        sync = None if self.sync in (None, "none", "") else self.sync
        return CertificationCell(firing=self.firing, sync=sync, backend=self.backend)

    @property
    def cell_key(self) -> str:
        """Canonical full-tuple key: ``<cert_key>|vehicle=…|dataset=…|…``.

        Prefixed by the canonical ``mode[/schedule]@backend`` E6 key so the embedded
        cert cell is the SAME named thing across the program, then the extending axes
        as ``axis=value`` segments in a stable order.
        """
        parts = [self.cert_cell.cell_key]
        for axis in _CELL_AXES:
            if axis in ("firing", "sync", "backend"):
                continue
            parts.append(f"{axis}={getattr(self, _FIELD_FOR_AXIS[axis])}")
        return "|".join(parts)

    @classmethod
    def from_key(cls, key: str) -> "HypervolumeCell":
        """Parse a canonical full-tuple key back into a cell."""
        cert_part, *rest = key.split("|")
        cert = CertificationCell.from_key(cert_part)
        kv = {}
        for seg in rest:
            if "=" not in seg:
                raise ValueError(f"malformed hypervolume cell segment {seg!r} in {key!r}")
            axis, value = seg.split("=", 1)
            kv[axis] = value
        return cls(
            firing=cert.firing,
            sync=cert.sync,
            backend=cert.backend,
            vehicle=kv["vehicle"],
            dataset=kv["dataset"],
            regime=kv["regime"],
            quantization=kv["quantization"],
            pruning=kv["pruning"],
            mapping_strategy=kv["mapping_strategy"],
            s=kv["S"],
        )


# axis-name → HypervolumeCell field name (``S`` is stored as ``s``).
_FIELD_FOR_AXIS: Dict[str, str] = {a: a for a in _CELL_AXES}
_FIELD_FOR_AXIS["S"] = "s"


class CoverageStatus(Enum):
    """The per-cell coverage status the GROUP BY assigns.

    VALID / VALID_FLAGGED / INVALID mirror the tiered ``ValidityVerdict`` from
    ``onchip_fraction`` (a covered cell's worst tier wins); UNTESTED is a claimed cell
    with NO row.
    """

    VALID = TIER_VALID
    VALID_FLAGGED = TIER_VALID_FLAGGED
    INVALID = TIER_INVALID
    UNTESTED = "UNTESTED"


# The worst-tier order (most → least conservative) for collapsing conflicting rows on
# one cell: INVALID dominates VALID_FLAGGED dominates VALID. A cell tested as both
# valid and flagged is reported FLAGGED (it owes the research gap); a cell tested
# valid and invalid is reported INVALID (the conservative read).
_TIER_SEVERITY: Dict[CoverageStatus, int] = {
    CoverageStatus.INVALID: 3,
    CoverageStatus.VALID_FLAGGED: 2,
    CoverageStatus.VALID: 1,
}


def classify_validity_tier(raw: Optional[str]) -> Optional[CoverageStatus]:
    """Normalize a ledger ``deployment_validity`` string to a canonical tier.

    The ledger's tier strings are free-form (``VALID_on_chip_majority``,
    ``VALID_FLAGGED_placement``, ``INVALID_host_majority``, ``VALID_clean_rc0_…``).
    Returns the canonical :class:`CoverageStatus` (VALID / VALID_FLAGGED / INVALID) or
    ``None`` for a row that carries NO validity verdict (a non-science / run-status
    row like ``FINALIZED_rc0`` or an empty/absent tier) — those do not name a science
    cell. Order matters: ``VALID_FLAGGED`` and ``INVALID`` are checked before the
    ``VALID`` prefix.
    """
    if not raw:
        return None
    text = str(raw).upper()
    if "INVALID" in text:
        return CoverageStatus.INVALID
    if "FLAGGED" in text:
        return CoverageStatus.VALID_FLAGGED
    if text.startswith("VALID"):
        return CoverageStatus.VALID
    return None


def _firing_of(row: Mapping[str, Any]) -> str:
    return str(row.get("spiking_mode") or row.get("mode") or "lif")


_VALID_SYNC = ("cascaded", "synchronized")


def _sync_of(row: Mapping[str, Any]) -> str:
    """The sync axis value: ``cascaded`` / ``synchronized`` / ``none``.

    Whitelists the two real sync schedules; any other ``schedule`` token (``all``,
    ``offload``, ``finalize_attempt`` and other non-sync run-status leakage) is
    normalized to ``none`` so it does not spawn a spurious sync cell.
    """
    schedule = row.get("schedule")
    if schedule in _VALID_SYNC:
        return str(schedule)
    return "none"


def _dataset_of(row: Mapping[str, Any]) -> str:
    ds = row.get("dataset")
    if not ds:
        return "unknown"
    return str(ds).lower().replace("fashionmnist", "fmnist").replace("fashion_mnist", "fmnist")


def _syncs_of(row: Mapping[str, Any]) -> List[str]:
    """The sync axis value(s) a row covers.

    A row with an explicit ``schedule`` covers that one sync. An ``arch_dataset`` row
    that reports BOTH ``cascaded_deployed_mean`` and ``synchronized_deployed_mean``
    covers BOTH sync cells (it is one row holding two schedules' results), so it
    expands to ``[cascaded, synchronized]``. Otherwise it covers the ``none`` sync
    cell (the firing-only family, or a row with no schedule signal at all).
    """
    explicit = _sync_of(row)
    if explicit in _VALID_SYNC:
        return [explicit]
    reported = [
        sync
        for sync, key in (
            ("cascaded", "cascaded_deployed_mean"),
            ("synchronized", "synchronized_deployed_mean"),
        )
        if row.get(key) is not None
    ]
    return reported or ["none"]


def _cell_with_sync(row: Mapping[str, Any], vehicle: str, sync: str) -> HypervolumeCell:
    return HypervolumeCell(
        firing=_firing_of(row),
        sync=sync,
        backend=str(row.get("backend") or "sanafe"),
        vehicle=str(vehicle),
        dataset=_dataset_of(row),
        regime=str(row.get("regime") or "from_scratch"),
        quantization=str(row.get("quantization") or "none"),
        pruning=str(row.get("pruning") or "dense"),
        mapping_strategy=str(row.get("mapping_strategy") or "packed"),
        s=str(row.get("S") if row.get("S") is not None else "any"),
    )


def row_to_cells(row: Mapping[str, Any]) -> List[HypervolumeCell]:
    """Map one science-valid ledger row to the hypervolume cell(s) it covers.

    Reads the row's deployment axes — ``model`` (vehicle), ``dataset``, ``schedule`` /
    the per-schedule deployed-mean fields (sync), ``spiking_mode``/``mode`` (firing),
    ``backend`` — and the extending axes where the row carries them (else the screened
    default). A dual-schedule ``arch_dataset`` row expands to both sync cells. Returns
    ``[]`` when the row carries no validity tier (not a science cell) or no model.
    """
    if classify_validity_tier(row.get("deployment_validity")) is None:
        return []
    vehicle = row.get("model") or row.get("model_type")
    if not vehicle:
        return []
    return [_cell_with_sync(row, vehicle, sync) for sync in _syncs_of(row)]


def row_to_cell(row: Mapping[str, Any]) -> Optional[HypervolumeCell]:
    """The first hypervolume cell a row covers (``None`` for a non-science row).

    Convenience over :func:`row_to_cells` for the common single-sync row; a
    dual-schedule row returns only its first cell (use :func:`row_to_cells` for both).
    """
    cells = row_to_cells(row)
    return cells[0] if cells else None


# Defaults the claimed-product builder fills for any axis the caller does not pin (so
# a partial claim resolves to ONE concrete representative per unpinned axis — the
# coverage fraction is over a concrete set of cells, not an ambiguous template).
_CLAIMED_DEFAULTS: Dict[str, Tuple[str, ...]] = {
    "firing": ("ttfs_cycle_based",),
    "sync": ("cascaded",),
    "backend": ("sanafe",),
    "vehicle": ("deep_cnn",),
    "dataset": ("mnist",),
    "regime": ("from_scratch",),
    "quantization": ("none",),
    "pruning": ("dense",),
    "mapping_strategy": ("packed",),
    "S": ("any",),
}


def claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """Enumerate a claimed sub-product of the hypervolume as concrete cells.

    Each keyword pins an axis to a list of values; unpinned axes take their single
    screened default. A COLLAPSED axis (``encoding_placement``) is IGNORED — pinning
    it to both values does not double the product (offload≡subsume), so the claimed
    set is invariant to it. Returns the cartesian product as :class:`HypervolumeCell`s.
    """
    chosen: Dict[str, Tuple[str, ...]] = {}
    for axis in _CELL_AXES:
        if axis in axis_values:
            chosen[axis] = tuple(axis_values[axis])
        else:
            chosen[axis] = _CLAIMED_DEFAULTS[axis]

    ordered_axes = list(_CELL_AXES)
    cells: List[HypervolumeCell] = []
    for combo in itertools.product(*(chosen[a] for a in ordered_axes)):
        kv = dict(zip(ordered_axes, combo))
        sync = None if kv["sync"] in (None, "none", "") else kv["sync"]
        cells.append(
            HypervolumeCell(
                firing=kv["firing"],
                sync=sync if sync is not None else "none",
                backend=kv["backend"],
                vehicle=kv["vehicle"],
                dataset=kv["dataset"],
                regime=kv["regime"],
                quantization=kv["quantization"],
                pruning=kv["pruning"],
                mapping_strategy=kv["mapping_strategy"],
                s=kv["S"],
            )
        )
    # Dedupe (a collapsed axis pinned to multiple values yields identical cells).
    seen: Dict[str, HypervolumeCell] = {}
    for cell in cells:
        seen.setdefault(cell.cell_key, cell)
    return list(seen.values())


@dataclass(frozen=True)
class CoverageReport:
    """The measured coverage of a claimed sub-product over a ledger.

    ``cell_status`` is the GROUP BY result: each COVERED cell → its worst tier.
    ``tier_counts`` tallies the covered cells by tier. When a ``claimed_subproduct``
    is supplied, ``claimed_cells`` is it, ``covered_claimed_count`` is how many were
    tested, ``coverage_fraction`` = covered / claimed, ``untested_frontier`` is the
    claimed cells with no row, and ``status_for`` answers a claimed cell's status
    (UNTESTED when uncovered). ``research_gap_frontier`` is the sorted, deduped union
    of ``research_gap_ops`` over the VALID_FLAGGED cells — the future-conversion
    targets (host ops with NO on-chip SNN mapping yet). ``placement_fixable_frontier``
    is the parallel union of ``placement_fixable_ops`` — supported encoders host-placed
    under ``subsume`` that an ``offload`` flip would map on-chip (un-flagging the cell);
    those are NOT research gaps, so they are reported separately.
    """

    cell_status: Dict[str, CoverageStatus]
    tier_counts: Dict[CoverageStatus, int]
    research_gap_frontier: List[str]
    placement_fixable_frontier: List[str] = field(default_factory=list)
    claimed_cells: Tuple[HypervolumeCell, ...] = ()
    _covered_keys: frozenset = field(default_factory=frozenset)

    @property
    def covered_cell_count(self) -> int:
        return len(self.cell_status)

    @property
    def claimed_cell_count(self) -> int:
        return len(self.claimed_cells)

    @property
    def covered_claimed_count(self) -> int:
        return sum(1 for c in self.claimed_cells if c.cell_key in self._covered_keys)

    @property
    def coverage_fraction(self) -> float:
        if not self.claimed_cells:
            return 0.0
        return self.covered_claimed_count / self.claimed_cell_count

    @property
    def untested_frontier(self) -> List[HypervolumeCell]:
        return [c for c in self.claimed_cells if c.cell_key not in self._covered_keys]

    def status_for(self, cell: HypervolumeCell) -> CoverageStatus:
        return self.cell_status.get(cell.cell_key, CoverageStatus.UNTESTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "covered_cell_count": self.covered_cell_count,
            "claimed_cell_count": self.claimed_cell_count,
            "covered_claimed_count": self.covered_claimed_count,
            "coverage_fraction": self.coverage_fraction,
            "tier_counts": {s.value: n for s, n in self.tier_counts.items()},
            "cell_status": {k: v.value for k, v in sorted(self.cell_status.items())},
            "untested_frontier": [c.cell_key for c in self.untested_frontier],
            "research_gap_frontier": list(self.research_gap_frontier),
            "placement_fixable_frontier": list(self.placement_fixable_frontier),
        }


# The tier-suffix back-compat marker the live ledger uses: a ``..._placement`` flag is
# a placement fix (offloadable encoder), NOT a research gap. Any other flag category is
# a research gap (an unsupported host op with no on-chip SNN mapping yet).
_PLACEMENT_FLAG_MARKER = "PLACEMENT"


def _mine_flagged_ops(row: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """``(research_gap_ops, placement_fixable_ops)`` named by one VALID_FLAGGED row.

    Primary source is the structured ``research_gap_ops`` / ``placement_fixable_ops``
    fields ``onchip_fraction.classify_validity`` emits. For a live ledger row that
    predates those fields (the flag is only in the ``deployment_validity`` tier
    suffix, e.g. ``VALID_FLAGGED_placement``), the category is derived from the suffix:
    a ``_placement`` flag yields one placement-fixable encoder, any other flag yields
    one research-gap op — so the frontier is never silently empty on real data.
    """
    structured_gaps = [str(op) for op in (row.get("research_gap_ops") or ())]
    structured_placement = [str(op) for op in (row.get("placement_fixable_ops") or ())]
    if structured_gaps or structured_placement:
        return structured_gaps, structured_placement

    text = str(row.get("deployment_validity") or "").upper()
    if _PLACEMENT_FLAG_MARKER in text:
        return [], ["encoding_layer(placement)"]
    return ["unsupported_host_op"], []


def coverage_report(
    rows: Iterable[Mapping[str, Any]],
    claimed_subproduct: Optional[Sequence[HypervolumeCell]] = None,
) -> CoverageReport:
    """GROUP BY the ledger by hypervolume cell-key + validity tier → coverage.

    Each science-valid row (one carrying a ``deployment_validity`` tier) is mapped to
    its cell and tier; a cell's status is the WORST tier of its rows (INVALID >
    VALID_FLAGGED > VALID, the conservative read). ``research_gap_frontier`` unions the
    ``research_gap_ops`` of the VALID_FLAGGED cells (the future-conversion targets);
    ``placement_fixable_frontier`` unions their ``placement_fixable_ops`` (offloadable
    encoders — NOT research gaps). When ``claimed_subproduct`` is given, the report
    measures coverage against it: the fraction tested, the named UNTESTED frontier, and
    each claimed cell's status. Non-science rows (no tier) and rows with no model are
    skipped.
    """
    materialized = list(rows)

    # Pass 1: resolve each cell's FINAL tier (the worst tier of its rows).
    cell_status: Dict[str, CoverageStatus] = {}
    row_cells: List[Tuple[Mapping[str, Any], CoverageStatus, List[HypervolumeCell]]] = []
    for row in materialized:
        tier = classify_validity_tier(row.get("deployment_validity"))
        if tier is None:
            continue
        cells = row_to_cells(row)
        if not cells:
            continue
        row_cells.append((row, tier, cells))
        for cell in cells:
            key = cell.cell_key
            prior = cell_status.get(key)
            if prior is None or _TIER_SEVERITY[tier] > _TIER_SEVERITY[prior]:
                cell_status[key] = tier

    # Pass 2: mine the flag ops only from rows whose cell's FINAL tier is FLAGGED, so
    # the frontiers are unions over the VALID_FLAGGED cells (a cell demoted to INVALID
    # by a conflicting row is no longer a flagged-cell research target).
    research_gaps: set = set()
    placement_fixable: set = set()
    for row, tier, cells in row_cells:
        if tier is not CoverageStatus.VALID_FLAGGED:
            continue
        if not any(
            cell_status[cell.cell_key] is CoverageStatus.VALID_FLAGGED for cell in cells
        ):
            continue
        gaps, placement = _mine_flagged_ops(row)
        research_gaps.update(gaps)
        placement_fixable.update(placement)

    tier_counts: Dict[CoverageStatus, int] = {
        CoverageStatus.VALID: 0,
        CoverageStatus.VALID_FLAGGED: 0,
        CoverageStatus.INVALID: 0,
    }
    for status in cell_status.values():
        tier_counts[status] += 1

    covered_keys = frozenset(cell_status)
    claimed = tuple(claimed_subproduct or ())

    return CoverageReport(
        cell_status=cell_status,
        tier_counts=tier_counts,
        research_gap_frontier=sorted(research_gaps),
        placement_fixable_frontier=sorted(placement_fixable),
        claimed_cells=claimed,
        _covered_keys=covered_keys,
    )
