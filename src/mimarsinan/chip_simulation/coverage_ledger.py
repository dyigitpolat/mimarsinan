"""Hypervolume coverage ledger: genericity as a measured covered/claimed fraction over ledger rows."""

from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.hypervolume_axis_encoder import (
    cell_coordinates_from_row,
    syncs_from_row,
)
from mimarsinan.mapping.verification.onchip_fraction import (
    TIER_INVALID,
    TIER_VALID,
    TIER_VALID_FLAGGED,
)


__all__ = [
    "AxisKind",
    "ScreeningStatus",
    "AttributionFidelity",
    "HypervolumeAxis",
    "AXES",
    "AXIS_WILDCARD",
    "collapse_orthogonal_axes",
    "collapsed_axis_representatives",
    "interacting_axes",
    "active_axes",
    "HypervolumeCell",
    "CoverageStatus",
    "FlagMetadata",
    "PLACEMENT_FIXABLE_DEFAULT_OWNER",
    "PLACEMENT_FIXABLE_FIX_PATH",
    "KNOWN_CRACKED_REGIONS",
    "classify_validity_tier",
    "claimed_subproduct",
    "honest_claimed_subproduct",
    "cell_covers",
    "row_to_cell",
    "row_to_cells",
    "CoverageReport",
    "coverage_report",
]


AXIS_WILDCARD = "any"


class AxisKind(Enum):
    """How an axis is covered: ORTHOGONAL (marginally) vs INTERACTING (jointly)."""

    ORTHOGONAL = "orthogonal"
    INTERACTING = "interacting"


class ScreeningStatus(Enum):
    """The classification gating whether an axis may collapse from the coverage denominator.

    ``SCREENED_COLLAPSED`` (proven equivalent, requires a ``screening_artifact``) collapses
    to one representative; ``ENUMERATED_INTERACTING`` and ``ASSERTED_UNSCREENED`` both stay enumerated.
    """

    SCREENED_COLLAPSED = "screened_collapsed"
    ENUMERATED_INTERACTING = "enumerated_interacting"
    ASSERTED_UNSCREENED = "asserted_unscreened"


class AttributionFidelity(Enum):
    """How trustworthy a covered region's per-neuron attribution is.

    ``ATTRIBUTION`` — bit-exact per-neuron reassembly; ``VALUE_DOMAIN_ONLY`` — only the
    deployed-accuracy value domain is bit-exact, the attribution reassembly is not gated in production.
    """

    ATTRIBUTION = "attribution"
    VALUE_DOMAIN_ONLY = "value_domain_only"


@dataclass(frozen=True)
class HypervolumeAxis:
    """One typed deployment axis of the hypervolume, with its coverage classification.

    ``screening_status`` gates collapse and the denominator; ``screening_artifact`` is
    mandatory for SCREENED_COLLAPSED. ``collapsed`` is DERIVED from the status alone.
    """

    name: str
    kind: AxisKind
    values: Tuple[str, ...]
    screening_status: ScreeningStatus = ScreeningStatus.ENUMERATED_INTERACTING
    interacts_with: Tuple[str, ...] = ()
    representative: Optional[str] = None
    screening_artifact: str = ""
    justification: str = ""

    def __post_init__(self) -> None:
        if self.screening_status is ScreeningStatus.SCREENED_COLLAPSED and not (
            self.screening_artifact and self.screening_artifact.strip()
        ):
            raise ValueError(
                f"axis {self.name!r} is SCREENED_COLLAPSED but carries no "
                f"screening_artifact — a collapse REQUIRES a linked artifact "
                f"(doc/test/result ref); collapse-on-a-hunch is forbidden"
            )

    @property
    def collapsed(self) -> bool:
        """DERIVED: an axis collapses iff a screen PROVED its values equivalent."""
        return self.screening_status is ScreeningStatus.SCREENED_COLLAPSED

    @classmethod
    def get(cls, name: str) -> "HypervolumeAxis":
        for axis in AXES:
            if axis.name == name:
                return axis
        raise KeyError(f"no hypervolume axis named {name!r}; known: {[a.name for a in AXES]}")


AXES: Tuple[HypervolumeAxis, ...] = (
    HypervolumeAxis(
        name="firing",
        kind=AxisKind.INTERACTING,
        values=("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
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
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
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
        screening_status=ScreeningStatus.SCREENED_COLLAPSED,
        representative="subsume",
        screening_artifact=(
            "docs/research/PROGRAM_CHECKPOINT.md + E3_SCALE_PROBE.md — offload==subsume "
            "to ~1e-6 under signed integrate-and-fire (the segment-boundary encode/decode "
            "is value-preserving either way). FIDELITY-ONLY: not collapsed for cost/"
            "utilization (see PROGRAM_PLAN_v2.md §E5/cost caveat)."
        ),
        justification=(
            "CHEAP-SCREEN RESULT (checkpoint): offload≡subsume under signed "
            "integrate-and-fire — the deployed numbers do not move with placement. "
            "Collapsed to a single representative; not a cell coordinate."
        ),
    ),
    HypervolumeAxis(
        name="quantization",
        kind=AxisKind.INTERACTING,
        values=("none", "wq", "aq", "wq_aq"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
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
        screening_status=ScreeningStatus.ASSERTED_UNSCREENED,
        justification=(
            "NO SCREEN YET — pruning is ASSERTED equivalent to dense but never "
            "cheap-screened; counted interacting (enumerated) until a P3 screen earns "
            "the collapse"
        ),
    ),
    HypervolumeAxis(
        name="backend",
        kind=AxisKind.ORTHOGONAL,
        values=("nevresim", "sanafe", "hcm", "lava"),
        screening_status=ScreeningStatus.SCREENED_COLLAPSED,
        representative="sanafe",
        screening_artifact=(
            "docs/research/findings/backend_cross_sim_screen.md (+ "
            "backend_cross_sim_screen.json) — a LIVE cross_sim_parity screen records "
            "nevresim/HCM/SCM (Python reference paths) AGREE to max_abs_diff=0 on "
            "representative cells spanning lif + ttfs_cycle_based × identity + "
            "neuron_split; the existing multi-backend parity LOCKS "
            "(tests/integration/test_scm_hcm_sim_parity.py, "
            "test_nf_hcm_per_node_spike_parity_mmixcore.py, "
            "tests/unit/pipelining/pipeline_steps/test_nf_scm_parity_gate.py, and the "
            "env-gated nevresim test_execute_simulator.py / SANA-FE "
            "test_sanafe_hcm_parity.py / Lava test_loihi_hcm_spike_parity.py) establish "
            "nevresim/sanafe/lava≡HCM in the validated corner. lava is INAPPLICABLE for "
            "TTFS (LIF-only capability gap). FIDELITY-ONLY (deployed value): faithful "
            "sims of the SAME contract must agree (a disagreement is a BUG, not an "
            "interaction). CAPABILITY (which backend×mode runs) and COST/UTILIZATION "
            "(per-backend energy) are NOT collapsed — frontiers, like the "
            "encoding_placement precedent."
        ),
        justification=(
            "FAITHFULNESS axis: backends are different SIMULATORS of the same "
            "deployment contract — they collapse on a measured fidelity/parity "
            "artifact (faithful sims agree on the deployed value), NOT a semantic "
            "screen. Collapsed to one representative for fidelity; capability + cost "
            "stay frontiers."
        ),
    ),
    HypervolumeAxis(
        name="mapping_strategy",
        kind=AxisKind.ORTHOGONAL,
        values=("packed", "identity", "neuron_split", "coalesced"),
        screening_status=ScreeningStatus.SCREENED_COLLAPSED,
        representative="packed",
        screening_artifact=(
            "docs/research/findings/mapping_strategy_fidelity_screen.md — "
            "tests/integration/test_torch_sim_fidelity.py PROVES torch-NF == deployed "
            "HCM sim BIT-EXACT (float64 atol=0; LIF per-neuron k==k) for "
            "identity / neuron_split / axon_fuse across every bit-exact mode × model "
            "(single-core, multi-core, sync-point). Equivalent packings of the SAME "
            "contract compute the SAME deployed value (a mismatch is a packing BUG). "
            "FIDELITY-ONLY (deployed value): coalescing carries the GAP-1 caveat — "
            "value-domain bit-exact + spike-conserved, but per-neuron ATTRIBUTION under "
            "coalescing+output-tiling is recorded VALUE_DOMAIN_ONLY: Wave-2 C3 made the "
            "fidelity-HARNESS reassembler bit-exact (joint output_slice+ir_id keying), "
            "yet the production NF↔SCM gate stays identity-mapping-only so the fragment "
            "path is unexercised in deployment. COST/UTILIZATION (cores, axon budget per "
            "strategy) is NOT collapsed — frontier."
        ),
        justification=(
            "FAITHFULNESS axis: mapping strategies are different PACKINGS of the same "
            "deployment contract — they collapse on a measured bit-exact fidelity "
            "artifact (equivalent packings agree on the deployed value), NOT a semantic "
            "screen. Collapsed to one representative for fidelity; per-neuron "
            "attribution for coalescing stays VALUE_DOMAIN_ONLY and cost stays a "
            "frontier."
        ),
    ),
    HypervolumeAxis(
        name="S",
        kind=AxisKind.INTERACTING,
        values=("4", "8", "16", "32"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
        interacts_with=("firing",),
        justification=(
            "S interacts with the firing law (d_max(S)≈0.56√S firing-gain budget); "
            "enumerated, not collapsed"
        ),
    ),
    HypervolumeAxis(
        name="depth",
        kind=AxisKind.INTERACTING,
        values=("4", "6", "8", "12", "16"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
        interacts_with=("firing", "S"),
        justification=(
            "depth interacts with the firing law (the death-cascade is a depth-driven "
            "collapse: deep_mlp's cascaded gap widens with depth, d4→d8); a depth's "
            "validity tier is depth-specific, so depth is a cell COORDINATE — a shallow "
            "INVALID and a deeper VALID_FLAGGED at the same recipe must NOT collapse to "
            "one cell (and mutually demote)"
        ),
    ),
    HypervolumeAxis(
        name="vehicle",
        kind=AxisKind.ORTHOGONAL,
        values=("deep_mlp", "deep_cnn", "lenet5", "mlp_mixer_core", "vit_b"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
        interacts_with=("depth", "dataset"),
        screening_artifact=(
            "docs/research/findings/WS3_depth_firing_gain.md + WS1_WS6_breadth_rigor.md — "
            "the dual-axis depth×dataset law with an ARCHITECTURE-DEPENDENT onset PROVES "
            "the vehicle interacts (the synchronized off-MNIST gap is architecture-"
            "dependent); enumerated, never collapsed"
        ),
        justification=(
            "the model architecture PROVABLY interacts (architecture-dependent "
            "death-cascade onset); enumerated by the claimed product, not collapsed"
        ),
    ),
    HypervolumeAxis(
        name="dataset",
        kind=AxisKind.ORTHOGONAL,
        values=("mnist", "fmnist", "kmnist", "svhn", "cifar10"),
        screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
        interacts_with=("depth", "vehicle"),
        screening_artifact=(
            "docs/research/findings/WS3_depth_firing_gain.md + PROGRAM_CHECKPOINT_v2.md — "
            "the dual-axis depth×dataset law (deep_cnn KMNIST d4→d10 gap shrink; the "
            "dataset-dominant death-cascade) PROVES the dataset interacts with depth; "
            "enumerated, never collapsed"
        ),
        justification=(
            "the dataset PROVABLY interacts with depth (the dual-axis depth×dataset "
            "law); enumerated by the claimed product, not collapsed"
        ),
    ),
    HypervolumeAxis(
        name="regime",
        kind=AxisKind.ORTHOGONAL,
        values=("from_scratch", "pretrained"),
        screening_status=ScreeningStatus.ASSERTED_UNSCREENED,
        justification=(
            "NO SCREEN YET — the from-scratch↔pretrained-bridge regimes are ASSERTED "
            "to share the recipe but never cross-screened; counted interacting "
            "(enumerated) until a P3 dual-regime screen"
        ),
    ),
)


KNOWN_CRACKED_REGIONS: Tuple[str, ...] = (
    (
        "GAP-1: coalescing+output-tiling per-neuron attribution at VGG scale — "
        "C3 fixed the fidelity-harness reassembler keying (joint (output_slice, ir_id) "
        "is bit-exact in tests/integration/_split_reassembly.py); the PRODUCTION "
        "NF↔SCM gate stays identity-mapping-only, so the fragment path is NOT exercised "
        "in deployment"
    ),
    "residual Tier-1 merge (fused-mapping reassembly)",
)


def collapse_orthogonal_axes(
    axes: Sequence[HypervolumeAxis],
) -> Tuple[HypervolumeAxis, ...]:
    """Drop every SCREENED_COLLAPSED axis, returning the active product the cell key is built over."""
    return tuple(
        a
        for a in axes
        if a.screening_status is not ScreeningStatus.SCREENED_COLLAPSED
    )


def collapsed_axis_representatives(
    axes: Sequence[HypervolumeAxis] = AXES,
) -> Dict[str, str]:
    """The SSOT ``{collapsed axis name → representative}`` map (falling back to the first screened value)."""
    reps: Dict[str, str] = {}
    for axis in axes:
        if axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED:
            reps[axis.name] = axis.representative or (
                axis.values[0] if axis.values else ""
            )
    return reps


def interacting_axes(
    axes: Sequence[HypervolumeAxis] = AXES,
) -> Tuple[HypervolumeAxis, ...]:
    """The axes counted INTERACTING (enumerated) in the honest denominator — the non-collapsed axes."""
    return collapse_orthogonal_axes(axes)


def active_axes() -> Tuple[HypervolumeAxis, ...]:
    """The active (non-collapsed) hypervolume axes — the cell key's coordinates."""
    return collapse_orthogonal_axes(AXES)


_CELL_AXES: Tuple[str, ...] = tuple(a.name for a in collapse_orthogonal_axes(AXES))

_COLLAPSED_REPRESENTATIVES: Dict[str, str] = collapsed_axis_representatives(AXES)


@dataclass(frozen=True)
class HypervolumeCell:
    """The FULL-TUPLE coverage cell — extends the (firing × sync × backend) ``CertificationCell`` to the config tuple.

    ``depth`` is a real coordinate so a shallow INVALID and a deeper VALID_FLAGGED at the
    same recipe do not collapse; the collapsed ``encoding_placement`` axis is NOT a field.
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
    depth: str

    @property
    def cert_cell(self) -> CertificationCell:
        """The embedded (firing × sync × backend) E6 cell, folding a collapsed backend to its representative."""
        sync = None if self.sync in (None, "none", "") else self.sync
        backend = _COLLAPSED_REPRESENTATIVES.get("backend", self.backend)
        return CertificationCell(firing=self.firing, sync=sync, backend=backend)

    @property
    def cell_key(self) -> str:
        """Canonical full-tuple key: the ``mode[/schedule]@backend`` cert prefix + ``axis=value`` segments."""
        parts = [self.cert_cell.cell_key]
        for axis in _CELL_AXES:
            if axis in ("firing", "sync", "backend"):
                continue
            parts.append(f"{axis}={getattr(self, _FIELD_FOR_AXIS[axis])}")
        return "|".join(parts)

    @classmethod
    def from_key(cls, key: str) -> "HypervolumeCell":
        """Parse a canonical full-tuple key back into a cell (dropped collapsed axes default to their representative)."""
        cert_part, *rest = key.split("|")
        cert = CertificationCell.from_key(cert_part)
        kv = {}
        for seg in rest:
            if "=" not in seg:
                raise ValueError(f"malformed hypervolume cell segment {seg!r} in {key!r}")
            axis, value = seg.split("=", 1)
            kv[axis] = value

        def _ext(axis: str) -> str:
            if axis in kv:
                return kv[axis]
            rep = _COLLAPSED_REPRESENTATIVES.get(axis)
            if rep is not None:
                return rep
            raise KeyError(axis)

        return cls(
            firing=cert.firing,
            sync=cert.sync,
            backend=cert.backend,
            vehicle=_ext("vehicle"),
            dataset=_ext("dataset"),
            regime=_ext("regime"),
            quantization=_ext("quantization"),
            pruning=_ext("pruning"),
            mapping_strategy=_ext("mapping_strategy"),
            s=_ext("S"),
            depth=_ext("depth"),
        )


_FIELD_FOR_AXIS: Dict[str, str] = {a: a for a in _CELL_AXES}
_FIELD_FOR_AXIS["S"] = "s"

_MATCH_FIELDS: Tuple[str, ...] = tuple(_FIELD_FOR_AXIS[a] for a in _CELL_AXES)


def _coord(cell: "HypervolumeCell", field_name: str) -> str:
    """A cell's coordinate value for a match field, with ``sync=None`` read as ``none``."""
    value = getattr(cell, field_name)
    if field_name == "sync" and value in (None, ""):
        return "none"
    return str(value)


def cell_covers(claimed: "HypervolumeCell", covered: "HypervolumeCell") -> bool:
    """True iff a COVERED cell satisfies a CLAIMED cell — a claimed :data:`AXIS_WILDCARD` matches any covered value, else exact."""
    for field_name in _MATCH_FIELDS:
        claim_value = _coord(claimed, field_name)
        if claim_value == AXIS_WILDCARD:
            continue
        if claim_value != _coord(covered, field_name):
            return False
    return True


class CoverageStatus(Enum):
    """The per-cell coverage status: VALID / VALID_FLAGGED / INVALID (worst tier wins) or UNTESTED (no row)."""

    VALID = TIER_VALID
    VALID_FLAGGED = TIER_VALID_FLAGGED
    INVALID = TIER_INVALID
    UNTESTED = "UNTESTED"


_TIER_SEVERITY: Dict[CoverageStatus, int] = {
    CoverageStatus.INVALID: 3,
    CoverageStatus.VALID_FLAGGED: 2,
    CoverageStatus.VALID: 1,
}


def classify_validity_tier(raw: Optional[str]) -> Optional[CoverageStatus]:
    """Normalize a free-form ledger ``deployment_validity`` string to a canonical tier (``None`` for a non-science row).

    Order matters: ``VALID_FLAGGED`` and ``INVALID`` are checked before the ``VALID`` prefix.
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


def _cell_with_sync(row: Mapping[str, Any], vehicle: str, sync: str) -> HypervolumeCell:
    coords = cell_coordinates_from_row(row, sync=sync, axis_wildcard=AXIS_WILDCARD)
    data = coords.as_cell_kwargs()
    data["vehicle"] = str(vehicle)
    return HypervolumeCell(**data)


def row_to_cells(row: Mapping[str, Any]) -> List[HypervolumeCell]:
    """Map one science-valid ledger row to the hypervolume cell(s) it covers (a dual-schedule row → both sync cells).

    Returns ``[]`` when the row carries no validity tier or no model.
    """
    if classify_validity_tier(row.get("deployment_validity")) is None:
        return []
    vehicle = row.get("model") or row.get("model_type")
    if not vehicle:
        return []
    return [_cell_with_sync(row, vehicle, sync) for sync in syncs_from_row(row)]


def row_to_cell(row: Mapping[str, Any]) -> Optional[HypervolumeCell]:
    """The first hypervolume cell a row covers (``None`` for a non-science row)."""
    cells = row_to_cells(row)
    return cells[0] if cells else None


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
    "S": (AXIS_WILDCARD,),
    "depth": (AXIS_WILDCARD,),
}


_WILDCARD_DEFAULT_AXES: Tuple[str, ...] = ("S", "depth")


def _enumerate_claim(chosen: Mapping[str, Sequence[str]]) -> List[HypervolumeCell]:
    """Build the deduped cartesian product of per-axis value lists into cells (collapsed axes fold to one)."""
    ordered_axes = list(_CELL_AXES)

    def field(kv: Dict[str, str], axis: str) -> str:
        if axis in kv:
            return kv[axis]
        return _COLLAPSED_REPRESENTATIVES.get(axis, "")

    cells: List[HypervolumeCell] = []
    for combo in itertools.product(*(tuple(chosen[a]) for a in ordered_axes)):
        kv = dict(zip(ordered_axes, combo))
        raw_sync = field(kv, "sync")
        sync = None if raw_sync in (None, "none", "") else raw_sync
        cells.append(
            HypervolumeCell(
                firing=field(kv, "firing"),
                sync=sync if sync is not None else "none",
                backend=field(kv, "backend"),
                vehicle=field(kv, "vehicle"),
                dataset=field(kv, "dataset"),
                regime=field(kv, "regime"),
                quantization=field(kv, "quantization"),
                pruning=field(kv, "pruning"),
                mapping_strategy=field(kv, "mapping_strategy"),
                s=field(kv, "S"),
                depth=field(kv, "depth"),
            )
        )
    seen: Dict[str, HypervolumeCell] = {}
    for cell in cells:
        seen.setdefault(cell.cell_key, cell)
    return list(seen.values())


def claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """Enumerate a claimed sub-product as concrete cells; unpinned axes take their single screened default.

    The LEGACY single-default claim — for the honest enumerated denominator use
    :func:`honest_claimed_subproduct`.
    """
    chosen: Dict[str, Tuple[str, ...]] = {}
    for axis in _CELL_AXES:
        if axis in axis_values:
            chosen[axis] = tuple(axis_values[axis])
        else:
            chosen[axis] = _CLAIMED_DEFAULTS[axis]
    return _enumerate_claim(chosen)


def honest_claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """The HONEST claimed sub-product whose denominator CONSUMES each unpinned axis's screening status.

    A SCREENED_COLLAPSED axis contributes one representative; an ENUMERATED_INTERACTING /
    ASSERTED_UNSCREENED axis is enumerated over its full domain (S / depth stay wildcard).
    """
    chosen: Dict[str, Tuple[str, ...]] = {}
    for axis in _CELL_AXES:
        if axis in axis_values:
            chosen[axis] = tuple(axis_values[axis])
            continue
        if axis in _WILDCARD_DEFAULT_AXES:
            chosen[axis] = _CLAIMED_DEFAULTS[axis]
            continue
        chosen[axis] = HypervolumeAxis.get(axis).values
    return _enumerate_claim(chosen)


def _parse_ts(value: Any) -> Optional[_dt.date]:
    """Parse an ISO ``YYYY-MM-DD`` string OR a Unix-epoch number (the ledger writes both) to a date."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _dt.datetime.utcfromtimestamp(float(value)).date()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    try:
        return _dt.datetime.utcfromtimestamp(float(text)).date()
    except (ValueError, OverflowError, OSError):
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class FlagMetadata:
    """Owner + aging metadata for one VALID_FLAGGED cell — the flag-aging schema.

    ``owner`` ``None`` ⇒ UNOWNED; ``age_days`` is measured vs the report's ``now_ts``;
    ``fix_path`` is set when the flag has a KNOWN fix (a placement-fixable offload flip).
    """

    cell_key: str
    owner: Optional[str]
    flag_ts: Optional[str]
    age_days: Optional[int]
    fix_path: Optional[str] = None

    @property
    def is_unowned(self) -> bool:
        return not (self.owner and str(self.owner).strip())


@dataclass(frozen=True)
class CoverageReport:
    """The measured coverage of a claimed sub-product over a ledger.

    ``cell_status`` is the GROUP BY result (each covered cell → its worst tier);
    ``coverage_fraction`` = covered / claimed. ``research_gap_frontier`` and
    ``placement_fixable_frontier`` union the respective ops over VALID_FLAGGED cells.
    """

    cell_status: Dict[str, CoverageStatus]
    tier_counts: Dict[CoverageStatus, int]
    research_gap_frontier: List[str]
    placement_fixable_frontier: List[str] = field(default_factory=list)
    claimed_cells: Tuple[HypervolumeCell, ...] = ()
    flag_metadata: Tuple[FlagMetadata, ...] = ()
    attribution_fidelity: Dict[str, AttributionFidelity] = field(default_factory=dict)
    _covered_keys: frozenset = field(default_factory=frozenset)

    @property
    def claimed_subproduct_size(self) -> int:
        """The claimed sub-product SIZE — the denominator printed next to the fraction."""
        return len(self.claimed_cells)

    @property
    def aged_unowned_flags(self) -> List[FlagMetadata]:
        """Flagged cells that have NO owner (regardless of age) — the aging worklist."""
        return [m for m in self.flag_metadata if m.is_unowned]

    def _matching_statuses(self, claimed: HypervolumeCell) -> List[CoverageStatus]:
        """The tiers of every COVERED cell that satisfies ``claimed`` (wildcard-aware)."""
        statuses: List[CoverageStatus] = []
        for key, status in self.cell_status.items():
            if cell_covers(claimed, HypervolumeCell.from_key(key)):
                statuses.append(status)
        return statuses

    @property
    def covered_cell_count(self) -> int:
        return len(self.cell_status)

    @property
    def claimed_cell_count(self) -> int:
        return len(self.claimed_cells)

    @property
    def covered_claimed_count(self) -> int:
        return sum(1 for c in self.claimed_cells if self._matching_statuses(c))

    @property
    def coverage_fraction(self) -> float:
        if not self.claimed_cells:
            return 0.0
        return self.covered_claimed_count / self.claimed_cell_count

    @property
    def untested_frontier(self) -> List[HypervolumeCell]:
        return [c for c in self.claimed_cells if not self._matching_statuses(c)]

    def status_for(self, cell: HypervolumeCell) -> CoverageStatus:
        matches = self._matching_statuses(cell)
        if not matches:
            return CoverageStatus.UNTESTED
        return max(matches, key=lambda s: _TIER_SEVERITY[s])

    def to_dict(self) -> Dict[str, Any]:
        # No merged valid-tier headline: VALID and VALID_FLAGGED are always reported separately.
        return {
            "covered_cell_count": self.covered_cell_count,
            "claimed_cell_count": self.claimed_cell_count,
            "claimed_subproduct_size": self.claimed_subproduct_size,
            "covered_claimed_count": self.covered_claimed_count,
            "coverage_fraction": self.coverage_fraction,
            "tier_counts": {s.value: n for s, n in self.tier_counts.items()},
            "cell_status": {k: v.value for k, v in sorted(self.cell_status.items())},
            "untested_frontier": [c.cell_key for c in self.untested_frontier],
            "research_gap_frontier": list(self.research_gap_frontier),
            "placement_fixable_frontier": list(self.placement_fixable_frontier),
            "attribution_fidelity": {
                region: fid.value for region, fid in self.attribution_fidelity.items()
            },
            "flag_metadata": [
                {
                    "cell_key": m.cell_key,
                    "owner": m.owner,
                    "flag_ts": m.flag_ts,
                    "age_days": m.age_days,
                    "fix_path": m.fix_path,
                }
                for m in self.flag_metadata
            ],
        }


_PLACEMENT_FLAG_MARKER = "PLACEMENT"

PLACEMENT_FIXABLE_DEFAULT_OWNER = "program:placement-offload"
PLACEMENT_FIXABLE_FIX_PATH = "set encoding_layer_placement=offload"


def _is_placement_fixable_flag(row: Mapping[str, Any]) -> bool:
    """True iff a VALID_FLAGGED row's flag is PLACEMENT-FIXABLE (a placement_fixable_ops encoder with no research gap)."""
    gaps, placement = _mine_flagged_ops(row)
    return bool(placement) and not gaps


def _mine_flagged_ops(row: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """``(research_gap_ops, placement_fixable_ops)`` named by one VALID_FLAGGED row.

    Prefers the structured fields; falls back to deriving the category from the
    ``deployment_validity`` tier suffix for a live ledger row that predates them.
    """
    structured_gaps = [str(op) for op in (row.get("research_gap_ops") or ())]
    structured_placement = [str(op) for op in (row.get("placement_fixable_ops") or ())]
    if structured_gaps or structured_placement:
        return structured_gaps, structured_placement

    text = str(row.get("deployment_validity") or "").upper()
    if _PLACEMENT_FLAG_MARKER in text:
        return [], ["encoding_layer(placement)"]
    return ["unsupported_host_op"], []


def _attribution_fidelity_map() -> Dict[str, AttributionFidelity]:
    """The per-region attribution-fidelity map, marking the KNOWN-CRACKED regions ``VALUE_DOMAIN_ONLY``."""
    return {region: AttributionFidelity.VALUE_DOMAIN_ONLY for region in KNOWN_CRACKED_REGIONS}


def _flag_owner_of(row: Mapping[str, Any]) -> Optional[str]:
    """The flag owner named by a row (``flag_owner`` / ``owner``), else ``None``."""
    owner = row.get("flag_owner") or row.get("owner")
    owner = str(owner).strip() if owner is not None else ""
    return owner or None


def _flag_ts_of(row: Mapping[str, Any]) -> Optional[str]:
    """The flag timestamp named by a row (``flag_ts`` / ``ts``), else ``None``."""
    ts = row.get("flag_ts") or row.get("ts")
    return str(ts) if ts else None


def coverage_report(
    rows: Iterable[Mapping[str, Any]],
    claimed_subproduct: Optional[Sequence[HypervolumeCell]] = None,
    now_ts: Optional[str] = None,
) -> CoverageReport:
    """GROUP BY the ledger by hypervolume cell-key + validity tier → coverage.

    A cell's status is the WORST tier of its rows; the frontiers union the flag ops over
    VALID_FLAGGED cells. When ``claimed_subproduct`` is given, coverage is measured against it.
    """
    materialized = list(rows)
    now = _parse_ts(now_ts) or _dt.date.today()

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

    research_gaps: set = set()
    placement_fixable: set = set()
    flag_meta: Dict[str, FlagMetadata] = {}
    for row, tier, cells in row_cells:
        if tier is not CoverageStatus.VALID_FLAGGED:
            continue
        flagged_cells = [
            cell
            for cell in cells
            if cell_status[cell.cell_key] is CoverageStatus.VALID_FLAGGED
        ]
        if not flagged_cells:
            continue
        gaps, placement = _mine_flagged_ops(row)
        research_gaps.update(gaps)
        placement_fixable.update(placement)
        owner = _flag_owner_of(row)
        fix_path = None
        if _is_placement_fixable_flag(row):
            fix_path = PLACEMENT_FIXABLE_FIX_PATH
            if owner is None:
                owner = PLACEMENT_FIXABLE_DEFAULT_OWNER
        flag_ts = _flag_ts_of(row)
        flagged_on = _parse_ts(flag_ts)
        age_days = (now - flagged_on).days if flagged_on is not None else None
        for cell in flagged_cells:
            flag_meta.setdefault(
                cell.cell_key,
                FlagMetadata(
                    cell_key=cell.cell_key,
                    owner=owner,
                    flag_ts=flag_ts,
                    age_days=age_days,
                    fix_path=fix_path,
                ),
            )

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
        flag_metadata=tuple(flag_meta[k] for k in sorted(flag_meta)),
        attribution_fidelity=_attribution_fidelity_map(),
        _covered_keys=covered_keys,
    )
