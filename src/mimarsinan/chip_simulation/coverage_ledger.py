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


# The per-axis WILDCARD sentinel — the single source of truth for "this axis is not
# constrained". It is the value ``row_to_cells`` records when a row carries no value
# for an axis (S/depth unknown), AND the default a ``claimed_subproduct`` gives an
# UNPINNED axis. In coverage matching a claimed-cell coordinate set to the wildcard
# matches ANY covered value on that axis, so a claim that does not pin S/depth is
# covered when the recipe was tested at any S/depth (the deep_cnn rows carry concrete
# S=4 / depth=8 yet a depth-agnostic claim must still match them).
AXIS_WILDCARD = "any"


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


class ScreeningStatus(Enum):
    """The KEYSTONE classification that gates whether an axis may collapse.

    The coverage DENOMINATOR is a function of this status so collapse-on-a-hunch is
    structurally impossible:

    * ``SCREENED_COLLAPSED`` — a cheap screen PROVED the axis's values equivalent, so
      it collapses to one representative and is NOT a cell coordinate. It REQUIRES a
      non-empty ``screening_artifact`` (a doc / test / result ref); constructing one
      without an artifact RAISES.
    * ``ENUMERATED_INTERACTING`` — the axis is PROVEN to interact (the death-cascade
      firing×sync law, the dual-axis depth×dataset law), so its cross-product is the
      real surface; counted interacting (enumerated) in the denominator.
    * ``ASSERTED_UNSCREENED`` — no screen has been run yet; we make NO collapse claim,
      so it is counted interacting (enumerated) until a screen (P3) earns the collapse.
    """

    SCREENED_COLLAPSED = "screened_collapsed"
    ENUMERATED_INTERACTING = "enumerated_interacting"
    ASSERTED_UNSCREENED = "asserted_unscreened"


class AttributionFidelity(Enum):
    """How TRUSTWORTHY a covered region's per-neuron attribution is.

    ``ATTRIBUTION`` — the per-neuron reassembly is bit-exact (the validated corner).
    ``VALUE_DOMAIN_ONLY`` — only the VALUE-domain (deployed accuracy) is bit-exact; the
    per-neuron ATTRIBUTION reassembly is not gated in production there (GAP-1: the C3
    fix makes coalescing+output-tiling bit-exact in the fidelity HARNESS, but the
    production NF↔SCM gate is identity-mapping-only so the fragment path is unexercised
    in deployment; the residual Tier-1 merge). Marking these NOW keeps the coverage
    instrument honest about what it can and cannot attribute.
    """

    ATTRIBUTION = "attribution"
    VALUE_DOMAIN_ONLY = "value_domain_only"


@dataclass(frozen=True)
class HypervolumeAxis:
    """One typed deployment axis of the hypervolume, with its coverage classification.

    ``name`` is the axis (a ``HypervolumeCell`` field); ``kind`` is the legacy
    ORTHOGONAL vs INTERACTING tag; ``screening_status`` is the KEYSTONE — it gates
    collapse and the denominator. ``interacts_with`` names the axes it is tested
    jointly with; ``values`` is the (small) screened domain; ``representative`` is the
    single value a SCREENED_COLLAPSED axis drops to; ``screening_artifact`` is the
    doc/test/result ref that JUSTIFIES the collapse (mandatory for SCREENED_COLLAPSED);
    ``justification`` is the human-facing reason. ``collapsed`` is DERIVED — it is
    exactly ``screening_status is SCREENED_COLLAPSED`` (no independent hunch flag).
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


# The KNOWN-CRACKED regions whose per-neuron attribution is VALUE_DOMAIN_ONLY (the
# deployed accuracy is bit-exact, but the per-neuron reassembly is NOT exercised in
# production): the residual Tier-1 merge is the remaining attribution overcount in the
# fused-mapping reassembler. Marking them NOW keeps the instrument honest about what it
# can attribute.
#
# GAP-1 STATUS (Wave-2 C3 reconciliation): C3 fixed the fidelity-HARNESS reassembler —
# the joint ``(perceptron_output_slice, ir_id)`` keying makes coalescing+output-tiling
# per-neuron attribution bit-exact in ``tests/integration/_split_reassembly.py`` (locked
# by ``test_coalescing_neuron_split_attribution.py``). But the PRODUCTION NF↔SCM gate
# (``nf_scm_parity._group_record_by_perceptron``) asserts identity-mapping-only
# (one placement/core, ``split_group_id is None``) and runs on a freshly-built identity
# mapping (``build_identity_mapping_for_pipeline``), so the coalesced/output-tiled
# FRAGMENT attribution path is NOT exercised in deployment — only the harness exercises
# it. GAP-1 therefore STAYS VALUE_DOMAIN_ONLY: production per-neuron attribution under
# coalescing+output-tiling is not gated. (See docs/research/HYPERVOLUME.md.)
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
    """Drop every SCREENED_COLLAPSED axis — the active product the cell key is built over.

    Collapse CONSUMES ``screening_status``: ONLY a ``SCREENED_COLLAPSED`` axis (one whose
    values a cheap screen PROVED equivalent, e.g. ``encoding_placement`` offload≡subsume)
    contributes a single representative and is therefore NOT a coordinate of the
    hypervolume cell. ``ENUMERATED_INTERACTING`` and ``ASSERTED_UNSCREENED`` axes are
    BOTH counted interacting (they survive as coordinates) — an axis cannot collapse
    without a linked artifact. Returns the axes that remain (the active product).
    """
    return tuple(
        a
        for a in axes
        if a.screening_status is not ScreeningStatus.SCREENED_COLLAPSED
    )


def collapsed_axis_representatives(
    axes: Sequence[HypervolumeAxis] = AXES,
) -> Dict[str, str]:
    """The SSOT ``{collapsed axis name → representative}`` map.

    Every ``SCREENED_COLLAPSED`` axis folds to a single representative value (its
    ``__post_init__`` already guarantees a linked artifact). A collapsed axis is NOT
    a cell coordinate, so any cell value on it must canonicalize to the representative
    — this map is the single place that mapping lives, consumed by ``cell_key`` /
    ``cert_cell`` (fold ``backend``) and ``HypervolumeCell.from_key`` (default a
    dropped extending axis like ``mapping_strategy``). A collapsed axis with no
    explicit ``representative`` falls back to its first screened value.
    """
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
    """The axes counted INTERACTING (enumerated) in the honest denominator.

    Exactly the non-``SCREENED_COLLAPSED`` axes: ``ENUMERATED_INTERACTING`` (proven to
    interact) and ``ASSERTED_UNSCREENED`` (no screen yet) are BOTH enumerated, so no
    axis inflates coverage without a linked artifact.
    """
    return collapse_orthogonal_axes(axes)


def active_axes() -> Tuple[HypervolumeAxis, ...]:
    """The active (non-collapsed) hypervolume axes — the cell key's coordinates."""
    return collapse_orthogonal_axes(AXES)


# The cell-key coordinate order (collapsed axes excluded). The (firing × sync ×
# backend) prefix is the embedded CertificationCell; the rest extend it.
_CELL_AXES: Tuple[str, ...] = tuple(a.name for a in collapse_orthogonal_axes(AXES))

# SSOT: the representative each SCREENED_COLLAPSED axis folds to. ``cell_key`` /
# ``cert_cell`` fold a collapsed axis's value to its representative so two rows
# differing ONLY in a collapsed axis (e.g. backend nevresim vs sanafe, or
# mapping_strategy identity vs packed) map to the SAME cell; ``from_key`` defaults a
# dropped extending axis (mapping_strategy) to it so the round-trip still works.
_COLLAPSED_REPRESENTATIVES: Dict[str, str] = collapsed_axis_representatives(AXES)


@dataclass(frozen=True)
class HypervolumeCell:
    """The FULL-TUPLE coverage cell — extends ``CertificationCell`` to the config tuple.

    The (firing × sync × backend) sub-coordinate is exactly an E6
    :class:`CertificationCell` (``cert_cell``); the remaining fields (vehicle,
    dataset, regime, quantization, pruning, mapping_strategy, S, depth) extend it so
    each science-valid ledger row maps to exactly one hypervolume cell. ``depth`` is a
    real coordinate so a shallow INVALID and a deeper VALID_FLAGGED at the same recipe
    do not collapse. The collapsed ``encoding_placement`` axis is NOT a field
    (offload≡subsume).
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
        """The embedded (firing × sync × backend) E6 cell.

        When ``backend`` is a SCREENED_COLLAPSED axis it folds to its representative
        so two cells differing ONLY in backend (a faithfulness axis) produce the SAME
        cert prefix — the collapse must not leak distinct cells through the cert key.
        """
        sync = None if self.sync in (None, "none", "") else self.sync
        backend = _COLLAPSED_REPRESENTATIVES.get("backend", self.backend)
        return CertificationCell(firing=self.firing, sync=sync, backend=backend)

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
        """Parse a canonical full-tuple key back into a cell.

        A SCREENED_COLLAPSED extending axis (e.g. ``mapping_strategy``) is dropped
        from the key, so it is absent from the parsed segments; it defaults to its
        representative (the SSOT ``_COLLAPSED_REPRESENTATIVES``) so the round-trip
        ``from_key(cell.cell_key)`` reconstructs the canonical cell. ``backend`` is
        already folded inside the cert prefix.
        """
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


# axis-name → HypervolumeCell field name (``S`` is stored as ``s``).
_FIELD_FOR_AXIS: Dict[str, str] = {a: a for a in _CELL_AXES}
_FIELD_FOR_AXIS["S"] = "s"

# The HypervolumeCell fields a claimed cell is matched on (every cell coordinate).
_MATCH_FIELDS: Tuple[str, ...] = tuple(_FIELD_FOR_AXIS[a] for a in _CELL_AXES)


def _coord(cell: "HypervolumeCell", field_name: str) -> str:
    """A cell's coordinate value for a match field, with ``sync=None`` read as ``none``."""
    value = getattr(cell, field_name)
    if field_name == "sync" and value in (None, ""):
        return "none"
    return str(value)


def cell_covers(claimed: "HypervolumeCell", covered: "HypervolumeCell") -> bool:
    """True iff a COVERED cell satisfies a CLAIMED cell under wildcard semantics.

    A claimed coordinate set to :data:`AXIS_WILDCARD` matches any covered value on that
    axis (the claim does not constrain it); every other claimed coordinate must EQUAL
    the covered one. So a depth/S-agnostic claim is covered by a row tested at a
    concrete depth/S, but a claim that pins an axis is only covered by an exact hit.
    """
    for field_name in _MATCH_FIELDS:
        claim_value = _coord(claimed, field_name)
        if claim_value == AXIS_WILDCARD:
            continue
        if claim_value != _coord(covered, field_name):
            return False
    return True


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


def _cell_with_sync(row: Mapping[str, Any], vehicle: str, sync: str) -> HypervolumeCell:
    coords = cell_coordinates_from_row(row, sync=sync, axis_wildcard=AXIS_WILDCARD)
    data = coords.as_cell_kwargs()
    data["vehicle"] = str(vehicle)
    return HypervolumeCell(**data)


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
    return [_cell_with_sync(row, vehicle, sync) for sync in syncs_from_row(row)]


def row_to_cell(row: Mapping[str, Any]) -> Optional[HypervolumeCell]:
    """The first hypervolume cell a row covers (``None`` for a non-science row).

    Convenience over :func:`row_to_cells` for the common single-sync row; a
    dual-schedule row returns only its first cell (use :func:`row_to_cells` for both).
    """
    cells = row_to_cells(row)
    return cells[0] if cells else None


# Defaults the claimed-product builder fills for any axis the caller does not pin.
# These are the SAME per-axis defaults ``row_to_cells`` (via ``_cell_with_sync``)
# records when a row carries no value for the axis — the single source of truth so a
# real claim's cells match its covered cells. The breadth axes the rows always carry
# explicitly (backend/quantization/pruning/mapping_strategy/regime) take their screened
# representative; ``S`` and ``depth`` take the ``AXIS_WILDCARD`` so an S/depth-agnostic
# claim is covered when the recipe was tested at any S/depth (the deep_cnn rows carry
# concrete S=4 / depth=8, but a claim that does not pin them must still match).
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


# The two enumerated-interacting axes whose SSOT default is the WILDCARD (rows carry
# concrete S/depth and a wildcard claim is covered by any concrete value). They are
# cell COORDINATES (counted interacting), but the honest denominator keeps them
# wildcard rather than exploding into untested S/depth values that would be a strictly
# HARSHER (different) claim than "any S/depth".
_WILDCARD_DEFAULT_AXES: Tuple[str, ...] = ("S", "depth")


def _enumerate_claim(chosen: Mapping[str, Sequence[str]]) -> List[HypervolumeCell]:
    """Build the deduped cartesian product of per-axis value lists into cells.

    Only the ACTIVE (non-collapsed) axes vary in the product; a SCREENED_COLLAPSED
    axis (``backend`` / ``mapping_strategy``) is NOT a coordinate, so it folds to its
    representative on every cell. ``cell_key`` re-folds collapsed coordinates, so even
    if a caller pins a collapsed axis to several values the cells dedupe to one.
    """
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
    # Dedupe (a collapsed axis pinned to multiple values yields identical cells).
    seen: Dict[str, HypervolumeCell] = {}
    for cell in cells:
        seen.setdefault(cell.cell_key, cell)
    return list(seen.values())


def claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """Enumerate a claimed sub-product of the hypervolume as concrete cells.

    Each keyword pins an axis to a list of values; unpinned axes take their single
    screened default. A COLLAPSED axis (``encoding_placement``) is IGNORED — pinning
    it to both values does not double the product (offload≡subsume), so the claimed
    set is invariant to it. Returns the cartesian product as :class:`HypervolumeCell`s.

    This is the LEGACY single-default claim (an unpinned breadth axis collapses to one
    screened default). For the HONEST denominator that ENUMERATES every unscreened /
    interacting axis (so no axis inflates coverage without a linked artifact), use
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
    """The HONEST claimed sub-product — the denominator CONSUMES screening status.

    Each keyword pins an axis. For every UNPINNED axis the denominator is built from the
    axis's ``screening_status``:

    * a ``SCREENED_COLLAPSED`` axis (``encoding_placement``) contributes its single
      representative — it cannot enlarge the denominator (offload≡subsume, artifact-
      backed);
    * an ``ENUMERATED_INTERACTING`` or ``ASSERTED_UNSCREENED`` axis is ENUMERATED over
      its full screened domain (the ``S`` / ``depth`` wildcard axes excepted — they stay
      wildcard, a strictly weaker "any S/depth" claim). A bigger denominator = LOWER
      honest coverage, so collapse-on-a-hunch is impossible: an unscreened axis can only
      shrink the denominator once a screen (P3) earns the collapse.
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
    """Parse an ISO ``YYYY-MM-DD`` string OR a Unix-epoch number to a date.

    The live ledger writes ``ts`` as a Unix-epoch FLOAT (e.g. ``1782258504.2``), while
    explicit flag timestamps are ISO strings — both must yield a real date so the aging
    check has teeth on real data. Returns ``None`` only when truly unparseable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _dt.datetime.utcfromtimestamp(float(value)).date()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    # A numeric string is also an epoch (the ledger sometimes stringifies ts).
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

    ``cell_key`` names the flagged cell; ``owner`` is who owns driving the flag to
    resolution (``None`` ⇒ UNOWNED); ``flag_ts`` is when it was raised; ``age_days`` is
    its age vs the report's ``now_ts`` (``None`` when either timestamp is missing);
    ``fix_path`` is the named resolution step when the flag has a KNOWN fix (e.g. a
    placement-fixable flag's encoding-offload flip), else ``None``. An UNOWNED flag aged
    past the CI threshold is a guard violation — a flag must not rot without an owner;
    a placement-fixable flag is auto-assigned the standing placement-offload owner so it
    is never UNOWNED.
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

    ``cell_status`` is the GROUP BY result: each COVERED cell → its worst tier.
    ``tier_counts`` tallies the covered cells by tier. When a ``claimed_subproduct``
    is supplied, ``claimed_cells`` is it, ``covered_claimed_count`` is how many were
    tested, ``coverage_fraction`` = covered / claimed, ``untested_frontier`` is the
    claimed cells with NO matching row, and ``status_for`` answers a claimed cell's
    status (UNTESTED when uncovered). A claimed cell is COVERED when at least one
    covered cell satisfies it under :func:`cell_covers` wildcard semantics — a claim
    that leaves S/depth unpinned matches a recipe tested at any S/depth; its status is
    the WORST tier among the covered cells it matches (conservative). ``research_gap_
    frontier`` is the sorted, deduped union of ``research_gap_ops`` over the
    VALID_FLAGGED cells — the future-conversion targets (host ops with NO on-chip SNN
    mapping yet). ``placement_fixable_frontier`` is the parallel union of
    ``placement_fixable_ops`` — supported encoders host-placed under ``subsume`` that an
    ``offload`` flip would map on-chip (un-flagging the cell); those are NOT research
    gaps, so they are reported separately.
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
        """The claimed sub-product SIZE — the denominator printed next to the fraction.

        The report ALWAYS surfaces this so a coverage fraction is never a bare ``0.75``
        with no denominator: ``0.75`` over 4 cells and over 4000 cells are very
        different honesty claims.
        """
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
        # The two valid tiers are ALWAYS reported separately — there is deliberately no
        # merged ``valid_total`` / ``covered_valid_total`` headline that fuses VALID and
        # VALID_FLAGGED (the CI guard fails on such a key; the instrument must never
        # claim a flagged cell as plainly valid).
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


# The tier-suffix back-compat marker the live ledger uses: a ``..._placement`` flag is
# a placement fix (offloadable encoder), NOT a research gap. Any other flag category is
# a research gap (an unsupported host op with no on-chip SNN mapping yet).
_PLACEMENT_FLAG_MARKER = "PLACEMENT"

# The DEFAULT owner + fix-path auto-assigned to a placement-fixable flag that carries no
# explicit owner. A placement-fixable flag is NOT drift: it has a KNOWN fix (flip the
# encoding-layer placement to offload, mapping the host-placed encoder on-chip and
# un-flagging the cell), so it is owned by the standing placement-offload program rather
# than left UNOWNED to rot. A research-gap flag (an unsupported host op with no on-chip
# SNN mapping) gets NO default owner — it is a genuine open research target.
PLACEMENT_FIXABLE_DEFAULT_OWNER = "program:placement-offload"
PLACEMENT_FIXABLE_FIX_PATH = "set encoding_layer_placement=offload"


def _is_placement_fixable_flag(row: Mapping[str, Any]) -> bool:
    """True iff a VALID_FLAGGED row's flag is PLACEMENT-FIXABLE (a known offload fix).

    A flag is placement-fixable when it names a structured ``placement_fixable_ops``
    encoder AND names NO ``research_gap_ops`` (a row that ALSO owes a real research gap
    is not auto-resolvable), or — for a live ledger row that predates those fields —
    when its ``deployment_validity`` tier carries the ``_placement`` suffix.
    """
    gaps, placement = _mine_flagged_ops(row)
    return bool(placement) and not gaps


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


def _attribution_fidelity_map() -> Dict[str, AttributionFidelity]:
    """The per-region attribution-fidelity map — mark the KNOWN-CRACKED regions NOW.

    The KNOWN-CRACKED regions (GAP-1 coalescing+output-tiling attribution at VGG scale,
    fixed in the fidelity harness by C3 but identity-mapping-only in the production
    gate; the residual Tier-1 merge) are ``VALUE_DOMAIN_ONLY`` — their deployed accuracy
    is bit-exact but the per-neuron ATTRIBUTION reassembly is not gated in deployment.
    Every other region is full ``ATTRIBUTION``. The instrument carries this so coverage
    is never silently claimed as attributable where only the value domain is sound.
    """
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

    Each science-valid row (one carrying a ``deployment_validity`` tier) is mapped to
    its cell and tier; a cell's status is the WORST tier of its rows (INVALID >
    VALID_FLAGGED > VALID, the conservative read). ``research_gap_frontier`` unions the
    ``research_gap_ops`` of the VALID_FLAGGED cells (the future-conversion targets);
    ``placement_fixable_frontier`` unions their ``placement_fixable_ops`` (offloadable
    encoders — NOT research gaps). When ``claimed_subproduct`` is given, the report
    measures coverage against it: the fraction tested, the named UNTESTED frontier, and
    each claimed cell's status. Non-science rows (no tier) and rows with no model are
    skipped.

    ``flag_metadata`` carries each FINAL-VALID_FLAGGED cell's owner + age (vs ``now_ts``,
    today's date if unset) so an unowned flag cannot rot silently; ``attribution_fidelity``
    marks the KNOWN-CRACKED regions VALUE_DOMAIN_ONLY.
    """
    materialized = list(rows)
    now = _parse_ts(now_ts) or _dt.date.today()

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

    # Pass 2: mine the flag ops + owner/aging metadata only from rows whose cell's FINAL
    # tier is FLAGGED (a cell demoted to INVALID is no longer a flagged-cell target).
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
        # A placement-fixable flag has a KNOWN fix (the encoding-offload flip), so it is
        # auto-assigned the standing placement-offload owner + fix-path rather than left
        # UNOWNED — an explicit owner on the row always wins.
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
