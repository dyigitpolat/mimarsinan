"""The typed hypervolume axis model: deployment axes, screening statuses, and collapse rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

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


def collapse_orthogonal_axes(axes: Sequence[HypervolumeAxis]) -> Tuple[HypervolumeAxis, ...]:
    """Drop every SCREENED_COLLAPSED axis, returning the active product the cell key is built over."""
    return tuple(a for a in axes if a.screening_status is not ScreeningStatus.SCREENED_COLLAPSED)


def collapsed_axis_representatives(axes: Sequence[HypervolumeAxis] = AXES) -> Dict[str, str]:
    """The SSOT ``{collapsed axis name → representative}`` map (falling back to the first screened value)."""
    reps: Dict[str, str] = {}
    for axis in axes:
        if axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED:
            reps[axis.name] = axis.representative or (axis.values[0] if axis.values else "")
    return reps


def interacting_axes(axes: Sequence[HypervolumeAxis] = AXES) -> Tuple[HypervolumeAxis, ...]:
    """The axes counted INTERACTING (enumerated) in the honest denominator — the non-collapsed axes."""
    return collapse_orthogonal_axes(axes)


def active_axes() -> Tuple[HypervolumeAxis, ...]:
    """The active (non-collapsed) hypervolume axes — the cell key's coordinates."""
    return collapse_orthogonal_axes(AXES)
