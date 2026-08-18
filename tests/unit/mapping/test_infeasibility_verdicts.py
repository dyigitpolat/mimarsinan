"""TS4: what a failed pack PROVES, versus what the greedy packer merely refused.

A census of search failures cannot act on "packing failed": a chip that
provably cannot host the program at all is a different fact from a packer that
walked into a corner on a program some other placement order would have taken.
The provers here are the proven half — each fires only where no packing exists
under the declared permissions — and everything they decline stays
``heuristic_failed`` by construction.
"""

from __future__ import annotations

from typing import Any, cast

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.packing.infeasibility_proofs import (
    VERDICT_FEASIBLE,
    VERDICT_HEURISTIC_FAILED,
    VERDICT_PROVEN_INFEASIBLE,
    cells_exceed_total_capacity,
    ensure_verdict_tag,
    failed_pack_verdict,
    largest_extents,
    no_core_type_fits,
    tag_with_verdict,
    verdict_of_message,
)
from mimarsinan.mapping.verification.verifier import verify_hardware_config
from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin

from .test_pass_planner_guards import UNPACKABLE_CORES, UNPACKABLE_SPECS

# Four 4x4 softcores over four 8x8 cores: the packer fits them with room to spare.
FITTING_TYPES = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=4)]
FITTING_SPECS = [
    LayoutSoftCoreSpec(input_count=4, output_count=4, name=f"f{i}") for i in range(4)
]

# 3 x 64 committed cells against a single 8x8 crossbar: no permission can shrink
# the demand below the declared total, so infeasibility is provable.
OVERFLOW_TYPES = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=1)]
OVERFLOW_SPECS = [
    LayoutSoftCoreSpec(input_count=8, output_count=8, name=f"o{i}") for i in range(3)
]

# 8 neurons where no declared type has more than 4, splitting withheld: the
# demand fits no core, and the cell total (32) is far under capacity (128).
TALL_TYPES = [LayoutHardCoreType(max_axons=8, max_neurons=4, count=4)]
TALL_SPECS = [LayoutSoftCoreSpec(input_count=4, output_count=8, name="tall")]

# 8 axons over two 4-axon cores: the placement engine FUSES hardcores whatever
# the declared coalescing flag says, so this packs — an axon-shaped proof at the
# pack seam would be unsound.
WIDE_TYPES = [LayoutHardCoreType(max_axons=4, max_neurons=4, count=2)]
WIDE_SPECS = [LayoutSoftCoreSpec(input_count=8, output_count=4, name="wide")]

# One axon past what fusing every declared core could reach (4 x 2 = 8 < 9), and
# only 9 committed cells: genuinely infeasible, but not by a sound pack-seam
# proof — the honest verdict is "refused", not "proven".
TOO_WIDE_SPECS = [LayoutSoftCoreSpec(input_count=9, output_count=1, name="too_wide")]

# The guards' composition ``[[a, c], [b]]``: pass 0 is packable in principle
# (a -> the 4x4 core, c -> the 3x8 core) and the greedy packer still refuses it.
GREEDY_CORNER_SPECS = [sc for sc in UNPACKABLE_SPECS if sc.name in ("a", "c")]
GREEDY_CORNER_TYPES = [
    LayoutHardCoreType(
        max_axons=int(ct["max_axons"]),
        max_neurons=int(ct["max_neurons"]),
        count=int(ct["count"]),
    )
    for ct in UNPACKABLE_CORES
]


class TestNoCoreTypeFits:
    """Which permission relaxes which dimension — the proof turns on exactly that."""

    def test_a_neuron_demand_no_declared_type_covers_is_a_proof(self):
        assert no_core_type_fits(
            TALL_SPECS, TALL_TYPES, neurons_may_split=False, axons_may_spread=True,
        )

    def test_splitting_relaxes_the_neuron_demand(self):
        assert not no_core_type_fits(
            TALL_SPECS, TALL_TYPES, neurons_may_split=True, axons_may_spread=False,
        )

    def test_an_axon_demand_no_declared_type_covers_is_a_proof(self):
        assert no_core_type_fits(
            WIDE_SPECS, WIDE_TYPES, neurons_may_split=False, axons_may_spread=False,
        )

    def test_spreading_relaxes_the_axon_demand(self):
        assert not no_core_type_fits(
            WIDE_SPECS, WIDE_TYPES, neurons_may_split=False, axons_may_spread=True,
        )

    def test_splitting_cuts_neurons_and_so_cannot_cover_a_too_wide_core(self):
        assert no_core_type_fits(
            WIDE_SPECS, WIDE_TYPES, neurons_may_split=True, axons_may_spread=False,
        )

    def test_spreading_moves_axons_and_so_cannot_cover_a_too_tall_core(self):
        assert no_core_type_fits(
            TALL_SPECS, TALL_TYPES, neurons_may_split=False, axons_may_spread=True,
        )

    def test_one_type_covering_both_dimensions_is_no_proof(self):
        assert not no_core_type_fits(
            FITTING_SPECS, FITTING_TYPES,
            neurons_may_split=False, axons_may_spread=False,
        )

    def test_a_covering_type_anywhere_in_the_declaration_answers(self):
        types = [LayoutHardCoreType(max_axons=1, max_neurons=1, count=9)] + FITTING_TYPES
        assert not no_core_type_fits(
            FITTING_SPECS, types, neurons_may_split=False, axons_may_spread=False,
        )

    def test_nothing_to_place_is_never_a_proof(self):
        assert not no_core_type_fits(
            [], FITTING_TYPES, neurons_may_split=False, axons_may_spread=False,
        )

    def test_a_declaration_with_no_core_types_at_all_is_a_proof(self):
        assert no_core_type_fits(
            FITTING_SPECS, [], neurons_may_split=True, axons_may_spread=True,
        )

    def test_the_largest_demand_is_read_per_dimension(self):
        specs = [
            LayoutSoftCoreSpec(input_count=7, output_count=2, name="wide"),
            LayoutSoftCoreSpec(input_count=2, output_count=5, name="tall"),
        ]
        assert largest_extents(specs) == (7, 5)
        assert largest_extents([]) == (0, 0)


class TestCellsExceedTotalCapacity:
    """Committed cells against declared cells — no permission shrinks the demand."""

    def test_a_demand_above_the_declared_total_is_a_proof(self):
        assert cells_exceed_total_capacity(OVERFLOW_SPECS, OVERFLOW_TYPES)

    def test_a_demand_the_declaration_can_hold_is_no_proof(self):
        assert not cells_exceed_total_capacity(FITTING_SPECS, FITTING_TYPES)

    def test_an_exactly_saturating_demand_is_no_proof(self):
        specs = [LayoutSoftCoreSpec(input_count=8, output_count=8, name="exact")]
        assert not cells_exceed_total_capacity(specs, OVERFLOW_TYPES)

    def test_the_declared_count_multiplies_the_capacity(self):
        enough = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=3)]
        one_short = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=2)]
        assert not cells_exceed_total_capacity(OVERFLOW_SPECS, enough)
        assert cells_exceed_total_capacity(OVERFLOW_SPECS, one_short)

    def test_every_declared_type_contributes_its_capacity(self):
        split_declaration = [
            LayoutHardCoreType(max_axons=8, max_neurons=8, count=1),
            LayoutHardCoreType(max_axons=8, max_neurons=8, count=2),
        ]
        assert not cells_exceed_total_capacity(OVERFLOW_SPECS, split_declaration)

    def test_nothing_to_place_is_never_a_proof(self):
        assert not cells_exceed_total_capacity([], [])


class TestTheVerdictAPackCarries:
    def test_a_pack_that_fits_is_feasible(self):
        result = pack_layout(softcores=FITTING_SPECS, core_types=FITTING_TYPES)
        assert result.feasible
        assert result.verdict == VERDICT_FEASIBLE
        assert result.error is None

    def test_a_cell_demand_above_the_declared_crossbar_total_is_proven(self):
        result = pack_layout(softcores=OVERFLOW_SPECS, core_types=OVERFLOW_TYPES)
        assert not result.feasible
        assert result.verdict == VERDICT_PROVEN_INFEASIBLE

    def test_the_cell_proof_survives_every_permission(self):
        """Splitting and coalescing PARTITION a softcore's cells; neither shrinks
        the committed total, so the proof cannot be permissioned away."""
        result = pack_layout(
            softcores=OVERFLOW_SPECS, core_types=OVERFLOW_TYPES,
            allow_neuron_splitting=True, allow_coalescing=True,
        )
        assert not result.feasible
        assert result.verdict == VERDICT_PROVEN_INFEASIBLE

    def test_a_neuron_demand_no_type_covers_is_proven(self):
        result = pack_layout(softcores=TALL_SPECS, core_types=TALL_TYPES)
        assert not result.feasible
        assert result.verdict == VERDICT_PROVEN_INFEASIBLE
        assert not cells_exceed_total_capacity(TALL_SPECS, TALL_TYPES)

    def test_splitting_removes_the_neuron_proof_and_the_failure_with_it(self):
        result = pack_layout(
            softcores=TALL_SPECS, core_types=TALL_TYPES, allow_neuron_splitting=True,
        )
        assert result.feasible
        assert result.verdict == VERDICT_FEASIBLE

    def test_a_greedy_corner_is_refused_not_proven(self):
        """The guards' vehicle: pass 0 is packable in principle, and the greedy
        packer that places the tightest fit first still refuses it."""
        result = pack_layout(
            softcores=GREEDY_CORNER_SPECS, core_types=GREEDY_CORNER_TYPES,
        )
        assert not result.feasible
        assert result.verdict == VERDICT_HEURISTIC_FAILED

    def test_the_fusion_the_engine_always_performs_is_never_proven_away(self):
        """The placement engine fuses hardcores whatever the declared coalescing
        flag says, so the axon dimension is never tight at this seam."""
        packed = pack_layout(
            softcores=WIDE_SPECS, core_types=WIDE_TYPES, allow_coalescing=False,
        )
        assert packed.feasible and packed.verdict == VERDICT_FEASIBLE
        assert no_core_type_fits(
            WIDE_SPECS, WIDE_TYPES, neurons_may_split=False, axons_may_spread=False,
        ), "the axon-shaped proof this pack would have contradicted"

    def test_an_axon_demand_past_fusion_is_refused_rather_than_misproven(self):
        result = pack_layout(
            softcores=TOO_WIDE_SPECS, core_types=WIDE_TYPES, allow_coalescing=False,
        )
        assert not result.feasible
        assert result.verdict == VERDICT_HEURISTIC_FAILED

    def test_the_verdict_helper_answers_what_the_pack_reports(self):
        assert failed_pack_verdict(
            OVERFLOW_SPECS, OVERFLOW_TYPES,
            neurons_may_split=False, axons_may_spread=True,
        ) == VERDICT_PROVEN_INFEASIBLE
        assert failed_pack_verdict(
            GREEDY_CORNER_SPECS, GREEDY_CORNER_TYPES,
            neurons_may_split=False, axons_may_spread=True,
        ) == VERDICT_HEURISTIC_FAILED


class TestTheRefusalMessageCarriesItsClass:
    def test_a_proven_refusal_says_so_and_keeps_the_engine_text(self):
        result = pack_layout(softcores=OVERFLOW_SPECS, core_types=OVERFLOW_TYPES)
        assert result.error is not None
        assert result.error.startswith(f"[{VERDICT_PROVEN_INFEASIBLE}] ")
        assert "No more hard cores available" in result.error

    def test_an_unproven_refusal_says_that_instead(self):
        result = pack_layout(
            softcores=GREEDY_CORNER_SPECS, core_types=GREEDY_CORNER_TYPES,
        )
        assert result.error is not None
        assert result.error.startswith(f"[{VERDICT_HEURISTIC_FAILED}] ")

    def test_a_census_reads_the_class_back_off_the_message(self):
        for verdict in (VERDICT_PROVEN_INFEASIBLE, VERDICT_HEURISTIC_FAILED):
            assert verdict_of_message(tag_with_verdict(verdict, "boom")) == verdict

    def test_an_untagged_message_classifies_as_unproven(self):
        assert verdict_of_message("HW bin-packing infeasible") == VERDICT_HEURISTIC_FAILED
        assert ensure_verdict_tag("HW bin-packing infeasible") == (
            f"[{VERDICT_HEURISTIC_FAILED}] HW bin-packing infeasible"
        )

    def test_tagging_an_already_tagged_message_changes_nothing(self):
        tagged = tag_with_verdict(VERDICT_PROVEN_INFEASIBLE, "boom")
        assert ensure_verdict_tag(tagged) == tagged

    def test_an_unknown_bracket_prefix_is_not_mistaken_for_a_verdict(self):
        assert verdict_of_message("[C4] scheduling note") == VERDICT_HEURISTIC_FAILED
        assert ensure_verdict_tag("[C4] scheduling note").startswith(
            f"[{VERDICT_HEURISTIC_FAILED}] "
        )


class TestTheSearchFailureSelfClassifies:
    """``_packing_failure`` is what a failure census reads (the ViT-style lines)."""

    @staticmethod
    def _failure(error):
        host = cast(Any, JointLayoutMixin())
        pcfg = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 1}]}
        return host._packing_failure(cast(Any, None), error, OVERFLOW_SPECS, pcfg)

    def test_it_carries_the_class_the_pack_proved(self):
        packed = pack_layout(softcores=OVERFLOW_SPECS, core_types=OVERFLOW_TYPES)
        failure = self._failure(packed.error)
        assert verdict_of_message(failure.message) == VERDICT_PROVEN_INFEASIBLE
        assert "No more hard cores available" in failure.message
        assert "total_hw_capacity=64" in failure.message

    def test_a_classless_failure_is_reported_as_unproven(self):
        failure = self._failure(None)
        assert verdict_of_message(failure.message) == VERDICT_HEURISTIC_FAILED
        assert "HW bin-packing infeasible" in failure.message

    def test_a_scheduling_failure_is_never_promoted_to_proven(self):
        """Scheduling REUSES cores across passes, so a flat cell overflow proves
        nothing about a scheduled program — the class must be propagated, never
        recomputed here."""
        failure = self._failure("Scheduling infeasible: at least one softcore cannot be packed")
        assert verdict_of_message(failure.message) == VERDICT_HEURISTIC_FAILED


class TestTheVerifierAsksTheSameProver:
    """One home: the verifier's largest-softcore check IS ``no_core_type_fits``."""

    @staticmethod
    def _verify(**kwargs):
        return verify_hardware_config(
            list(TOO_WIDE_SPECS),
            [{"max_axons": 4, "max_neurons": 4, "count": 2}],
            **kwargs,
        )

    def test_with_no_permissions_it_names_both_dimensions(self):
        out = self._verify()
        assert out["field_errors"]["core_types"].startswith(
            "No core type fits the largest soft core (9 axons, 1 neurons)."
        )

    def test_with_splitting_it_names_the_axon_count(self):
        out = self._verify(allow_neuron_splitting=True)
        assert out["field_errors"]["core_types"].startswith(
            "No core type fits the largest soft core's axon count (9 axons)."
        )

    def test_with_coalescing_it_names_the_neuron_count(self):
        out = verify_hardware_config(
            list(TALL_SPECS),
            [{"max_axons": 8, "max_neurons": 4, "count": 4}],
            allow_coalescing=True,
        )
        assert out["field_errors"]["core_types"].startswith(
            "No core type fits the largest soft core's neuron count (8 neurons)."
        )

    def test_a_covered_demand_raises_no_core_type_error(self):
        out = verify_hardware_config(
            list(FITTING_SPECS),
            [{"max_axons": 8, "max_neurons": 8, "count": 4}],
        )
        assert out["feasible"]
        assert out["field_errors"] == {}


class TestFeasibleBoolSemanticsAreUntouched:
    """The A/B: the verdict is additive — nothing downstream reads a new answer."""

    VEHICLES = (
        (FITTING_SPECS, FITTING_TYPES),
        (OVERFLOW_SPECS, OVERFLOW_TYPES),
        (TALL_SPECS, TALL_TYPES),
        (WIDE_SPECS, WIDE_TYPES),
        (TOO_WIDE_SPECS, WIDE_TYPES),
        (GREEDY_CORNER_SPECS, GREEDY_CORNER_TYPES),
    )

    def test_the_feasible_bool_and_the_feasible_verdict_agree_everywhere(self):
        for softcores, types in self.VEHICLES:
            result = pack_layout(softcores=softcores, core_types=types)
            assert result.feasible is (result.verdict == VERDICT_FEASIBLE), softcores

    def test_a_failed_pack_reports_the_same_empty_census_as_before(self):
        result = pack_layout(softcores=OVERFLOW_SPECS, core_types=OVERFLOW_TYPES)
        assert (
            result.cores_used, result.total_capacity, result.used_area,
            result.unused_area_total, result.avg_unused_area_per_core,
            result.unusable_space_total, result.avg_unusable_space_per_core,
            result.used_core_snapshots, result.placements,
        ) == (0, 0, 0, 0, float("inf"), 0, 0.0, None, None)

    def test_a_feasible_pack_still_reports_its_full_census(self):
        result = pack_layout(
            softcores=FITTING_SPECS, core_types=FITTING_TYPES, collect_placements=True,
        )
        snapshots = result.used_core_snapshots or ()
        assert result.used_area == sum(sc.area for sc in FITTING_SPECS)
        assert result.used_area == sum(snap.used_area for snap in snapshots)
        assert result.cores_used == len(snapshots)
        assert len(result.placements or ()) == len(FITTING_SPECS)

    def test_the_verdict_field_defaults_to_feasible(self):
        result = LayoutPackingResult(
            feasible=True, cores_used=1, total_capacity=64, used_area=16,
            unused_area_total=48, avg_unused_area_per_core=48.0,
        )
        assert result.verdict == VERDICT_FEASIBLE
