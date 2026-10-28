"""The theorem box's hypothesis table (the in-tree fixture the box and its tables are generated from)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from mimarsinan.certification.randomized.arms import ARMS, HYPOTHESIS_ARMS
from mimarsinan.certification.randomized.cost_draws import cost_only_set
from mimarsinan.certification.randomized.screen import FIXED_AXES

ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "tests" / "fixtures" / "theorem_box" / "hypothesis_table.json"
#: Changing the fixture changes the box: re-pin deliberately (and regenerate RA/10_wave1/thm/).
#: N2 (Wave 2) re-pin: ``cores.operand.write_source`` joins the cost-only list (pricing-only, X3a ``PRICING_ONLY``);
#: N2's fix round adds the finding that states its condition and the stationary choice's (``findings_on_classes``).
#: N2b re-pin: ``cores.write_physics`` joins the cost-only list (pricing-only, X3a ``PRICING_ONLY``).
#: Integration re-pin (tcad/wave1-merged, N2 onto the closure and Swin): the closure's ``cores.weight_sign_granularity``
#: and finding ``per_axon_sign_split`` beside N2's two entries and N2b's one, 29 cost-only rows.
#: Integration re-pin (tcad/wave1-merged, pack onto N2): W02's ``position_multiplexing`` joins the cost-only list
#: (X3a ``DEFERRED``, its ``cost_only_hypotheses`` row residency_plan's), 30 cost-only rows.
#: Integration re-pin (tcad/wave1-merged, lanes onto pack): W08's two lane-block pricing-only fields
#: (``onchip_arithmetic.lane_block``, ``onchip_arithmetic.lanes_per_core``) join the cost-only list, 32 rows.
#: CL9 final re-pin (the six Wave 2 units on tcad/wave1-merged): the fixture is, byte for byte, the three-way JSON union
#: of the units' own fixtures over 75998141's (closure, N2 with N2b, pack, lanes; Swin and D19 leave it unchanged), with
#: no conflict (RA/11_wave2/integration/cl9/scripts/fixture_union.py); RA/10_wave1/thm/ is regenerated from it.
#: Integration re-pin (tcad/wave1-merged, E1 onto the exactness unit): E1's R4 row (gap, memo, sites on tree E1,
#: witnesses, code-level text) and its ``position_dependent_encode`` status, folded into the CL9 fixture by the same
#: three-way JSON union with no conflict (RA/11_wave2/integration2/e1/repin/); RA/10_wave1/thm/ regenerated from it.
#: CL9 final re-pin of integration pass 2 (explore, exact, E1, E6, P1): the fixture is, byte for byte, the three-way
#: JSON union of the pass-1 CL9 fixture and E1's over the closure tree's, with no conflict; the exploration, exactness,
#: E6 and P1 units leave it unchanged (RA/11_wave2/integration2/cl9/repin/); RA/10_wave1/thm/ is regenerated from it.
#: E-6 re-pin (U-E, N2 follow-up F5): ``cores.compute_physics`` joins the cost-only list (pricing-only, X3a
#: ``PRICING_ONLY``), 33 cost-only rows; RA/10_wave1/thm/ is regenerated from the fixture at integration.
#: U-S re-pin (the event census, w2-us on fd006ea6): ``event_census_samples`` joins the cost-only list (pricing-only,
#: X3a ``PRICING_ONLY``: a measurement of the built program the record prices), 33 cost-only rows.
#: Integration re-pin (tcad/wave1-merged, us2 onto E-6): the fixture is the union of the two re-pins (both keys), merged
#: with no conflict and equal byte for byte to U-S's union table (RA/11_wave2/final_units/us/integration/
#: hypothesis_table.union_us_ue.json), 34 cost-only rows; RA/10_wave1/thm/ is regenerated from the fixture.
TABLE_SHA256 = "09537a21349b8b49dcd5eb73028eeb1cebba81249a6fd1761e41974627cbe1b9"
#: The twelve arms of D17 (RA/04_experiment_portfolio.md:580, :1304).
D17 = ("H1-theta", "H1-latency", "H2", "H3", "H4", "H5-collapse", "H5-skew", "H6", "H7", "H8", "H9", "H10")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


def test_the_fixture_is_the_pinned_one():
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == TABLE_SHA256


def test_the_class_list_is_the_registry_cost_only_set(table):
    assert set(table["cost_only"]) == cost_only_set()
    assert len(table["cost_only"]) == len(set(table["cost_only"])) == 34


#: The proof check's split (PC1): conditions on the two deployments, and conditions on every run (the
#: identity program's included); Proposition 2 alone adds the data-dependent H2-H4 and H8's staircase clause.
PROGRAM = {"H1-theta", "H1-latency", "H5-collapse", "H5-skew", "H6", "H10", "R1", "R2", "R3", "R4", "R5"}
EXECUTOR = {"H0", "H7", "H8", "H9", "R6"}
NETWORK = {"H2", "H3", "H4", "H8-staircase"}


def test_the_hypotheses_are_h0_d17s_twelve_and_the_pass_projection_rules(table):
    """H0, D17's arms (H8's staircase clause beside it), then G5's pass-projection rules R1-R6."""
    ids = [h["id"] for h in table["hypotheses"]]
    at = D17.index("H8") + 1
    assert ids == ["H0", *D17[:at], "H8-staircase", *D17[at:], "R1", "R2", "R6", "R3", "R4", "R5"]


def test_the_conditions_split_into_program_executor_and_network_ones(table):
    kinds = {h["id"]: h["kind"] for h in table["hypotheses"]}
    assert {k for k, v in kinds.items() if v == "program"} == PROGRAM
    assert {k for k, v in kinds.items() if v == "executor"} == EXECUTOR
    assert {k for k, v in kinds.items() if v == "network"} == NETWORK


@pytest.mark.parametrize("field", ["id", "name", "kind", "relation", "semantics", "statement", "memo", "sites",
                                   "witness", "arms"])
def test_every_hypothesis_states_every_field(table, field):
    for h in table["hypotheses"]:
        assert h.get(field), (h["id"], field)


def test_every_test_node_witness_names_a_test_in_the_tree(table):
    for h in table["hypotheses"]:
        for witness in h["witness"]:
            if not witness.startswith("tests/"):
                continue
            path, _, node = witness.partition("::")
            assert (ROOT / path).is_file(), (h["id"], witness)
            name = node.split("::")[-1].split("[")[0]
            assert f"def {name}(" in (ROOT / path).read_text(), (h["id"], witness)


def _row(table, hid):
    return next(h for h in table["hypotheses"] if h["id"] == hid)


def test_theorem_1_takes_only_h8s_accumulation_clause(table):
    """Review round 2: a staircase is a deterministic function of the same integer charge in every deployment,
    so decision invariance needs exact accumulation only; the staircase and saturation clauses bind Proposition 2
    alone (the base's synchronized executors violated the staircase clause; met on the CL1 fix round's tree)."""
    h8, stair = _row(table, "H8"), _row(table, "H8-staircase")
    assert (h8["kind"], h8["relation"]) == ("executor", "both")
    assert "staircase" not in h8["statement"] and "saturat" not in h8["statement"]
    # Re-check PC27: the TTFS level sum is not integer arithmetic; the per-neuron 1/(2S) lattice recovers it.
    assert "1/(2S) lattice" in h8["statement"] and "per neuron" in h8["statement"]
    assert any(s["site"].endswith("SegmentTtfsArrays.integral_neurons") for s in h8["sites"])
    assert any(s["site"].endswith("membrane_lattice:lattice_membrane") for s in h8["sites"])
    assert (stair["kind"], stair["relation"]) == ("network", "network")
    assert "integer charge" in stair["statement"] and "saturat" in stair["statement"]
    assert all("staircase" not in w for w in h8["witness"])
    assert any("exact_tie" in w for w in stair["witness"])
    defect = next(d for d in table["defects"] if d["id"] == "synchronized_staircase")
    assert defect["hypothesis"] == "H8-staircase"
    assert not any("H8-staircase" in labels for labels in table["cost_only_hypotheses"].values())


def test_h8_states_the_synchronized_count_rebuild_and_its_fix(table):
    """Targeted pass: the base's synchronized LIF executor rebuilt each core's count as stair*T/theta in float64
    (sync_counts.py:56-59), exact on the screen's declared range (T in 1-8, theta in 1-16) and off the integer
    lattice beyond it, so a pass cut changed outputs where R6 holds. The CL1 fix round (sign-off P1 (a) item 1)
    reads the count off the window charge; H8 states the base class and the fix with their witnesses, the defect
    row names its owner and status, and every key that moves a pass cut rests on H8."""
    h8 = _row(table, "H8")
    assert "sync_counts.py:56-59" in h8["gap"] and "count_rebuild" in h8["gap"] and "T in 1-8" in h8["gap"]
    assert "lif_window_count" in h8["gap"]
    assert any("TestTheCountRebuild" in w for w in h8["witness"])
    defect = next(d for d in table["defects"] if d["id"] == "count_rebuild")
    assert defect["hypothesis"] == "H8" and "sync_counts.py:56-59" in defect["statement"]
    assert "P1 (a)" in defect["owner"] and defect["probe_arm"] == "on_long_window"
    assert defect["status"].startswith("fixed on the integrated tree")
    # Proposition 2's third base cause on synchronized LIF: a rebuilt count off N with no staircase tie involved.
    assert "count_rebuild" in _row(table, "H8-staircase")["gap"]
    for key, labels in table["cost_only_hypotheses"].items():
        if "H5-collapse" in labels:
            assert "H8" in labels, key


def test_r4_is_g5s_per_slice_input_kinds_with_its_switching_arm(table):
    """G5's rule 4: a carried slice is replayed and every other slice encoded as the fused stage encodes it, in
    its fused channel positions; the base violated it (crash class mixed_input_pass, silent class
    position_dependent_encode under a dithered encode), CL1 met the per-slice kinds and E1 the rotation (keyed on each
    channel at unbounded geometry), so the R4 arm switches that key off with a paired control."""
    r4 = _row(table, "R4")
    assert "carried slice is replayed" in r4["statement"] and "fused channel positions" in r4["statement"]
    assert r4["arms"]["code"] == "R4" and ARMS["R4"].changes_a_field and not ARMS["R4"].mechanism
    assert "mixed_input_pass" in r4["gap"] and "position_dependent_encode" in r4["gap"] and "E1" in r4["gap"]
    assert any(s["tree"] == "E1" and "phase_channels" in s["site"] for s in r4["sites"])
    assert "mixed_input_pass" not in _row(table, "R5")["gap"]


def test_the_per_event_fusion_refusal_is_a_hypothesis_with_its_named_sites(table):
    """H7 requires the refusal; the golden refuses fusion, the FPGA branch's guard refuses co-residency and
    splitting under per-event cores, and its refusal of a fused fan-in is checked in Wave 3."""
    h7 = _row(table, "H7")
    assert "requires" in h7["statement"] and "Wave 3" in h7["statement"]
    refusals = {(s["site"], s["tree"]) for s in h7["sites"] if s["kind"] == "refusal"}
    assert ("x3a_golden.packer:fragments", "golden") in refusals
    assert any(tree == "FPGA" and "guards" in site for site, tree in refusals)


def test_h1_latency_is_the_identity_program_latency_with_no_cycle_check_site(table):
    h = next(h for h in table["hypotheses"] if h["id"] == "H1-latency")
    assert "identity-program" in h["statement"]
    assert not any("_compute_all_delays" in s["site"] for s in h["sites"])
    assert any(s["tree"] == "CL1" for s in h["sites"]) and "executed-latency" in h["gap"]


def _base_sites(table, hid):
    return {s["site"] for s in _row(table, hid)["sites"] if s["tree"] == "base"}


def test_cl1_states_the_integrated_trees_sites_and_retires_what_it_asserts(table):
    """CL1: H1-latency is asserted at run time by the pass projection's refusal (no code arm, the reason stated);
    H5-skew and R6 are met by G5's projection and the dependency-ordered walks, which their arms switch off; the
    value path's H8 bound is the law solver's refusal; the code level's value law is G2's readout law."""
    h1 = _row(table, "H1-latency")
    assert "mimarsinan.mapping.packing.pass_timing:_core_latencies" in _base_sites(table, "H1-latency")
    assert h1["arms"]["code"] is None and "asserted at run time" in h1["code_level"]
    assert not ARMS["H1-latency"].exercisable
    assert "mimarsinan.mapping.packing.pass_timing:project_passes" in _base_sites(table, "H5-skew")
    assert {"mimarsinan.chip_simulation.value_run.value_execution:dependency_order",
            "mimarsinan.chip_simulation.ttfs.ttfs_segment:dependency_order"} <= _base_sites(table, "R6")
    for hid in ("H5-skew", "R6"):
        assert ARMS[_row(table, hid)["arms"]["code"]].changes_a_field, hid
        assert _row(table, hid)["gap"].startswith("met on the integrated tree (CL1)"), hid
    assert "mimarsinan.mapping.export.law_solver:solve_core_law" in _base_sites(table, "H8")
    assert table["statement"]["fixed_at_code_level"]["value_law"] == FIXED_AXES["value_law"]
    assert "integer readout law" in table["statement"]["relations"]["network"]
    fixed = {d["id"] for d in table["defects"] if d.get("status", "").startswith("fixed on the integrated tree")}
    assert fixed == {"latency_order_walk", "mixed_input_pass", "synchronized_cached_trains", "retiming_dropped",
                     "synchronized_staircase", "count_rebuild"}


def test_the_cl1_fix_round_meets_h0_and_h8_staircase_with_their_arms(table):
    """Sign-off P1 (a) items 1 to 3 at CL1: H0 is met by leveling a coalescing group as one unit, and its arm now
    switches that rule off (a paired control, no base violation); H8-staircase is met by the staircases on the
    integer charge, measured on the network relation."""
    h0 = _row(table, "H0")
    assert h0["gap"].startswith("met on the integrated tree (CL1 fix round") and h0["arms"]["code"] == "H0"
    assert ARMS["H0"].changes_a_field and not ARMS["H0"].mechanism
    assert any(s["tree"] == "CL1" and "_unit_depths" in s["site"] for s in h0["sites"])
    stair = _row(table, "H8-staircase")
    assert "Met on the integrated tree (CL1 fix round" in stair["gap"]
    assert "lif_window_count" in stair["gap"] and "ttfs_quantized_staircase_lattice_np" in stair["gap"]


def test_the_pass_projection_rules_take_their_sites_from_g5_at_cl1(table):
    for hid in ("R1", "R2", "R3", "R4", "R5", "R6"):
        h = next(h for h in table["hypotheses"] if h["id"] == hid)
        assert any(s["tree"] == "CL1" for s in h["sites"]), hid


def test_per_event_order_binds_every_slot_moving_key(table):
    for key in ("allow_coalescing", "allow_neuron_splitting", "cores.max_axons", "max_axons", "cores.count"):
        assert "H7" in table["cost_only_hypotheses"][key], key


def test_a_hypothesis_without_a_base_tree_assertion_is_a_named_gap(table):
    for h in table["hypotheses"]:
        on_base = [s for s in h["sites"] if s["tree"] == "base" and s["kind"] in ("refusal", "admission")]
        if not on_base:
            assert h.get("gap"), h["id"]


def test_a_hypothesis_without_a_code_arm_says_why(table):
    for h in table["hypotheses"]:
        if h["arms"]["code"] is None:
            assert h.get("gap") or h.get("code_level"), h["id"]


def test_the_code_arms_are_the_tables(table):
    by_label = {}
    for arm_id in HYPOTHESIS_ARMS:
        by_label.setdefault(ARMS[arm_id].hypothesis, []).append(arm_id)
    for h in table["hypotheses"]:
        code = h["arms"]["code"]
        arms = by_label.get(h["id"], [])
        if code is None:
            assert all(not ARMS[a].exercisable for a in arms), h["id"]
        else:
            assert code in arms and ARMS[code].exercisable, h["id"]
    assert set(by_label) <= {h["id"] for h in table["hypotheses"]}


def test_the_model_level_labels_are_the_tables(table):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from x3a_golden.hypotheses import LABELS
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    hyps = table["hypotheses"]
    assert tuple(h["arms"]["model"] for h in hyps if h["arms"]["model"]) == tuple(LABELS)
    for h in hyps:
        if not h["arms"]["model"]:
            assert h.get("model_level"), h["id"]


def test_the_sign_granularity_rests_on_the_packing_hypotheses_and_exact_accumulation(table):
    """CL-SG (S0 amendment, owner answer L1 = a): the per-axon split changes a soft core's rows as a narrower core
    does, so the path rests on cores.max_axons' hypotheses; its own exactness is H8's exact accumulation (W = W+ + W-
    on the integer grid) per cycle and H7's per-event order (the pair lemma) on per-event cores."""
    labels = table["cost_only_hypotheses"]["cores.weight_sign_granularity"]
    assert labels == table["cost_only_hypotheses"]["cores.max_axons"]
    assert {"H8", "H7"} <= set(labels)


def test_every_cost_only_path_names_known_hypotheses(table):
    ids = {h["id"] for h in table["hypotheses"]}
    for path, labels in table["cost_only_hypotheses"].items():
        assert path in table["cost_only"], path
        assert set(labels) <= ids, path
