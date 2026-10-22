"""The theorem box's hypothesis table (the in-tree fixture the box and its tables are generated from)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from mimarsinan.certification.randomized.arms import ARMS, HYPOTHESIS_ARMS
from mimarsinan.certification.randomized.cost_draws import cost_only_set

ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "tests" / "fixtures" / "theorem_box" / "hypothesis_table.json"
#: Changing the fixture changes the box: re-pin deliberately (and regenerate RA/10_wave1/thm/).
TABLE_SHA256 = "e627d9aaaf4ec4a1608d4086e0dd471781b1d42f092c23b9ebdaea429c49ced9"
#: The twelve arms of D17 (RA/04_experiment_portfolio.md:580, :1304).
D17 = ("H1-theta", "H1-latency", "H2", "H3", "H4", "H5-collapse", "H5-skew", "H6", "H7", "H8", "H9", "H10")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


def test_the_fixture_is_the_pinned_one():
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == TABLE_SHA256


def test_the_class_list_is_the_registry_cost_only_set(table):
    assert set(table["cost_only"]) == cost_only_set()
    assert len(table["cost_only"]) == len(set(table["cost_only"])) == 26


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
    alone (the base's synchronized executors violate the staircase clause on every run)."""
    h8, stair = _row(table, "H8"), _row(table, "H8-staircase")
    assert (h8["kind"], h8["relation"]) == ("executor", "both")
    assert "staircase" not in h8["statement"] and "saturat" not in h8["statement"]
    # Re-check PC27: under synchronized TTFS the level sum is not integer arithmetic; the lattice snap recovers it.
    assert "1/(2S) lattice" in h8["statement"]
    assert any(s["site"].endswith("SegmentTtfsArrays.integer_lattice") for s in h8["sites"])
    assert (stair["kind"], stair["relation"]) == ("network", "network")
    assert "integer charge" in stair["statement"] and "saturat" in stair["statement"]
    assert all("staircase" not in w for w in h8["witness"])
    assert any("exact_tie" in w for w in stair["witness"])
    defect = next(d for d in table["defects"] if d["id"] == "synchronized_staircase")
    assert defect["hypothesis"] == "H8-staircase"
    assert not any("H8-staircase" in labels for labels in table["cost_only_hypotheses"].values())


def test_h8_states_the_synchronized_count_rebuild_as_a_base_violation(table):
    """Targeted pass: the synchronized LIF executor rebuilds each core's count as stair*T/theta in float64
    (sync_counts.py:56-59), exact on the screen's declared range (T in 1-8, theta in 1-16) and off the integer
    lattice beyond it, so the base violates H8 there and a pass cut changes outputs where R6 holds. H8 states it
    as a boundary with its witness, a defect row requests its owner, and every key that moves a pass cut rests
    on H8."""
    h8 = _row(table, "H8")
    assert "sync_counts.py:56-59" in h8["gap"] and "count_rebuild" in h8["gap"] and "T in 1-8" in h8["gap"]
    assert any("TestTheCountRebuild" in w for w in h8["witness"])
    defect = next(d for d in table["defects"] if d["id"] == "count_rebuild")
    assert defect["hypothesis"] == "H8" and "sync_counts.py:56-59" in defect["statement"]
    assert "conductor" in defect["owner"] and defect["probe_arm"] == "on_long_window"
    # Proposition 2's third base cause on synchronized LIF: a rebuilt count off N with no staircase tie involved.
    assert "count_rebuild" in _row(table, "H8-staircase")["gap"]
    for key, labels in table["cost_only_hypotheses"].items():
        if "H5-collapse" in labels:
            assert "H8" in labels, key


def test_r4_is_g5s_per_slice_input_kinds_with_a_base_violation_arm(table):
    """G5's rule 4: a carried slice is replayed and every other slice encoded as the fused stage encodes it, in
    its fused channel positions; the base violates it (crash class mixed_input_pass, silent class
    position_dependent_encode under a dithered encode)."""
    r4 = _row(table, "R4")
    assert "carried slice is replayed" in r4["statement"] and "fused channel positions" in r4["statement"]
    assert r4["arms"]["code"] == "R4"
    assert "mixed_input_pass" in r4["gap"] and "position_dependent_encode" in r4["gap"]
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


def test_every_cost_only_path_names_known_hypotheses(table):
    ids = {h["id"] for h in table["hypotheses"]}
    for path, labels in table["cost_only_hypotheses"].items():
        assert path in table["cost_only"], path
        assert set(labels) <= ids, path
