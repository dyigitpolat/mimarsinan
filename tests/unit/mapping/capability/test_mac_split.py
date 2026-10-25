"""The MAC fields a level row states: both counters, the cores' MAC floor, and a named reason per withheld field.

A MAC number the counters cannot back is absent with its reason, never wrong (plan §11.1): an estimator total
below the cores' MAC floor, two host counters that disagree, and a product the MAC SSOT counts 0.
"""

from __future__ import annotations

from mimarsinan.mapping.capability.census import ladder_census, level_census
from mimarsinan.mapping.capability.ladder import capability_level
from mimarsinan.mapping.capability.mac_split import (
    COUNTERS_DISAGREE,
    ESTIMATOR_BELOW_CORE_FLOOR,
    UNCOUNTED_PRODUCT,
    MacCounters,
    StatedBoundary,
    joined_reasons,
    mac_statement,
)
from mimarsinan.mapping.capability.pricing_inputs import level_quantities

from .ir_fixtures import (
    adapter_product_graph,
    biased_padded_core_graph,
    bilinear_upsample_graph,
    contraction_graph,
    convtranspose_graph,
    grouped_conv3d_graph,
    grouped_graph,
    lane_graph,
    seed_mac_gap,
)

BITS = 8
GROUPED_BOUNDARY = "grouped convolution on the host until N5"


def _row(ir, level):
    return level_census(ir, capability_level(level), value_bits=BITS)


def _counters(host=0, total=10**6, *, estimator_host=None, multi=False):
    return MacCounters(ir_host_macs=host, estimator_host_macs=host if estimator_host is None else estimator_host,
                       estimator_total_macs=total, has_multi_input_op=multi)


def test_the_core_mac_floor_counts_only_cells_fed_by_a_value():
    row = _row(biased_padded_core_graph(), "C0")
    assert row.cells_committed == 6 * 3, "four inputs, one padding axon, one bias axon"
    assert row.core_mac_floor == 4 * 3, "a bias row and a padding axon carry no MAC"
    lane = _row(lane_graph(), "C0")
    assert lane.core_mac_floor == lane.cells_committed == 4 * 4 + 4 * 4 + 4 * 4 + 4 * 2


def test_an_estimator_total_below_the_core_floor_is_withheld_with_its_reason():
    row = _row(lane_graph(), "C0")
    statement = mac_statement(row, _counters(total=55))
    assert statement.total_macs is None and statement.onchip_macs is None
    assert statement.host_macs == 0, "count_host_ops stays stated: the two host counters agree"
    assert statement.total_withheld == ESTIMATOR_BELOW_CORE_FLOOR == "estimator_conv_one_position"
    assert statement.withheld["onchip_macs"] == statement.withheld["total_macs"] == ESTIMATOR_BELOW_CORE_FLOOR
    quantities = level_quantities(row, counters=_counters(total=55))
    assert "onchip_macs" not in quantities and quantities["host_macs"] == 0
    assert mac_statement(row, _counters(total=56)).onchip_macs == 56, "a total at the floor is stated"


def test_the_floor_is_compared_with_the_estimators_own_on_chip_share():
    row = _row(lane_graph(), "C0")
    hidden = _counters(host=0, estimator_host=50, total=100, multi=True)
    assert hidden.estimator_total_macs >= row.core_mac_floor, "the total alone would pass"
    assert mac_statement(row, hidden).total_withheld == ESTIMATOR_BELOW_CORE_FLOOR, "its on-chip 50 is below 56"


def test_a_stated_total_gives_on_chip_macs_net_of_host_macs():
    ir = grouped_graph()
    counters = _counters(host=8, total=200)
    c0, c2 = _row(ir, "C0"), _row(ir, "C2")
    assert c0.core_mac_floor == 80
    q0, q2 = level_quantities(c0, counters=counters), level_quantities(c2, counters=counters)
    assert (q0["host_macs"], q0["onchip_macs"]) == (8, 192)
    assert (q2["host_macs"], q2["onchip_macs"]) == (0, 200)
    s0 = mac_statement(c0, counters)
    assert s0.withheld == {} and s0.total_macs == 200 and s0.total_withheld is None
    assert [(b.boundary, b.host_macs, b.withheld) for b in s0.stated_boundaries] == [(GROUPED_BOUNDARY, 8, None)]
    assert mac_statement(c2, counters).stated_boundaries == ()


def test_disagreeing_counters_withhold_host_macs_on_every_row_with_the_named_disagreement():
    """CL4: since G1's input_shapes rule the two host counters agree on every row, a multi-input op's included, so a
    disagreement is unexplained wherever it occurs: the same named reason with or without a multi-input op."""
    counters = _counters(host=0, estimator_host=40, total=500, multi=True)
    assert counters.reason == COUNTERS_DISAGREE and _counters(multi=True).reason is None
    for row in ladder_census(lane_graph(), value_bits=BITS):
        statement = mac_statement(row, counters)
        assert statement.host_macs is None and statement.onchip_macs is None, row.level
        assert statement.withheld["host_macs"] == COUNTERS_DISAGREE
        assert COUNTERS_DISAGREE in statement.withheld["onchip_macs"]
        assert statement.total_macs == 500, "the estimator's own total is still backed by the core floor"
        quantities = level_quantities(row, counters=counters)
        assert "host_macs" not in quantities and "onchip_macs" not in quantities
    unexplained = mac_statement(_row(lane_graph(), "C0"), _counters(host=0, estimator_host=40, total=500))
    assert unexplained.withheld["host_macs"] == COUNTERS_DISAGREE


def test_a_product_the_mac_ssot_counts_zero_is_withheld_on_the_side_that_runs_it():
    ir = bilinear_upsample_graph()
    counters = _counters(total=10**6)
    c1, c2 = _row(ir, "C1"), _row(ir, "C2")
    assert c1.uncounted_host_products == {"mms": ("Upsample",)} and c1.uncounted_chip_products == {}
    assert c2.uncounted_host_products == {} and c2.uncounted_chip_products == {"mms": ("Upsample",)}
    s1, s2 = mac_statement(c1, counters), mac_statement(c2, counters)
    assert s1.host_macs is None and s1.withheld["host_macs"] == f"{UNCOUNTED_PRODUCT}: Upsample"
    assert s1.synaptic_events == c1.cells_committed
    assert s2.host_macs == 0 and s2.synaptic_events is None
    assert s2.withheld["synaptic_events"] == f"{UNCOUNTED_PRODUCT}: Upsample"
    assert "synaptic_events" not in level_quantities(c2) and "synaptic_events" in level_quantities(c1)


def test_a_product_with_a_mac_rule_is_counted_on_either_side():
    for level in ("C0", "C2"):
        row = _row(grouped_graph(), level)
        assert row.uncounted_host_products == {} == row.uncounted_chip_products
    ir = convtranspose_graph()
    c1, c2 = _row(ir, "C1"), _row(ir, "C2")
    assert c1.uncounted_host_products == {} == c2.uncounted_chip_products, "the SSOT's scatter-form rule"
    assert (c1.host_macs, c2.host_macs, c2.onchip_op_macs) == (64, 0, 64), "8 input elements x 2 x 2 x 2"
    assert mac_statement(c2, _counters(total=10**6)).synaptic_events == c2.cells_committed + 64


def test_a_withheld_total_keeps_the_stated_boundary_host_macs():
    statement = mac_statement(_row(grouped_graph(), "C0"), _counters(host=8, total=10))
    assert statement.total_withheld == ESTIMATOR_BELOW_CORE_FLOOR
    (boundary,) = statement.stated_boundaries
    assert boundary.host_macs == 8, "the grouped conv's own count stands"
    assert boundary.withheld is None and statement.total_macs is None, "only its share needs the total"


def test_a_stated_boundary_whose_product_the_mac_ssot_counts_zero_withholds_its_host_macs(monkeypatch):
    counted = _row(grouped_conv3d_graph(), "C0")
    assert counted.uncounted_host_products == {} and counted.stated_boundaries == {GROUPED_BOUNDARY: 8}, (
        "the SSOT counts a grouped Conv3d at every output position")
    seed_mac_gap(monkeypatch, "Conv3d")
    row = _row(grouped_conv3d_graph(), "C0")
    assert row.uncounted_host_products == {"mmg": ("Conv3d",)} and row.stated_boundaries == {GROUPED_BOUNDARY: 0}
    statement = mac_statement(row, _counters(total=200))
    assert statement.total_withheld is None, "a total at or above the core floor 80 is backed"
    assert statement.stated_boundaries == (StatedBoundary(GROUPED_BOUNDARY, None, f"{UNCOUNTED_PRODUCT}: Conv3d"),)
    assert mac_statement(_row(grouped_conv3d_graph(), "C2"), _counters(total=200)).stated_boundaries == ()


def test_withheld_host_macs_name_both_the_counter_disagreement_and_the_uncounted_product(monkeypatch):
    """Host MACs withheld for two reasons name both, the counters' first (a row reading "mac_counters_disagree;
    uncounted_product: <op type>"); naming only the first would drop the uncounted product."""
    seed_mac_gap(monkeypatch, "Conv3d")
    row = _row(grouped_conv3d_graph(), "C0")
    statement = mac_statement(row, MacCounters(0, 8, 10**6, True))
    assert statement.withheld["host_macs"] == f"{COUNTERS_DISAGREE}; {UNCOUNTED_PRODUCT}: Conv3d"
    assert statement.host_macs is None and statement.onchip_macs is None


def test_the_dynamic_product_is_counted_by_its_declared_macs():
    """G1's ``forward_macs()`` rule (plan §5.1): the MAC SSOT counts a Contraction by its declaration, m x k x n."""
    ir = contraction_graph()
    c0, c3 = _row(ir, "C0"), _row(ir, "C3")
    assert c0.uncounted_host_products == {} and c3.uncounted_chip_products == {}
    assert (c0.host_macs, c3.host_macs, c3.onchip_op_macs) == (2 * 4 * 3, 0, 2 * 4 * 3)


def test_a_dynamic_product_without_a_declared_count_is_withheld_on_either_side(monkeypatch):
    """A product of two activations the SSOT has no rule for is named. [integration of CL4 with Swin W10] The SSOT
    counts a generic host op of ``torch.matmul`` since W10 (a product in functional form), so the gap is seeded
    (``seed_mac_gap``): the census's reaction to a gap is the behaviour under test, never which ops have one."""
    ir = adapter_product_graph()
    matmul = str(ir.nodes[2].op_type)
    assert _row(ir, "C0").uncounted_host_products == {} and _row(ir, "C0").host_macs == 2 * 4 * 3   # W10's rule
    seed_mac_gap(monkeypatch, matmul)
    assert _row(ir, "C0").uncounted_host_products == {"dmm": (matmul,)}
    assert _row(ir, "C3").uncounted_chip_products == {"dmm": (matmul,)}
    assert mac_statement(_row(ir, "C3"), None).withheld["synaptic_events"] == f"{UNCOUNTED_PRODUCT}: {matmul}"


def test_the_per_kind_host_macs_leave_out_a_kind_with_an_uncounted_product():
    c1 = mac_statement(_row(bilinear_upsample_graph(), "C1"), None)
    assert c1.host_macs_by_kind == {} and c1.withheld["host_macs_by_kind"] == f"{UNCOUNTED_PRODUCT}: Upsample"
    assert mac_statement(_row(convtranspose_graph(), "C1"), None).host_macs_by_kind == {"mms": 64}
    lane = mac_statement(_row(lane_graph(), "C0"), None)
    assert lane.host_macs_by_kind == {"clamp": 0, "affine": 0, "add": 0, "smooth": 0}
    assert "host_macs_by_kind" not in lane.withheld
    assert mac_statement(_row(grouped_graph(), "C0"), _counters(host=8)).host_macs_by_kind == {"mmg": 8}


def test_a_field_derived_from_withheld_inputs_names_each_reason_in_order():
    assert joined_reasons() is None and joined_reasons(None, "") is None
    assert joined_reasons("host gap", None, "total gap") == "host gap; total gap"
    assert joined_reasons("total gap", "host gap") == "total gap; host gap", "the caller's order, numerator first"
