"""CL5 reading columns of a ladder point and the schedule/price split of the harness.

Pure columns against hand computations (reuse per write, the two-ceiling minimum batch,
latency semantics) and the harness seam that prices one scheduled program under several
physics declarations and batch sizes without scheduling it again. Every constant here is
a test constant.
"""

from __future__ import annotations

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
import mimarsinan.deployment_record.introspection.scaling_ladder_reading as reading
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from unit.deployment_record.introspection.test_scaling_ladder import (
    CONSTANTS, READING, _facts, _network, _platform,
)

HOURS = 10.0
RATE = 2.0
SERVICE = RATE * HOURS * 3600.0  # r T, inferences over the declared life


class TestReuse:
    def test_macs_per_weight_is_onchip_macs_over_logical_cells(self):
        out = reading.reuse_columns(onchip_macs=1200, logical_cells=300, k=16,
                                    pi_by_axis={"energy": [1.0, 2.0, 4.0], "time": None})
        assert out["macs_per_weight"] == 4.0
        assert out["reuse_per_write"] == 64.0
        assert out["write_to_mac_ratio"] == {"energy": [4.0, 8.0, 16.0], "time": None}

    def test_an_absent_mac_count_is_named(self):
        out = reading.reuse_columns(onchip_macs=None, logical_cells=300, k=1, pi_by_axis={})
        assert out["macs_per_weight"] is None and "onchip_macs" in out["absent"]


class TestMinimumBatch:
    def _batch(self, **over):
        args = dict(x_fluid=3.0, x_packed=5.0, max_writes=6.0, rate_hz=RATE, lifetime_hours=HOURS,
                    endurance=(1e3, 1e4, 1e5), write_time_per_period=(0.5, 1.0, 2.0),
                    write_time_source="test", k=16)
        args.update(over)
        return reading.min_batch(**args)

    def test_the_two_ceilings_against_a_hand_computation(self):
        out = self._batch()
        # endurance ceilings at E low, nominal, high: (x - 1) r T / E and max_writes r T / E
        assert out["k_endurance_closed_fluid"] == pytest.approx([2 * SERVICE / e for e in (1e3, 1e4, 1e5)],
                                                                rel=1e-15)
        assert out["k_endurance_closed_packed"] == pytest.approx([4 * SERVICE / e for e in (1e3, 1e4, 1e5)],
                                                                 rel=1e-15)
        assert out["k_endurance_census"] == pytest.approx([6 * SERVICE / e for e in (1e3, 1e4, 1e5)], rel=1e-15)
        # write-bandwidth ceiling r t_w at t_w low, nominal, high
        assert out["k_bandwidth"] == pytest.approx([1.0, 2.0, 4.0], rel=1e-15)
        # K_min per corner: favourable pairs the high endurance with the low write time, adverse the reverse
        want = {"favourable": max(6 * SERVICE / 1e5, 1.0), "nominal": max(6 * SERVICE / 1e4, 2.0),
                "adverse": max(6 * SERVICE / 1e3, 4.0)}
        assert out["k_min_census"] == pytest.approx(want, rel=1e-15)

    def test_the_crossover_endurance_equalizes_the_two_ceilings(self):
        out = self._batch()
        for t_w, cross in zip((0.5, 1.0, 2.0), out["e_cross_census"]):
            assert cross == pytest.approx(HOURS * 3600.0 * 6.0 / t_w, rel=1e-15)
            assert 6.0 * SERVICE / cross == pytest.approx(RATE * t_w, rel=1e-12)
        assert out["e_cross_closed_packed"][1] == pytest.approx(HOURS * 3600.0 * 4.0 / 1.0, rel=1e-15)

    def test_binding_ceiling_and_verdicts_per_corner(self):
        out = self._batch(endurance=(1e9, 1e9, 1e9), k=3)
        assert out["binding"] == {"favourable": "write bandwidth", "nominal": "write bandwidth",
                                  "adverse": "write bandwidth"}
        assert out["verdict"]["bandwidth"] == "straddles"  # K 3 admits t_w 0.5 and 1.0, refuses 2.0
        assert out["verdict"]["endurance"] == "admitted"
        assert self._batch(k=10 ** 9)["verdict"]["endurance"] == "admitted"
        assert self._batch(k=1)["verdict"]["endurance"] == "refused"

    def test_a_program_that_writes_nothing_needs_no_batch(self):
        out = self._batch(max_writes=0.0, write_time_per_period=(0.0, 0.0, 0.0), x_packed=0.5, x_fluid=0.4, k=1)
        assert out["k_endurance_census"] == [0.0, 0.0, 0.0] and out["k_bandwidth"] == [0.0, 0.0, 0.0]
        assert out["k_endurance_closed_fluid"] is None and "x_fluid" in out["closed_absent"]
        assert out["binding"]["nominal"] == "neither (no weight writes per period)"
        assert out["verdict"] == {"endurance": "admitted", "bandwidth": "admitted", "both": "admitted"}

    def test_undeclared_inputs_are_named(self):
        out = self._batch(endurance=None, write_time_per_period=None)
        assert out["k_endurance_census"] is None and "endurance" in out["endurance_absent"]
        assert out["k_bandwidth"] is None and "write time" in out["bandwidth_absent"]
        assert out["verdict"]["both"] == "undetermined"


class TestLatencySemantics:
    def test_period_request_latency_throughput_and_buffer(self):
        out = reading.latency_semantics(time_per_inference=(1e-3, 2e-3, 4e-3), compute_only=(1e-4, 2e-4, 4e-4),
                                        k=8, rate_hz=4.0, carry={"peak_live_bytes": 100}, passes=3,
                                        latency_steps=10, multiplexed=True, source="test")
        assert out["time_per_inference_label"] == reading.TIME_PER_INFERENCE
        assert out["batch_period_s"] == pytest.approx([8e-3, 16e-3, 32e-3])
        assert out["request_latency_s"] == pytest.approx([2.0 + 8e-3, 2.0 + 16e-3, 2.0 + 32e-3])
        assert out["throughput_inferences_per_s"] == pytest.approx([1e3, 5e2, 2.5e2])
        assert out["carry_buffer_bytes"] == 800 and out["carry_buffer_home"] == reading.CARRY_HOME
        assert out["layer_pipelined_bound_s"] is None and "passes" in out["layer_pipelined_absent"]

    def test_a_resident_multi_pass_program_prints_the_bound_and_a_multiplexed_one_does_not(self):
        common = dict(time_per_inference=(1e-3, 2e-3, 4e-3), compute_only=(1e-3, 2e-3, 4e-3), k=1, rate_hz=None,
                      carry={}, passes=3, latency_steps=10, source="test")
        assert reading.latency_semantics(multiplexed=False, **common)["layer_pipelined_bound_s"] == pytest.approx(
            [1e-4, 2e-4, 4e-4])
        assert reading.latency_semantics(multiplexed=True, **common)["layer_pipelined_bound_s"] is None

    def test_a_resident_single_pass_program_prints_the_layer_pipelined_bound(self):
        out = reading.latency_semantics(time_per_inference=(1e-3, 2e-3, 4e-3), compute_only=(1e-3, 2e-3, 4e-3),
                                        k=1, rate_hz=None, carry={}, passes=1, latency_steps=10,
                                        multiplexed=False, source="test")
        assert out["layer_pipelined_bound_s"] == pytest.approx([1e-4, 2e-4, 4e-4])
        assert out["request_latency_s"] is None and "inference_rate_hz" in out["request_latency_absent"]
        assert out["carry_buffer_bytes"] == 0


class TestScheduleOnceAndPriceMany:
    def test_the_composition_equals_price_program(self):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        once = ladder.price_program(network, census, platform, _facts)
        scheduled = ladder.schedule_program(network, census, platform, _facts)
        split = ladder.price_scheduled(network, census, scheduled, platform)
        point = {**platform, "inference_batch_K": 4, "inference_rate_hz": RATE, "min_lifetime_hours": HOURS}
        a, b = ladder.ladder_row(network, census, once, point, READING), ladder.ladder_row(
            network, census, split, point, READING)
        for row in (a, b):
            for key in ("layout_wall_s", "point_wall_s", "peak_rss_bytes"):
                row.pop(key)
        assert a == b

    def test_a_second_physics_prices_without_scheduling_again(self, monkeypatch):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        scheduled = ladder.schedule_program(network, census, platform, _facts)
        doubled = profile_from_dict({
            "format_version": 1, "name": "ladder_test_x2", "display_name": "x2", "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {k: {"nominal": 2 * v, "evidence_kind": "estimated", "note": "test"}
                          for k, v in CONSTANTS.items()},
        })
        calls = []
        monkeypatch.setattr(ladder, "compute_mapping_stats",
                            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError))
        variant = {**platform, "platform_physics_resolved": doubled.to_dict()}
        first = ladder.price_scheduled(network, census, scheduled, platform)
        second = ladder.price_scheduled(network, census, scheduled, variant)
        assert not calls
        assert second.passes == first.passes and second.period_bytes == first.period_bytes
        e1 = first.terms["program"]["energy_programming_mj"].nominal
        e2 = second.terms["program"]["energy_programming_mj"].nominal
        assert e2 == pytest.approx(2 * e1, rel=1e-12)

    def test_a_platform_differing_in_more_than_physics_and_the_row_keys_is_refused(self):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        scheduled = ladder.schedule_program(network, census, platform, _facts)
        with pytest.raises(ladder.LadderDeclarationError) as refused:
            ladder.price_scheduled(network, census, scheduled, _platform(9))
        assert isinstance(refused.value, ToolchainRefusal)

    def test_a_declared_batch_prices_the_write_law_at_that_batch(self):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        scheduled = ladder.schedule_program(network, census, platform, _facts)
        rows = {}
        for k in (1, 4):
            point = {**platform, "inference_batch_K": k, "inference_rate_hz": RATE, "min_lifetime_hours": HOURS}
            priced = ladder.price_scheduled(network, census, scheduled, point)
            rows[k] = ladder.ladder_row(network, census, priced, point, READING)
        law1, law4 = rows[1]["write_law"], rows[4]["write_law"]
        assert law1["inference_batch_K"] == 1 and law4["inference_batch_K"] == 4
        # the write transfer energy per inference falls as 1/K (e_dma_per_byte x period bytes / K)
        e1, e4 = law1["terms"]["energy_write_mj"], law4["terms"]["energy_write_mj"]
        assert e1 is not None and e4[1] == pytest.approx(e1[1] / 4, rel=1e-12)
        assert rows[4]["reuse"]["reuse_per_write"] == pytest.approx(4 * rows[4]["reuse"]["macs_per_weight"])
        assert rows[4]["latency_semantics"]["time_per_inference_label"] == reading.TIME_PER_INFERENCE
        assert "min_batch" in rows[4]

    def test_the_headline_at_a_batch_is_the_sum_of_its_named_energy_parts(self):
        # [CL7] the write law's energy headline at K carries its parts: dynamic, static power over the priced
        # latency (write time included), host and writes, so a reader splits it without subtracting
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        scheduled = ladder.schedule_program(network, census, platform, _facts)
        static = profile_from_dict({
            "format_version": 1, "name": "ladder_test_static", "display_name": "static", "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {**{k: {"nominal": v, "evidence_kind": "estimated", "note": "test"}
                             for k, v in CONSTANTS.items()},
                          "p_static_global": {"nominal": 0.5, "evidence_kind": "estimated", "note": "test"}},
        })
        point = {**platform, "platform_physics_resolved": static.to_dict(), "inference_batch_K": 1,
                 "inference_rate_hz": RATE, "min_lifetime_hours": HOURS}
        row = ladder.ladder_row(network, census, ladder.price_scheduled(network, census, scheduled, point), point,
                                READING)
        terms = row["write_law"]["terms"]
        assert terms["energy_per_inference_mj"] is not None and terms["energy_static_mj"][1] > 0.0
        assert terms["energy_dynamic_mj"] is not None and terms["energy_write_mj"] is not None
        # a part the pricer leaves unpriced (here the host: no host constant declared) adds nothing to the headline
        parts = [terms[name] or [0.0, 0.0, 0.0] for name in reading.ENERGY_HEADLINE_PARTS]
        assert terms["energy_per_inference_mj"] == pytest.approx([sum(c) for c in zip(*parts)], rel=1e-12)
