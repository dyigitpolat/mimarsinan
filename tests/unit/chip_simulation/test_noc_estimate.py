"""N3 — the candidate NoC estimator mirrors the record's trace conventions.

The record side counts one message per (firing source neuron, destination
core), walks it over the mesh with the x-first XY rule, and classifies it
inter/intra/input-path (`analysis/noc.py`, `analysis/diagnostics.py`). The
estimator prices the SAME abstract traffic from the shape-only fragments:
messages(p→c) = wires(p→c) × activity × T, on-wires fire every cycle, input
and on somas live on their consumer core. The synthetic equality pin builds
one traffic pattern both ways and demands identical numbers.
"""

from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.sanafe.noc_estimate import (
    NocEstimate,
    estimate_noc,
)


def _fragments(pass_placements, pair_wires, input_wires, on_wires):
    census = SimpleNamespace(
        pair_wires=pair_wires, input_wires=input_wires, on_wires=on_wires,
    )
    return SimpleNamespace(pass_placements=pass_placements, census=census)


class TestTheTrafficModel:
    def test_cross_tile_pair_prices_inter_packets_and_hops(self):
        # cores 0,1 on tiles 0,1 of a 2x1 mesh (cpt=1): manhattan = 1.
        frags = _fragments(
            pass_placements=(((0, 0), (1, 1)),),
            pair_wires={(0, 1): 3},
            input_wires=(0, 0), on_wires=(0, 0),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=1, mesh_height=1,
            activity_factor=0.5, timesteps=4,
        )
        assert est.inter_tile_packets == pytest.approx(3 * 0.5 * 4)
        assert est.total_hops == pytest.approx(3 * 0.5 * 4)
        assert est.intra_tile_packets == 0.0
        assert est.total_packets == pytest.approx(est.inter_tile_packets)

    def test_manhattan_distance_weights_the_hops(self):
        # cpt=1, mesh_height=2: tile 0 -> (0,0), tile 3 -> (1,1): 2 hops.
        frags = _fragments(
            pass_placements=(((0, 0), (1, 3)),),
            pair_wires={(0, 1): 2},
            input_wires=(0, 0), on_wires=(0, 0),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=1, mesh_height=2,
            activity_factor=1.0, timesteps=1,
        )
        assert est.total_hops == pytest.approx(2 * 2)

    def test_same_tile_pair_is_intra_with_zero_hops(self):
        frags = _fragments(
            pass_placements=(((0, 0), (1, 1)),),
            pair_wires={(0, 1): 5},
            input_wires=(0, 0), on_wires=(0, 0),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=2, mesh_height=1,
            activity_factor=1.0, timesteps=2,
        )
        assert est.intra_tile_packets == pytest.approx(10)
        assert est.inter_tile_packets == 0.0
        assert est.total_hops == 0.0

    def test_input_wires_are_intra_input_path_traffic(self):
        """Input somas live on their consumer core (coreN_in groups)."""
        frags = _fragments(
            pass_placements=(((0, 0),),),
            pair_wires={},
            input_wires=(4,), on_wires=(0,),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=1, mesh_height=1,
            activity_factor=0.25, timesteps=8,
        )
        assert est.input_path_packets == pytest.approx(4 * 0.25 * 8)
        assert est.intra_tile_packets == pytest.approx(4 * 0.25 * 8)
        assert est.total_hops == 0.0

    def test_on_wires_fire_every_cycle_not_activity_scaled(self):
        frags = _fragments(
            pass_placements=(((0, 0),),),
            pair_wires={},
            input_wires=(0,), on_wires=(1,),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=1, mesh_height=1,
            activity_factor=0.1, timesteps=6,
        )
        assert est.input_path_packets == pytest.approx(6)
        assert est.intra_tile_packets == pytest.approx(6)

    def test_cross_pass_pairs_are_carry_not_mesh_traffic(self):
        frags = _fragments(
            pass_placements=(((0, 0),), ((1, 0),)),
            pair_wires={(0, 1): 7},
            input_wires=(0, 0), on_wires=(0, 0),
        )
        est = estimate_noc(
            fragments=frags, cores_per_tile=1, mesh_height=1,
            activity_factor=1.0, timesteps=3,
        )
        assert est.total_packets == 0.0
        assert est.total_hops == 0.0

    def test_undeclared_activity_refuses(self):
        frags = _fragments((((0, 0),),), {}, (0,), (0,))
        with pytest.raises(ValueError, match="activity"):
            estimate_noc(
                fragments=frags, cores_per_tile=1, mesh_height=1,
                activity_factor=0.0, timesteps=4,
            )


class TestTheRecordSideAgrees:
    """The synthetic equality pin: one abstract traffic pattern, computed by
    the estimator AND by the record's own trace aggregation, byte-equal."""

    def _synthetic_trace(self):
        # cores 0->tile0(0,0), 1->tile1(1,0); wires(0->1)=3, a=1, T=2:
        # 6 cross-tile messages. Plus 2 input-soma messages on core 1
        # (input_wires=1, a=1, T=2), local to tile 1.
        cross = [
            {"src_tile_id": 0, "dest_tile_id": 1,
             "src_x": 0, "src_y": 0, "dest_x": 1, "dest_y": 0,
             "src_neuron_group_id": "core0_lif"}
            for _ in range(6)
        ]
        local_input = [
            {"src_tile_id": 1, "dest_tile_id": 1,
             "src_x": 1, "src_y": 0, "dest_x": 1, "dest_y": 0,
             "src_neuron_group_id": "core1_in"}
            for _ in range(2)
        ]
        return [cross + local_input]

    def test_link_load_sum_equals_estimated_hops(self):
        from mimarsinan.chip_simulation.sanafe.analysis.noc import (
            _aggregate_noc_link_load,
        )

        trace = self._synthetic_trace()
        measured_hops = sum(
            link.packet_count
            for link in _aggregate_noc_link_load(trace, None)
        )
        est = estimate_noc(
            fragments=_fragments(
                pass_placements=(((0, 0), (1, 1)),),
                pair_wires={(0, 1): 3},
                input_wires=(0, 1), on_wires=(0, 0),
            ),
            cores_per_tile=1, mesh_height=1,
            activity_factor=1.0, timesteps=2,
        )
        assert est.total_hops == pytest.approx(measured_hops)

    def test_classification_matches_the_trace_summary(self):
        from mimarsinan.chip_simulation.sanafe.analysis.diagnostics import (
            _summarize_message_trace,
        )

        summary = _summarize_message_trace(self._synthetic_trace())
        est = estimate_noc(
            fragments=_fragments(
                pass_placements=(((0, 0), (1, 1)),),
                pair_wires={(0, 1): 3},
                input_wires=(0, 1), on_wires=(0, 0),
            ),
            cores_per_tile=1, mesh_height=1,
            activity_factor=1.0, timesteps=2,
        )
        assert est.inter_tile_packets == pytest.approx(
            summary["inter_tile_packets"]
        )
        assert est.intra_tile_packets == pytest.approx(
            summary["intra_tile_packets"]
        )
        assert est.input_path_packets == pytest.approx(
            summary["input_path_packets"]
        )

    def test_the_estimate_is_a_noc_estimate(self):
        est = estimate_noc(
            fragments=_fragments((((0, 0),),), {}, (0,), (0,)),
            cores_per_tile=1, mesh_height=1,
            activity_factor=1.0, timesteps=1,
        )
        assert isinstance(est, NocEstimate)
