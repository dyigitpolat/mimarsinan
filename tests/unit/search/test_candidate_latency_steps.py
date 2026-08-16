"""[E1] The candidate sizes the wall the runner actually runs.

The retired formula was ``timesteps x neural_segment_count``. Measured on the
sealed MLP study run it said 4 where the record's ``compute_steps`` was 15:
the program retimes into 3 depth-level stages and each runs ``T + 1`` (the
input-delivery cycle). Every term that multiplies latency — static energy,
e2e, throughput — inherited the 3.75x shortfall.
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.stage_timesteps import program_latency_steps
from mimarsinan.mapping.noc import execution_stage_latencies

from unit.search.test_candidate_fragments_live_path import (
    _candidate,
    _cfg,
    _physics_cfg,
    _problem,
)


class TestTheCandidateWall:
    @pytest.mark.parametrize("retimed", [False, True])
    def test_it_equals_the_shared_rule_over_its_own_stage_structure(self, retimed):
        """No second formula: the quantity IS the SSOT applied to the
        candidate's own pass/level structure, under EITHER retiming arm."""
        problem = _problem(
            _physics_cfg(), ["e2e_latency_s"], per_hop_retiming=retimed)
        layout = problem.candidate_layout(_candidate(problem))
        assert problem.stage_semantics.retimed is retimed
        expected = program_latency_steps(
            stage_max_latencies=execution_stage_latencies(
                layout.softcores, layout.noc.pass_placements,
                retimed=retimed,  # the ARM this run declared, not the value
            ),                    # under test — else the pin checks itself
            timesteps=int(layout.platform["simulation_steps"]),
            is_cycle=False, is_cascade=False,
        )
        assert layout.view.quantities.get("latency_steps").value == expected

    def test_the_wall_exceeds_the_retired_formula(self):
        """The retired T x segments ignored BOTH the depth levels and the
        input-delivery cycle, so the honest wall is strictly larger here."""
        problem = _problem(_physics_cfg(), ["e2e_latency_s"])
        layout = problem.candidate_layout(_candidate(problem))
        retired = (int(layout.platform["simulation_steps"])
                   * int(layout.stats.neural_segment_count))
        assert layout.view.quantities.get("latency_steps").value > retired

    def test_one_stage_per_level_each_at_T_plus_one(self):
        """The shape the sealed study run measured: levels x (T+1). Retiming
        is armed there by the exact-QAT recipe pairing (lif_exact_qat=True ->
        lif_per_hop_retiming=True in the RESOLVED config), so the candidate is
        told the same, and each re-based level runs T + 0 + 1."""
        problem = _problem(
            _physics_cfg(), ["e2e_latency_s"], per_hop_retiming=True)
        layout = problem.candidate_layout(_candidate(problem))
        stages = execution_stage_latencies(
            layout.softcores, layout.noc.pass_placements, retimed=True,
        )
        T = int(layout.platform["simulation_steps"])
        assert layout.view.quantities.get("latency_steps").value == len(stages) * (T + 1)


class TestAbsenceRatherThanAWrongWall:
    def test_a_layoutless_candidate_claims_no_wall(self):
        """No layout, no pass structure, no wall — and the latency-bearing
        axes then refuse BY NAME instead of pricing a short one."""
        from mimarsinan.deployment_record.objectives import candidate_probe_without

        probe = candidate_probe_without("layout")
        assert not probe.quantities.has("latency_steps")

    def test_the_latency_axis_refuses_when_the_wall_is_absent(self):
        from mimarsinan.deployment_record.objectives import (
            OBJECTIVES,
            candidate_probe_without,
        )

        probe = candidate_probe_without("layout")
        spec = OBJECTIVES.get("e2e_latency_s")
        assert not spec.available(probe)
        with pytest.raises(ValueError, match="e2e_latency_s"):
            spec.value(probe)


class TestTheRetimingSemanticsAreLoadBearing:
    def test_the_two_arms_price_different_walls_on_the_SAME_candidate(self):
        """The arming is not cosmetic: one candidate, two recipes, two walls.
        A run that ignored ``per_hop_retiming`` and always re-timed (or never
        did) would price this model's wall 2.8x off in one direction."""
        walls = {}
        for arm in (False, True):
            problem = _problem(
                _physics_cfg(), ["e2e_latency_s"], per_hop_retiming=arm)
            layout = problem.candidate_layout(_candidate(problem))
            walls[arm] = layout.view.quantities.get("latency_steps").value
        assert walls[True] > walls[False]


    def test_retimed_and_fused_size_different_walls(self):
        """A retimed program runs one window PER LEVEL; a fused one runs a
        single window spanning the levels. Conflating them mis-sizes the wall."""
        softcores_levels = [0, 1, 2]
        placements = (tuple((i, i) for i in range(3)),)

        class _Spec:
            def __init__(self, tag):
                self.latency_tag = tag
                self.segment_id = 0

        specs = [_Spec(t) for t in softcores_levels]
        retimed = execution_stage_latencies(specs, placements, retimed=True)
        fused = execution_stage_latencies(specs, placements, retimed=False)
        assert retimed == (0, 0, 0)
        assert fused == (2,)
        assert program_latency_steps(
            stage_max_latencies=retimed, timesteps=4,
            is_cycle=False, is_cascade=False) == 15
        assert program_latency_steps(
            stage_max_latencies=fused, timesteps=4,
            is_cycle=False, is_cascade=False) == 7
