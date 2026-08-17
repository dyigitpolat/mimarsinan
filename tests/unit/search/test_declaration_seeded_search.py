"""[R6] Generation 1 carries the declaration — the known-feasible point.

The ViT cell measured the failure mode this closes: 4,728 softcores whose
packing is a shape needle, 72/72 random offspring infeasible, search dead.
The declared platform deploys in fixed mode, so it is BY CONSTRUCTION a
feasible, in-bounds candidate — a search that never visits it is throwing
away the one thing the operator proved.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.search.optimizers.nsga2_optimizer import _seeded_sampling

from unit.search.test_candidate_fragments_live_path import (
    _cfg,
    _problem,
)


class TestTheSeedEncodesTheDeclaration:
    def test_decode_of_the_seed_is_the_declared_chip(self):
        problem = _problem(_cfg(), ["total_param_capacity"])
        seeds = problem.seed_vectors()
        assert len(seeds) == 1
        decoded = problem.decode(seeds[0])
        declared = problem.fixed_platform_constraints["cores"]
        candidate = decoded["platform_constraints"]["cores"]
        assert [(c["max_axons"], c["max_neurons"], c["count"])
                for c in candidate] == [
            (c["max_axons"], c["max_neurons"], int(c["count"]))
            for c in declared
        ]

    def test_the_seed_evaluates_feasibly(self):
        """The whole point: individual 0 must never be a penalty row."""
        problem = _problem(_cfg(), ["total_param_capacity"])
        result = problem.validate_detailed(problem.decode(problem.seed_vectors()[0]))
        assert result.is_valid

    def test_a_mismatched_type_count_declines_to_seed(self):
        """One searched type against a two-type declaration: a reshaped seed
        would not BE the declaration, so there is none — stated, not silent."""
        cfg = _cfg(cores=[
            {"max_axons": 256, "max_neurons": 256, "count": 32},
            {"max_axons": 128, "max_neurons": 512, "count": 32},
        ])
        problem = _problem(cfg, ["total_param_capacity"], num_core_types=1)
        assert problem.seed_vectors() == []


class TestTheSamplingInjectsTheSeed:
    def test_row_zero_is_the_seed(self):
        seed = np.array([5.0, 7.0, 9.0])

        class _P:
            n_var = 3
            xl = np.zeros(3)
            xu = np.full(3, 10.0)

            def has_bounds(self):
                return True

            def bounds(self):
                return self.xl, self.xu

        X = _seeded_sampling([seed])._do(
            _P(), 4, random_state=np.random.default_rng(0))
        assert np.allclose(X[0], seed)
        assert X.shape == (4, 3)
        assert not np.allclose(X[1], seed)
