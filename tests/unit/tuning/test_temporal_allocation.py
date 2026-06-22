"""EW1 — per-layer-S temporal-allocation resolver (RESERVED axis, default uniform).

Locks: default ``uniform`` returns the SAME global S for every depth (byte-identical);
``explicit`` parses + validates a per-depth list; ``budget`` is a no-op that returns
uniform + a ``derivation_deferred`` marker; the mode resolver rejects unknowns.
"""

import pytest

from mimarsinan.tuning.orchestration.temporal_allocation import (
    BUDGET_DERIVATION_DEFERRED,
    S_ALLOCATION_BUDGET,
    S_ALLOCATION_EXPLICIT,
    S_ALLOCATION_MODES,
    S_ALLOCATION_UNIFORM,
    TemporalAllocation,
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
)


class TestModeResolution:
    def test_default_is_uniform(self):
        assert resolve_s_allocation_mode({}) == S_ALLOCATION_UNIFORM
        assert resolve_s_allocation_mode({"s_allocation": None}) == S_ALLOCATION_UNIFORM

    def test_explicit_and_budget(self):
        assert resolve_s_allocation_mode({"s_allocation": "explicit"}) == "explicit"
        assert resolve_s_allocation_mode({"s_allocation": "budget"}) == "budget"

    def test_case_insensitive(self):
        assert resolve_s_allocation_mode({"s_allocation": "UNIFORM"}) == "uniform"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_s_allocation_mode({"s_allocation": "per_depth"})

    def test_modes_tuple(self):
        assert S_ALLOCATION_MODES == ("uniform", "explicit", "budget")


class TestUniformIsByteIdentical:
    """The default: every depth gets the SAME global simulation_steps."""

    @pytest.mark.parametrize("depth", [1, 3, 6, 9])
    @pytest.mark.parametrize("steps", [4, 32, 256])
    def test_uniform_repeats_global_steps(self, depth, steps):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": steps}
        )
        alloc = resolver.resolve(depth=depth)
        assert alloc.mode == S_ALLOCATION_UNIFORM
        assert alloc.per_depth_steps == tuple([steps] * depth)
        assert alloc.depth == depth
        assert alloc.is_uniform is True
        assert alloc.global_steps == steps
        assert alloc.derivation_deferred is None
        for d in range(depth):
            assert alloc.steps_at(d) == steps

    def test_uniform_ignores_explicit_and_budget_inputs(self):
        # The escape-hatch inputs are inert when the mode is uniform.
        resolver = TemporalAllocationResolver.from_config(
            {
                "simulation_steps": 32,
                "s_allocation_explicit": [4, 8, 16],
                "s_allocation_budget": {"target": 0.95},
            }
        )
        alloc = resolver.resolve(depth=3)
        assert alloc.per_depth_steps == (32, 32, 32)
        assert alloc.is_uniform is True

    def test_missing_simulation_steps_raises(self):
        with pytest.raises(ValueError):
            TemporalAllocationResolver.from_config({})

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_global_steps_raises(self, bad):
        with pytest.raises(ValueError):
            TemporalAllocationResolver.from_config({"simulation_steps": bad})

    @pytest.mark.parametrize("bad_depth", [0, -3])
    def test_nonpositive_depth_raises(self, bad_depth):
        resolver = TemporalAllocationResolver.from_config({"simulation_steps": 32})
        with pytest.raises(ValueError):
            resolver.resolve(depth=bad_depth)


class TestExplicitReserved:
    """``explicit`` parses + validates a per-depth list (RESERVED — not threaded)."""

    def test_explicit_list_is_returned(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "explicit",
             "s_allocation_explicit": [4, 8, 32]}
        )
        alloc = resolver.resolve(depth=3)
        assert alloc.mode == S_ALLOCATION_EXPLICIT
        assert alloc.per_depth_steps == (4, 8, 32)
        assert alloc.is_uniform is False
        assert alloc.derivation_deferred is None
        assert alloc.steps_at(0) == 4
        assert alloc.steps_at(2) == 32

    def test_explicit_requires_the_list(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "explicit"}
        )
        with pytest.raises(ValueError):
            resolver.resolve(depth=3)

    def test_explicit_length_must_match_depth(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "explicit",
             "s_allocation_explicit": [4, 8]}
        )
        with pytest.raises(ValueError):
            resolver.resolve(depth=3)

    @pytest.mark.parametrize("bad", [[4, 0, 8], [4, -1], [], "32", {"a": 1}])
    def test_explicit_list_validation(self, bad):
        with pytest.raises(ValueError):
            TemporalAllocationResolver.from_config(
                {"simulation_steps": 32, "s_allocation": "explicit",
                 "s_allocation_explicit": bad}
            )

    def test_explicit_coerces_floats_to_ints(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "explicit",
             "s_allocation_explicit": [4.0, 8.0]}
        )
        assert resolver.resolve(depth=2).per_depth_steps == (4, 8)


class TestBudgetIsNoOpReserved:
    """``budget`` is a no-op: returns uniform + the deferred marker (research)."""

    def test_budget_returns_uniform_with_marker(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "budget",
             "s_allocation_budget": {"max_latency_steps": 100, "target": 0.95}}
        )
        alloc = resolver.resolve(depth=4)
        assert alloc.mode == S_ALLOCATION_BUDGET
        assert alloc.per_depth_steps == (32, 32, 32, 32)
        assert alloc.is_uniform is True
        assert alloc.derivation_deferred == BUDGET_DERIVATION_DEFERRED
        assert alloc.budget == {"max_latency_steps": 100, "target": 0.95}

    def test_budget_body_optional(self):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "budget"}
        )
        alloc = resolver.resolve(depth=2)
        assert alloc.is_uniform is True
        assert alloc.derivation_deferred == BUDGET_DERIVATION_DEFERRED
        assert alloc.budget == {}

    def test_budget_must_be_a_dict(self):
        with pytest.raises(ValueError):
            TemporalAllocationResolver.from_config(
                {"simulation_steps": 32, "s_allocation": "budget",
                 "s_allocation_budget": [1, 2, 3]}
            )

    def test_budget_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            TemporalAllocationResolver.from_config(
                {"simulation_steps": 32, "s_allocation": "budget",
                 "s_allocation_budget": {"made_up": 1}}
            )

    @pytest.mark.parametrize(
        "body",
        [
            {"max_energy_proxy": 1.0},
            {"max_latency_steps": 64},
            {"target": 0.96},
            {"max_energy_proxy": 1.0, "max_latency_steps": 64, "target": 0.96},
        ],
    )
    def test_budget_accepts_valid_keys(self, body):
        resolver = TemporalAllocationResolver.from_config(
            {"simulation_steps": 32, "s_allocation": "budget",
             "s_allocation_budget": body}
        )
        assert resolver.resolve(depth=2).budget == body


class TestTemporalAllocationDataclass:
    def test_is_uniform_detects_mixed(self):
        alloc = TemporalAllocation(
            mode="explicit",
            per_depth_steps=(4, 32),
            global_steps=32,
        )
        assert alloc.is_uniform is False
        assert alloc.depth == 2

    def test_is_uniform_true_when_all_global(self):
        alloc = TemporalAllocation(
            mode="uniform",
            per_depth_steps=(32, 32),
            global_steps=32,
        )
        assert alloc.is_uniform is True
