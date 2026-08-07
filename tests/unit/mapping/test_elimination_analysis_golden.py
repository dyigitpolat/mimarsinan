"""Golden differential gate for elimination-analysis performance work.

The optimisation programme (docs/elimination_analysis_performance_design.md) is
allowed to change HOW the analysis is computed and nothing about WHAT it
computes. These tests pin the observable surface -- kill sets, the constant
lattice bit-for-bit, fixpoint iterations, and per-kill depths -- across every
arm and both folding policies.

Two properties are load-bearing:

* the fingerprint captures floats as ``float.hex()``, so a 1-ulp drift fails
  rather than passing a tolerance;
* the mutation tests below prove the fingerprint CAN fail. A differential
  harness that cannot detect a corrupted result proves nothing about the
  optimisation it is guarding.
"""

import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from unit.mapping.analysis_fingerprint import (
    analysis_fingerprint,
    fingerprint_digest,
    first_difference,
)
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets

from unit.mapping.test_constant_propagation_vehicles import (
    _boundary_policy_exemptions,
    _tiny_vit_constant_vehicle,
    _vit_with_dead_patch_embedding,
)

ARMS = ("masked", "closure", "cascade")
FOLDING = ("off", "full")


def _run(graph, seeds, *, mode, folding):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    return compute_global_pruned_sets(
        graph, zero_threshold=1e-8,
        initial_per_node=seeds.get("seeds_node"),
        initial_per_bank=seeds.get("seeds_bank"),
        exempt_rows_per_node=exempt_rows, exempt_cols_per_node=exempt_cols,
        mode=mode, elimination_constant_folding=folding,
        spiking_mode=INERT_SPIKING_MODE,
    )


@pytest.fixture(scope="module")
def vehicles():
    """Both ViT vehicles: the zeroed twin and the pristine one."""
    return {
        "twin": _vit_with_dead_patch_embedding(_tiny_vit_constant_vehicle()),
        "pristine": _vit_with_dead_patch_embedding(
            _tiny_vit_constant_vehicle(zero_wiring_constants=False)
        ),
    }


class TestTheFingerprintIsDeterministic:
    """A golden is worthless if the baseline itself wobbles."""

    @pytest.mark.parametrize("vehicle", ("twin", "pristine"))
    @pytest.mark.parametrize("mode", ARMS)
    def test_same_inputs_give_the_same_digest(self, vehicles, vehicle, mode):
        graph, seeds = vehicles[vehicle]
        a = analysis_fingerprint(_run(graph, seeds, mode=mode, folding="full"))
        b = analysis_fingerprint(_run(graph, seeds, mode=mode, folding="full"))
        assert first_difference(a, b) is None
        assert fingerprint_digest(a) == fingerprint_digest(b)


class TestTheFingerprintSeesEveryObservable:
    """Each field the optimisation could break must be covered."""

    def test_it_captures_kills_lattice_and_iterations(self, vehicles):
        graph, seeds = vehicles["pristine"]
        fp = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="full"))
        assert fp["pruned_rows_per_node"], "kills must be captured"
        assert fp["constant_lattice"], "the lattice must be captured"
        assert fp["fixpoint_iterations"] >= 1
        # floats are bit patterns, not decimal reprs
        assert all(v.startswith(("0x", "-0x", "inf", "-inf", "nan"))
                   for v in fp["constant_lattice"].values())

    def test_folding_off_and_full_are_distinguishable(self, vehicles):
        graph, seeds = vehicles["pristine"]
        off = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="off"))
        full = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="full"))
        assert first_difference(off, full) is not None, (
            "the constant arm reaches strictly more; a fingerprint that cannot "
            "tell them apart cannot guard the optimisation"
        )


class TestTheHarnessCanActuallyFail:
    """Mutation tests: corrupt a result and prove the gate rejects it."""

    def test_a_single_dropped_kill_is_caught(self, vehicles):
        graph, seeds = vehicles["pristine"]
        good = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="full"))
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in good.items()}
        node = next(n for n, rows in bad["pruned_rows_per_node"].items() if rows)
        bad["pruned_rows_per_node"][node] = bad["pruned_rows_per_node"][node][:-1]
        assert first_difference(good, bad) is not None
        assert fingerprint_digest(good) != fingerprint_digest(bad)

    def test_a_one_ulp_constant_drift_is_caught(self, vehicles):
        """The reason floats are stored as hex: a tolerance would swallow this."""
        import math

        graph, seeds = vehicles["pristine"]
        good = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="full"))
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in good.items()}
        key = next(k for k, v in bad["constant_lattice"].items()
                   if math.isfinite(float.fromhex(v)) and float.fromhex(v) != 0.0)
        drifted = math.nextafter(float.fromhex(bad["constant_lattice"][key]), math.inf)
        bad["constant_lattice"][key] = drifted.hex()
        assert first_difference(good, bad) is not None, (
            "a 1-ulp drift in a folded constant MUST fail the gate"
        )

    def test_an_iteration_count_change_is_caught(self, vehicles):
        graph, seeds = vehicles["pristine"]
        good = analysis_fingerprint(_run(graph, seeds, mode="cascade", folding="full"))
        bad = dict(good)
        bad["fixpoint_iterations"] = good["fixpoint_iterations"] + 1
        assert first_difference(good, bad) is not None
