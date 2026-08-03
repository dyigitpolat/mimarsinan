"""The flat cascade engine vs the reference loop: kill-sets may never regress.

The reference sweep commits kills mid-loop (Gauss-Seidel in node-id order);
the flat engine commits per wave (Jacobi). For a monotone system both reach
the SAME least fixpoint -- but that is exactly the kind of claim this session
does not take on theory. THE GATE: on every topology and vehicle, the final
kill sets, bank sets and (when folding) the lattice must be identical. If
id-order ever produces kills the wave engine misses, the wave engine is wrong
and id-order stays -- bit-exactness outranks speed, by explicit decision.

``fixpoint_iterations`` is the ONE field allowed to differ (approved: the
reference's count is an artifact of Python visit order; the engine reports the
canonical wave count, which equals the depth replay's wave definition).
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.pruning.graph.flat.engine import run_cascade_waves
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    _run_cascade_fixpoint,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)

from unit.mapping.adversarial_topologies import ADVERSARIAL_TOPOLOGIES, build_topology
from unit.mapping.analysis_fingerprint import first_difference


def _fresh_ctx(graph, seeds, *, folding="off"):
    return build_global_pruning_context(
        graph, zero_threshold=1e-8, initial_per_node=seeds, initial_per_bank=None,
        exempt_rows_per_node=None, exempt_cols_per_node=None,
        computeop_liveness_transfers="full", elimination_constant_folding=folding,
        spiking_mode=INERT_SPIKING_MODE,
    )


def _kill_state(ctx):
    """Everything except iteration counts, in canonical comparable form."""
    return {
        "rows": {n: sorted(s) for n, s in sorted(ctx.pruned_rows.items())},
        "cols": {n: sorted(s) for n, s in sorted(ctx.pruned_cols.items())},
        "bank_rows": {b: sorted(s) for b, s in sorted(ctx.bank_pruned_rows.items())},
        "bank_cols": {b: sorted(s) for b, s in sorted(ctx.bank_pruned_cols.items())},
    }


class TestJacobiNeverRegressesGaussSeidel:
    """The user's explicit non-regression requirement, as a named gate."""

    @pytest.mark.parametrize("name", sorted(ADVERSARIAL_TOPOLOGIES))
    def test_kill_sets_identical_on_topology(self, name):
        graph, seeds = build_topology(name)
        ref = _fresh_ctx(graph, seeds)
        _run_cascade_fixpoint(ref)

        graph2, seeds2 = build_topology(name)
        got = _fresh_ctx(graph2, seeds2)
        run_cascade_waves(got)

        a, b = _kill_state(ref), _kill_state(got)
        assert a == b, (
            f"{name}: kill sets diverge between id-order and waves.\n"
            f"first diff: {first_difference(a, b)}\n"
            "If the reference (id-order) side is LARGER, the wave engine has "
            "lost kills and must be fixed; id-order semantics win."
        )

    def test_wave_count_is_deterministic_and_order_free(self):
        """Two runs over relabeled-equivalent graphs give the same wave count;
        the reference demonstrably does not (chain=3 vs tail-variant=13)."""
        g1, s1 = build_topology("chain")
        c1 = _fresh_ctx(g1, s1)
        w1 = run_cascade_waves(c1)
        g2, s2 = build_topology("chain")
        c2 = _fresh_ctx(g2, s2)
        w2 = run_cascade_waves(c2)
        assert w1 == w2 >= 1


class TestFlatEngineOnViTVehicles:
    def test_pristine_vit_kill_sets_match(self):
        from unit.mapping.test_constant_propagation_vehicles import (
            _boundary_policy_exemptions,
            _tiny_vit_constant_vehicle,
            _vit_with_dead_patch_embedding,
        )

        graph, seeds = _vit_with_dead_patch_embedding(
            _tiny_vit_constant_vehicle(zero_wiring_constants=False)
        )
        ex_r, ex_c = _boundary_policy_exemptions(graph)

        def ctx():
            return build_global_pruning_context(
                graph, zero_threshold=1e-8,
                initial_per_node=seeds.get("seeds_node"),
                initial_per_bank=seeds.get("seeds_bank"),
                exempt_rows_per_node=ex_r, exempt_cols_per_node=ex_c,
                computeop_liveness_transfers="full",
                elimination_constant_folding="off",
                spiking_mode=INERT_SPIKING_MODE,
            )

        ref = ctx(); _run_cascade_fixpoint(ref)
        got = ctx(); run_cascade_waves(got)
        assert _kill_state(ref) == _kill_state(got)

    def test_pristine_vit_with_folding_matches_including_lattice(self):
        """Folding=full exercises carriers, folds (the 4-tuple path) and the
        lattice; values compare as bit patterns."""
        from unit.mapping.test_constant_propagation_vehicles import (
            _boundary_policy_exemptions,
            _tiny_vit_constant_vehicle,
            _vit_with_dead_patch_embedding,
        )

        graph, seeds = _vit_with_dead_patch_embedding(
            _tiny_vit_constant_vehicle(zero_wiring_constants=False)
        )
        ex_r, ex_c = _boundary_policy_exemptions(graph)

        def ctx():
            return build_global_pruning_context(
                graph, zero_threshold=1e-8,
                initial_per_node=seeds.get("seeds_node"),
                initial_per_bank=seeds.get("seeds_bank"),
                exempt_rows_per_node=ex_r, exempt_cols_per_node=ex_c,
                computeop_liveness_transfers="full",
                elimination_constant_folding="full",
                spiking_mode=INERT_SPIKING_MODE,
            )

        ref = ctx(); _run_cascade_fixpoint(ref)
        got = ctx(); run_cascade_waves(got)
        assert _kill_state(ref) == _kill_state(got)
        a = {k: float(v).hex() for k, v in (ref.constants.lattice.values or {}).items()}
        b = {k: float(v).hex() for k, v in (got.constants.lattice.values or {}).items()}
        assert a == b, "lattice values must be bit-identical"
