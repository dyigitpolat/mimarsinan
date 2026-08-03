"""Run-scoped probe memoisation: pure-function caching of op probe batteries.

``derive_constant_outputs`` is pure in (op, bitwise input key): hosted modules
are never mutated by folds (folds touch core matrices and biases), and the
probe battery forks RNG. A memo shared across fresh contexts of one run must
therefore change NOTHING in kills or lattice — it only removes repeated
batteries. Pins: (1) bitwise key discipline (None / -0.0 / 0.0 never
collide), (2) bit-identity under sharing, (3) the memo is actually consulted,
(4) the ledger arms thread one shared memo end-to-end.
"""

import pytest

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.pruning.elimination_ledger.arm_runs import (
    compute_elimination_arms,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    _run_cascade_fixpoint,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)
from mimarsinan.mapping.pruning.liveness_transfer.probe_memo import (
    probe_key_bytes,
)

import unit.mapping.constant_vehicles as vehicles

FOLD_VEHICLES = [
    "bias_only_collapse_graph", "sigmoid_chain_graph",
    "residual_join_graph", "gelu_execution_exact_graph",
]


class TestProbeKeyBytes:
    """The memo key must be bitwise-exact, stricter than float ==."""

    def test_negative_zero_is_a_distinct_key(self):
        assert probe_key_bytes([0.0]) != probe_key_bytes([-0.0])

    def test_none_is_distinct_from_zero(self):
        assert probe_key_bytes([None]) != probe_key_bytes([0.0])

    def test_equal_lines_produce_equal_bytes(self):
        assert probe_key_bytes([1.5, None, -2.0]) == probe_key_bytes([1.5, None, -2.0])

    def test_marker_cannot_alias_a_value_prefix(self):
        # [None, x] must never collide with a single-line key or [x, None].
        assert probe_key_bytes([None, 1.0]) != probe_key_bytes([1.0, None])


def _run_full_cascade(maker: str, memo):
    graph = getattr(vehicles, maker)()
    ctx = build_global_pruning_context(
        graph, zero_threshold=1e-8,
        initial_per_node=None, initial_per_bank=None,
        exempt_rows_per_node=None, exempt_cols_per_node=None,
        computeop_liveness_transfers="full",
        elimination_constant_folding="full",
        spiking_mode=INERT_SPIKING_MODE,
        probe_memo=memo,
    )
    _run_cascade_fixpoint(ctx)
    return {
        "rows": {n: sorted(s) for n, s in sorted(ctx.pruned_rows.items())},
        "cols": {n: sorted(s) for n, s in sorted(ctx.pruned_cols.items())},
        "lattice": {
            k: float(v).hex()
            for k, v in (ctx.constants.lattice.values or {}).items()
        },
    }


class TestSharedMemoIsBitIdentical:
    """The load-bearing gate: sharing may remove work, never change results."""

    @pytest.mark.parametrize("maker", FOLD_VEHICLES)
    def test_memo_reuse_across_fresh_contexts_changes_nothing(self, maker):
        memo = {}
        first = _run_full_cascade(maker, memo)
        second = _run_full_cascade(maker, memo)   # every battery replayed from memo
        baseline = _run_full_cascade(maker, None)  # per-context memo, no sharing
        assert first == baseline, f"{maker}: populating pass diverged"
        assert second == baseline, f"{maker}: memo-hit pass diverged"


class TestTheMemoIsConsulted:
    """A cache nobody reads is dead code: prove batteries stop re-running."""

    def test_second_run_fires_zero_probe_batteries(self, monkeypatch):
        import mimarsinan.mapping.pruning.liveness_transfer.constant_transfer as ct

        calls = []
        real = ct._probe

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(ct, "_probe", counting)
        memo = {}
        _run_full_cascade("gelu_execution_exact_graph", memo)
        cold = len(calls)
        assert cold > 0, "vehicle must exercise the probe battery"
        _run_full_cascade("gelu_execution_exact_graph", memo)
        assert len(calls) == cold, "shared memo must eliminate every repeat battery"


class TestUnsatisfiableProbesNeverExecute:
    """An op none of whose output regions can be a subset of the known inputs
    cannot resolve anything — the battery's result is discarded for every
    output, so running it is pure waste. The precheck must skip it with the
    IDENTICAL {} result. (Real-scale evidence: a 150,528-line opaque `cat`
    with 7,448 known inputs burned 6.5 s to certify nothing.)"""

    def _battery_counter(self, monkeypatch):
        import mimarsinan.mapping.pruning.liveness_transfer.constant_transfer as ct

        calls = []
        real = ct._probe

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(ct, "_probe", counting)
        return calls

    @staticmethod
    def _opaque_op(module, n_in=4):
        from mimarsinan.mapping.ir import ComputeOp
        from unit.mapping.constant_vehicles import srcs

        return ComputeOp(
            id=9, name="probe",
            input_sources=srcs([(0, j) for j in range(n_in)]),
            op_type=type(module).__name__,
            params={"module": module, "input_shape": (n_in,)},
            input_shape=(n_in,), output_shape=(n_in,),
        )

    def test_opaque_partial_inputs_skip_the_battery(self, monkeypatch):
        from torch import nn

        from mimarsinan.mapping.pruning.liveness_transfer.constant_transfer import (
            derive_constant_outputs,
        )
        from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
            OPAQUE_TRANSFER,
        )

        calls = self._battery_counter(monkeypatch)
        op = self._opaque_op(nn.Sigmoid().eval())
        out = derive_constant_outputs(op, OPAQUE_TRANSFER, [0.5, None, 0.5, 0.5])
        assert out == {}
        assert calls == [], "an unsatisfiable opaque op must not execute"

    def test_opaque_all_known_still_probes(self, monkeypatch):
        from torch import nn

        from mimarsinan.mapping.pruning.liveness_transfer.constant_transfer import (
            derive_constant_outputs,
        )
        from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
            OPAQUE_TRANSFER,
        )

        calls = self._battery_counter(monkeypatch)
        op = self._opaque_op(nn.Sigmoid().eval())
        out = derive_constant_outputs(op, OPAQUE_TRANSFER, [0.0, 0.0, 0.0, 0.0])
        assert calls, "a fully-known opaque op must still run the battery"
        assert out == {j: 0.5 for j in range(4)}


class TestArmsShareOneMemo:
    """compute_elimination_arms threads one memo across masked/closure/cascade
    and exposes it for the replay context."""

    def test_arms_expose_a_populated_memo(self):
        graph = vehicles.gelu_execution_exact_graph()
        arms = compute_elimination_arms(
            graph,
            elimination_constant_folding="full",
            computeop_liveness_transfers="full",
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert arms.probe_memo, "fold vehicle must have populated the shared memo"
