"""[W4b-2] the constant lattice itself: order, descent, gates, transfer rules.

TOP > CONST(c) with one-way descent is what makes the graph fixpoint
terminate, and the four execution probes are what make an executed constant
trustworthy. Both are pinned here; the graph-level consequences (arms,
certificate, ledger) live in ``test_constant_propagation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import ComputeOp, IRSource, NeuralCore
from mimarsinan.mapping.pruning.liveness_transfer import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    ELIMINATION_CONSTANT_FOLDING_FULL,
    ELIMINATION_CONSTANT_FOLDING_OFF,
    ConstantLattice,
    ConstantLatticeError,
    build_computeop_transfer_index,
    derive_constant_outputs,
    derive_core_constants,
    domain_admits_nonzero_constants,
    effective_constant_folding,
    require_elimination_constant_folding,
    resolve_constant_carrier,
    resolve_elimination_constant_folding,
    source_constant,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_core import (
    CARRIER_BIAS,
    CARRIER_ROW,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    OPAQUE_TRANSFER,
)

from unit.mapping.constant_vehicles import srcs


class TestLatticeOrderAndDescent:
    def test_absent_line_is_top(self):
        assert ConstantLattice().get((0, 0)) is None

    def test_descent_is_one_way_and_reports_novelty(self):
        lat = ConstantLattice()
        assert lat.descend((0, 0), 0.5) is True
        assert lat.descend((0, 0), 0.5) is False
        assert lat.get((0, 0)) == 0.5

    def test_conflicting_redescent_fails_loud(self):
        lat = ConstantLattice()
        lat.descend((7, 3), 0.5)
        with pytest.raises(ConstantLatticeError, match="not monotone"):
            lat.descend((7, 3), 0.25)

    def test_non_finite_constants_stay_top(self):
        lat = ConstantLattice()
        assert lat.descend((0, 0), float("nan")) is False
        assert lat.descend((0, 1), float("inf")) is False
        assert lat.values == {}

    def test_zero_only_domain_refuses_nonzero_but_keeps_zero(self):
        lat = ConstantLattice(admits_nonzero=False)
        assert lat.descend((0, 0), 0.5) is False
        assert lat.descend((0, 1), 0.0) is True
        assert lat.values == {(0, 1): 0.0}


class TestSourceConstantIsTheOneLineQuery:
    def test_wiring_kinds(self):
        lat = ConstantLattice()
        assert source_constant(
            IRSource(-1, 0), lattice=lat, pruned_cols={}
        ) == 0.0
        assert source_constant(
            IRSource(-3, 0), lattice=lat, pruned_cols={}
        ) == 1.0
        assert source_constant(
            IRSource(-2, 0), lattice=lat, pruned_cols={}
        ) is None

    def test_eliminated_producer_reads_zero(self):
        lat = ConstantLattice()
        assert source_constant(
            IRSource(4, 2), lattice=lat, pruned_cols={4: {2}}
        ) == 0.0

    def test_recorded_constant_outranks_elimination(self):
        """A CONST(c != 0) column is only eliminated after every reader folded
        c away, so the recorded value — not the post-elimination zero — is what
        the consumers were compiled against."""
        lat = ConstantLattice()
        lat.descend((4, 2), 0.5)
        assert source_constant(
            IRSource(4, 2), lattice=lat, pruned_cols={4: {2}}
        ) == 0.5


class TestPolicyAxisAndDomainGate:
    def test_default_is_full_and_unknown_fails_loud(self):
        assert resolve_elimination_constant_folding({}) == (
            ELIMINATION_CONSTANT_FOLDING_FULL
        )
        assert resolve_elimination_constant_folding(
            {"elimination_constant_folding": "off"}
        ) == ELIMINATION_CONSTANT_FOLDING_OFF
        with pytest.raises(ValueError, match="elimination_constant_folding"):
            require_elimination_constant_folding("sometimes")

    def test_identity_only_transfers_force_folding_off(self):
        assert effective_constant_folding(
            policy=ELIMINATION_CONSTANT_FOLDING_FULL,
            computeop_liveness_transfers=(
                COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY
            ),
        ) == ELIMINATION_CONSTANT_FOLDING_OFF

    def test_nonzero_constants_only_in_the_value_domain(self):
        assert domain_admits_nonzero_constants(INERT_SPIKING_MODE) is True
        for mode in ("lif", "ttfs", "rate", ""):
            assert domain_admits_nonzero_constants(mode) is False


def _op(module, n_in=4, n_out=4, op_type=None, params=None):
    p = {"module": module, "input_shape": (n_in,)}
    p.update(params or {})
    return ComputeOp(
        id=9, name="probe", input_sources=srcs([(0, j) for j in range(n_in)]),
        op_type=op_type or type(module).__name__, params=p,
        input_shape=(n_in,), output_shape=(n_out,),
    )


class TestComputeOpForwardTransferIsGenericAndExecuted:
    def test_sigmoid_of_zero_folds_to_one_half(self):
        """Zero-preservation is NOT required for forward constant flow."""
        out = derive_constant_outputs(
            _op(nn.Sigmoid().eval()), OPAQUE_TRANSFER, [0.0] * 4
        )
        assert out == {j: 0.5 for j in range(4)}

    def test_gelu_of_zero_folds_to_zero(self):
        out = derive_constant_outputs(
            _op(nn.GELU().eval()), OPAQUE_TRANSFER, [0.0] * 4
        )
        assert out == {j: 0.0 for j in range(4)}

    def test_elementwise_region_folds_per_port_with_live_neighbours(self):
        """ELEMENTWISE_1TO1: port j needs only port j constant."""
        op = _op(nn.ReLU().eval())
        index = build_computeop_transfer_index(
            _graph_with(op)
        )
        out = derive_constant_outputs(
            op, index.per_op[op.id], [0.0, None, 0.0, None]
        )
        assert set(out) == {0, 2}

    def test_opaque_join_needs_every_branch_constant(self):
        class Add(nn.Module):
            def forward(self, a, b):
                return a + b

        op = _op(
            Add().eval(), n_in=4, n_out=2, op_type="add",
            params={"input_shapes": [(2,), (2,)]},
        )
        assert derive_constant_outputs(
            op, OPAQUE_TRANSFER, [0.5, 0.5, None, None]
        ) == {}
        assert derive_constant_outputs(
            op, OPAQUE_TRANSFER, [0.5, 0.5, 0.25, 0.25]
        ) == {0: 0.75, 1: 0.75}

    def test_add_with_a_parameter_folds_because_the_parameter_is_known(self):
        class AddParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = nn.Parameter(torch.full((4,), 0.25))

            def forward(self, x):
                return x + self.offset

        out = derive_constant_outputs(
            _op(AddParam().eval()), OPAQUE_TRANSFER, [0.5] * 4
        )
        assert out == {j: 0.75 for j in range(4)}

    def test_nondeterministic_module_is_refused(self):
        class Coin(nn.Module):
            def forward(self, x):
                return x + torch.rand_like(x)

        assert derive_constant_outputs(
            _op(Coin().eval()), OPAQUE_TRANSFER, [0.0] * 4
        ) == {}

    def test_batch_coupled_module_is_refused(self):
        class BatchMean(nn.Module):
            def forward(self, x):
                return x + x.shape[0]

        assert derive_constant_outputs(
            _op(BatchMean().eval()), OPAQUE_TRANSFER, [0.0] * 4
        ) == {}

    def test_training_mode_dropout_is_refused_by_mode_independence(self):
        assert derive_constant_outputs(
            _op(nn.Dropout(p=0.9).train()), OPAQUE_TRANSFER, [0.5] * 4
        ) == {}

    def test_dtype_unstable_constant_is_refused(self):
        """GELU of a NON-zero constant genuinely differs between fp32 and
        fp64, so folding it would not be bit-exact in every deployment."""
        assert derive_constant_outputs(
            _op(nn.GELU().eval()), OPAQUE_TRANSFER, [0.3] * 4
        ) == {}

    def test_module_less_identity_relay_folds_index_wise(self):
        op = ComputeOp(
            id=9, name="relay", input_sources=srcs([(0, 0), (0, 1)]),
            op_type="identity", output_shape=(2,),
        )
        assert derive_constant_outputs(
            op, OPAQUE_TRANSFER, [0.5, None]
        ) == {0: 0.5}


def _graph_with(op):
    from mimarsinan.mapping.ir import IRGraph

    core = NeuralCore(
        id=0, name="c0", input_sources=srcs([(-2, 0)]),
        core_matrix=np.ones((1, 4)), threshold=1.0, latency=0,
    )
    return IRGraph(nodes=[core, op], output_sources=srcs([(9, 0)]))


def _core(sources, matrix, bias=None):
    return NeuralCore(
        id=0, name="c", input_sources=srcs(sources), core_matrix=matrix,
        threshold=1.0, hardware_bias=bias, latency=0,
    )


class TestCarrierResolution:
    def test_always_on_row_is_the_carrier(self):
        node = _core([(-2, 0), (-3, 0)], np.ones((2, 2)))
        carrier = resolve_constant_carrier(
            node, pruned_rows=frozenset(), exempt_rows=frozenset()
        )
        assert carrier is not None
        assert (carrier.kind, carrier.row, carrier.writable) == (
            CARRIER_ROW, 1, True
        )

    def test_hardware_bias_is_the_carrier_when_no_always_on_row(self):
        node = _core([(-2, 0)], np.ones((1, 2)), bias=np.zeros(2))
        carrier = resolve_constant_carrier(
            node, pruned_rows=frozenset(), exempt_rows=frozenset()
        )
        assert carrier is not None and carrier.kind == CARRIER_BIAS

    def test_no_carrier_when_the_core_encodes_no_constant(self):
        node = _core([(-2, 0)], np.ones((1, 2)))
        assert resolve_constant_carrier(
            node, pruned_rows=frozenset(), exempt_rows=frozenset()
        ) is None

    def test_bank_backed_carrier_is_read_only(self):
        node = NeuralCore(
            id=0, name="c", input_sources=srcs([(-2, 0), (-3, 0)]),
            core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
            threshold=1.0, latency=0,
        )
        carrier = resolve_constant_carrier(
            node, pruned_rows=frozenset(), exempt_rows=frozenset()
        )
        assert carrier is not None and carrier.writable is False


class TestCoreConstantRules:
    def _facts(self, row_values, matrix, carrier, **kw):
        return derive_core_constants(
            matrix=matrix, bias=kw.pop("bias", None),
            threshold=kw.pop("threshold", 1.0), row_values=row_values,
            pruned_rows=kw.pop("pruned_rows", frozenset()),
            pruned_cols=frozenset(), exempt_rows=frozenset(),
            carrier=carrier, admits_nonzero=kw.pop("admits_nonzero", True),
        )

    def test_zero_constant_rows_die_without_a_carrier(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        facts = self._facts([0.0, None], matrix, None)
        assert facts.dead_rows == {0} and facts.fold_rows == {}

    def test_nonzero_constant_rows_fold_onto_a_writable_carrier(self):
        matrix = np.array([[2.0, 4.0], [1.0, 1.0]])
        carrier = resolve_constant_carrier(
            _core([(-2, 0), (-3, 0)], matrix),
            pruned_rows=frozenset(), exempt_rows=frozenset(),
        )
        facts = self._facts([0.5, 1.0], matrix, carrier)
        assert facts.fold_rows == {0: 0.5}

    def test_nonzero_folds_refused_in_the_zero_only_domain(self):
        matrix = np.array([[2.0, 4.0], [1.0, 1.0]])
        carrier = resolve_constant_carrier(
            _core([(-2, 0), (-3, 0)], matrix),
            pruned_rows=frozenset(), exempt_rows=frozenset(),
        )
        facts = self._facts(
            [0.5, 1.0], matrix, carrier, admits_nonzero=False
        )
        assert facts.fold_rows == {}

    def test_column_is_constant_once_the_carrier_is_the_only_contributor(self):
        matrix = np.array([[2.0, 4.0], [0.5, 0.25]])
        carrier = resolve_constant_carrier(
            _core([(-2, 0), (-3, 0)], matrix),
            pruned_rows=frozenset(), exempt_rows=frozenset(),
        )
        facts = self._facts(
            [None, 1.0], matrix, carrier, pruned_rows={0}
        )
        assert facts.column_values == {0: 0.5, 1: 0.25}

    def test_a_live_unresolved_row_blocks_its_columns(self):
        matrix = np.array([[2.0, 0.0], [0.5, 0.25]])
        carrier = resolve_constant_carrier(
            _core([(-2, 0), (-3, 0)], matrix),
            pruned_rows=frozenset(), exempt_rows=frozenset(),
        )
        facts = self._facts([None, 1.0], matrix, carrier)
        assert facts.column_values == {1: 0.25}

    def test_threshold_divides_the_constant_like_the_value_executor(self):
        matrix = np.array([[0.0, 0.0], [1.0, 2.0]])
        carrier = resolve_constant_carrier(
            _core([(-2, 0), (-3, 0)], matrix),
            pruned_rows=frozenset(), exempt_rows=frozenset(),
        )
        facts = self._facts(
            [None, 1.0], matrix, carrier, pruned_rows={0}, threshold=4.0
        )
        assert facts.column_values == {0: 0.25, 1: 0.5}
