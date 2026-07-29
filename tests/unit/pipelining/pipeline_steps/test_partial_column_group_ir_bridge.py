"""W3d bridge lock: ``partial_column_group`` ELEMENT kills reach the IR cascade.

BC-1 flagged a suspected gap: the criterion's 1-D seed buffers
(``prune_row_mask`` / ``prune_col_mask`` = ``element.all(dim=...)``) only carry
rows/cols COMPLETED at seed time, and ``get_initial_pruning_masks_from_model``
reads only those buffers — so incomplete group kills allegedly never reach the
IR cascade. The bridge is VALUE SEEDING: the production one-shot path COMMITS
the element mask into the raw params (``_commit_pruning_to_raw_params``:
``mask * param``), the committed zeros are baked into the mapped IR matrices,
and ``build_global_pruning_context._seed_value_based`` (``zero_threshold``)
harvests every row/col those zeros complete — including completions that only
materialize AFTER weight quantization rounds small survivors to zero.

This file pins that bridge end-to-end through the REAL production instruments
(``apply_structured_pruning_if_enabled`` → ``SoftCoreMappingStep.
_commit_pruning_to_raw_params`` → ``IRMapping.map`` → ``quantize_ir_graph`` →
``apply_ir_pruning_if_enabled``) on a hand-crafted layer where every group
kill is chosen analytically:

- torch layer (out=16, in=16), ``group_size=4`` ⇒ 4 groups per column,
  64 groups; ``fraction=20/64`` kills exactly the 20 lowest-L2 groups.
- KILL (weight 1e-4): group-row 0 (rows 0..3, every column) + groups
  (1..3, col 4) + group (2, col 7).
- QUANT-DIE (weight 1e-3): groups (1..3, col 9) — survive the criterion,
  but round to zero under the real ``quantize_ir_graph``.
- KEEP (weight 0.75): everything else (quantizes to a nonzero integer).

Expected completions:
- torch rows 0..3 complete — each ONLY by the union of 16 group kills;
- torch col 4 completes ONLY by the union of 4 vertically adjacent group
  kills (no single kill suffices);
- torch col 7 stays PARTIAL (rows 8..11 zeroed, row/col alive);
- torch col 9 completes ONLY IN VALUES after quantization (mask rows 0..3 +
  quantized-to-zero rows 4..15) — never present in the 1-D seed buffers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mapping_utils import (
    EinopsRearrangeMapper,
    Ensure2DMapper,
    InputMapper,
    ModelRepresentation,
    ModuleMapper,
    PerceptronMapper,
)
from mimarsinan.mapping.pruning.ir_pruning_masks import (
    get_initial_pruning_masks_from_model,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_ir_pruning import (
    apply_ir_pruning_if_enabled,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    apply_structured_pruning_if_enabled,
)
from mimarsinan.transformations.pruning.committed_masks import (
    verify_committed_pruning,
)


# --------------------------------------------------------------------------- #
# Fixture: post-fusion perceptron MLP with hand-crafted middle-layer weights.  #
# --------------------------------------------------------------------------- #
class _FusedMLP(PerceptronFlow):
    """Identity-norm perceptron MLP (the post-NormalizationFusionStep state)."""

    def __init__(self, input_shape, widths):
        super().__init__("cpu")
        self.input_activation = nn.Identity()
        self.input_shape = input_shape
        self.perceptrons = nn.ModuleList(
            Perceptron(
                output_channels=widths[i + 1],
                input_features=widths[i],
                normalization=nn.Identity(),
            )
            for i in range(len(widths) - 1)
        )
        inp = InputMapper(input_shape)
        self._iam = ModuleMapper(inp, self.input_activation)
        out = EinopsRearrangeMapper(self._iam, "... c h w -> ... (c h w)")
        out = Ensure2DMapper(out)
        for p in self.perceptrons:
            out = PerceptronMapper(out, p)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        self._iam.module = activation

    def forward(self, x):
        return self._mapper_repr(x)


_INPUT_SHAPE = (1, 4, 4)  # 16 input features
_WIDTHS = [16, 16, 16, 10]  # p0 (input-exempt), p1 (target), p2 (output-exempt)

_GROUP_SIZE = 4
_FRACTION = 20.0 / 64.0  # exactly the 20 analytically chosen kill groups

_KILL = 1e-4  # criterion-killed group weight (20 lowest L2 scores)
_QUANT_DIE = 1e-3  # survives the criterion, rounds to 0 under quantization
_KEEP = 0.75  # quantizes to a nonzero integer

_TARGET = 1  # the only non-exempt perceptron


def _craft_model() -> _FusedMLP:
    torch.manual_seed(7)
    model = _FusedMLP(_INPUT_SHAPE, _WIDTHS)
    with torch.no_grad():
        for p in model.get_perceptrons():
            p.layer.weight.data.fill_(_KEEP)
            p.layer.bias.data.fill_(0.6)  # quantizes to nonzero (bias alive)
        w = model.get_perceptrons()[_TARGET].layer.weight.data
        # KILL: group-row 0 across every column (completes torch rows 0..3)
        w[0:4, :] = _KILL
        # KILL: the 3 remaining groups of column 4 (completes torch col 4
        # only via the union of 4 vertically adjacent group kills)
        w[4:16, 4] = _KILL
        # KILL: group (2, col 7) — partial, completes nothing
        w[8:12, 7] = _KILL
        # QUANT-DIE: the 3 remaining groups of column 9 — the criterion keeps
        # them, quantization rounds them to zero (value-only completion)
        w[4:16, 9] = _QUANT_DIE
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, *_INPUT_SHAPE))
    return model


def _expected_element_mask() -> torch.Tensor:
    e = torch.zeros(16, 16, dtype=torch.bool)
    e[0:4, :] = True
    e[4:16, 4] = True
    e[8:12, 7] = True
    return e


def _step(**overrides) -> SoftCoreMappingStep:
    cfg = default_config()
    cfg.update(
        pruning=True,
        prune_sparsity=_FRACTION,
        prune_criterion="partial_column_group",
        prune_group_size=_GROUP_SIZE,
        input_shape=_INPUT_SHAPE,
        num_classes=_WIDTHS[-1],
    )
    cfg.update(overrides)
    return SoftCoreMappingStep(MockPipeline(config=cfg))


def _ir_of(model):
    mapper_repr = model.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    ir_mapping = IRMapping(
        q_max=127,
        firing_mode="Default",
        max_axons=256,
        max_neurons=64,
        allow_coalescing=True,
        hardware_bias=True,
    )
    return ir_mapping.map(mapper_repr)


def _core_of(ir_graph, perceptron_index: int) -> NeuralCore:
    cores = [
        n for n in ir_graph.nodes
        if isinstance(n, NeuralCore)
        and getattr(n, "perceptron_index", None) == perceptron_index
    ]
    assert len(cores) == 1, (
        f"expected exactly one NeuralCore for perceptron {perceptron_index}, "
        f"got {len(cores)} (the fixture must fit untiled)"
    )
    return cores[0]


def _seeded_committed_model():
    """The REAL one-shot path up to mapping: seed install + raw-param commit."""
    model = _craft_model()
    step = _step()
    result = apply_structured_pruning_if_enabled(step, model, "W3dBridgeTest")
    assert result is None, "foreign criterion must not structurally shrink"
    step._commit_pruning_to_raw_params(model)
    verify_committed_pruning(model.get_perceptrons(), where="W3dBridgeTest")
    return model, step


# --------------------------------------------------------------------------- #
# 1. Seed-time truth: what the buffers do and do NOT carry (BC-1's premise).   #
# --------------------------------------------------------------------------- #
class TestSeedBuffersCarryOnlyCompletions:
    def test_element_mask_is_exactly_the_analytic_kill_set(self):
        model = _craft_model()
        apply_structured_pruning_if_enabled(_step(), model, "W3dBridgeTest")
        p1 = model.get_perceptrons()[_TARGET]
        assert torch.equal(p1.layer.prune_mask, _expected_element_mask())

    def test_1d_buffers_carry_completed_rows_cols_and_nothing_more(self):
        model = _craft_model()
        apply_structured_pruning_if_enabled(_step(), model, "W3dBridgeTest")
        p1 = model.get_perceptrons()[_TARGET]
        expected_rows = torch.zeros(16, dtype=torch.bool)
        expected_rows[0:4] = True  # completed only by the union of 16 kills
        expected_cols = torch.zeros(16, dtype=torch.bool)
        expected_cols[4] = True  # completed only by the union of 4 kills
        assert torch.equal(p1.layer.prune_row_mask, expected_rows)
        assert torch.equal(p1.layer.prune_col_mask, expected_cols)
        # BC-1's premise, pinned: the partial kill (col 7) and the
        # quantization-completed col 9 are NOT claimed by the 1-D buffers.
        assert not p1.layer.prune_col_mask[7]
        assert not p1.layer.prune_col_mask[9]
        assert not p1.layer.prune_row_mask[8:12].any()

    def test_exempt_boundary_layers_get_no_kills(self):
        model = _craft_model()
        apply_structured_pruning_if_enabled(_step(), model, "W3dBridgeTest")
        ps = model.get_perceptrons()
        assert not ps[0].layer.prune_mask.any()
        assert not ps[-1].layer.prune_mask.any()


# --------------------------------------------------------------------------- #
# 2. (a) Committed element zeros are present in the mapped IR matrices.        #
# --------------------------------------------------------------------------- #
class TestCommittedZerosSurfaceInIRMatrices:
    def test_ir_core_matrix_carries_the_element_kills(self):
        model, _ = _seeded_committed_model()
        ir_graph = _ir_of(model)
        core = _core_of(ir_graph, _TARGET)
        mat = np.asarray(core.get_core_matrix(ir_graph))
        # IR convention: rows = axons (torch input cols), cols = neurons
        # (torch output rows); an extra trailing bias row may be present.
        assert mat.shape[1] == 16
        assert mat.shape[0] in (16, 17)
        # completed torch col 4 -> axon row 4 all zero
        assert np.all(mat[4, :] == 0.0)
        # partial group kill (2, col 7) -> element zeros, row/col NOT dead
        assert np.all(mat[7, 8:12] == 0.0)
        assert np.any(mat[7, :] != 0.0)
        # completed torch rows 0..3 -> neuron cols 0..3 all zero (bias
        # entries included: prune_bias_mask committed them to zero)
        assert np.all(mat[:, 0:4] == 0.0)
        if core.hardware_bias is not None:
            hb = np.asarray(core.hardware_bias)
            assert np.all(hb[0:4] == 0.0)
        # QUANT-DIE survivors are still nonzero BEFORE quantization
        assert np.all(np.abs(mat[9, 4:16]) > 0.0)

    def test_real_quantization_completes_the_quant_die_column(self):
        model, _ = _seeded_committed_model()
        ir_graph = _ir_of(model)
        quantize_ir_graph(ir_graph, 8, weight_quantization=True)
        core = _core_of(ir_graph, _TARGET)
        mat = np.asarray(core.get_core_matrix(ir_graph))
        # the tiny survivors of torch col 9 rounded to zero -> axon row 9 dead
        assert np.all(mat[9, :] == 0)
        # KEEP weights stay alive under quantization (nonzero integers)
        assert np.all(mat[0:4, 4:16] != 0)


# --------------------------------------------------------------------------- #
# 3. (b) Completed rows/cols reach the IR seed extraction (buffer path).       #
# --------------------------------------------------------------------------- #
class TestCompletionsReachIRSeedExtraction:
    def test_completed_rows_cols_land_in_initial_node_masks(self):
        model, _ = _seeded_committed_model()
        ir_graph = _ir_of(model)
        initial_node, _initial_bank = get_initial_pruning_masks_from_model(
            model, ir_graph
        )
        core = _core_of(ir_graph, _TARGET)
        assert core.id in initial_node
        ir_row_mask, ir_col_mask = initial_node[core.id]
        # axons: completed torch col 4 pruned; partial col 7 and
        # quant-die col 9 NOT claimed by the buffer path
        assert ir_row_mask[4] is True
        assert ir_row_mask[7] is False
        assert ir_row_mask[9] is False
        # neurons: completed torch rows 0..3 pruned, everything else alive
        assert list(ir_col_mask) == [True] * 4 + [False] * 12


# --------------------------------------------------------------------------- #
# 4. (b)+(c) The REAL cascade harvests every union-completion end-to-end.      #
# --------------------------------------------------------------------------- #
class TestRealCascadeHarvestsUnionCompletions:
    def _pruned_graph(self):
        model, step = _seeded_committed_model()
        ir_graph = _ir_of(model)
        quantize_ir_graph(ir_graph, 8, weight_quantization=True)
        pre_shapes = {
            i: np.asarray(_core_of(ir_graph, i).get_core_matrix(ir_graph)).shape
            for i in range(3)
        }
        pruned = apply_ir_pruning_if_enabled(
            step, model, ir_graph, "W3dBridgeTest"
        )
        return model, pruned, pre_shapes

    def test_union_completed_rows_cols_are_compacted_away(self):
        _, pruned, pre = self._pruned_graph()
        core = _core_of(pruned, _TARGET)
        mat = np.asarray(core.get_core_matrix(pruned))
        nr_pre, nc_pre = pre[_TARGET]
        # (c) axon 4 (union of 4 adjacent group kills) and axon 9 (completed
        # only by quantized VALUES, never in the buffers) are both harvested;
        # neurons 0..3 (each completed only by the union of 16 group kills)
        # are harvested. Nothing else on this core dies.
        assert mat.shape == (nr_pre - 2, nc_pre - 4)

    def test_partial_kill_zeros_survive_compaction_without_removal(self):
        _, pruned, _pre = self._pruned_graph()
        core = _core_of(pruned, _TARGET)
        mat = np.asarray(core.get_core_matrix(pruned))
        # old axon 7 -> index 6 (axon 4 removed below it, axon 9 above);
        # old neurons 8..11 -> cols 4..7 (neurons 0..3 removed)
        assert np.all(mat[6, 4:8] == 0)
        # the rest of that axon row is alive: the partial kill did NOT
        # escalate to a row/col removal
        assert np.any(mat[6, :] != 0)

    def test_cascade_propagates_completions_across_layers(self):
        _, pruned, pre = self._pruned_graph()
        # downstream: p2 loses the 4 input axons fed by p1's dead neurons
        p2 = _core_of(pruned, 2)
        p2_mat = np.asarray(p2.get_core_matrix(pruned))
        nr_pre, nc_pre = pre[2]
        assert p2_mat.shape == (nr_pre - 4, nc_pre)  # logits exempt
        # upstream: p0's neurons 4 and 9 are orphaned by p1's dead axons
        p0 = _core_of(pruned, 0)
        p0_mat = np.asarray(p0.get_core_matrix(pruned))
        nr0_pre, nc0_pre = pre[0]
        assert p0_mat.shape == (nr0_pre, nc0_pre - 2)  # model-input axons exempt

    def test_model_outputs_remain_intact(self):
        _, pruned, _pre = self._pruned_graph()
        assert pruned.output_sources.size == _WIDTHS[-1]

    def test_control_without_quantization_the_quant_die_column_survives(self):
        """Adversarial control: skipping quantization keeps col 9's tiny
        survivors nonzero, so ONLY axon 4 dies on the axon side — proving the
        axon-9 harvest in the main test is the value-seeding bridge acting on
        quantized zeros, not a buffer-path accident."""
        model, step = _seeded_committed_model()
        ir_graph = _ir_of(model)  # NO quantize_ir_graph
        nr_pre, nc_pre = np.asarray(
            _core_of(ir_graph, _TARGET).get_core_matrix(ir_graph)
        ).shape
        pruned = apply_ir_pruning_if_enabled(
            step, model, ir_graph, "W3dBridgeTest"
        )
        mat = np.asarray(_core_of(pruned, _TARGET).get_core_matrix(pruned))
        assert mat.shape == (nr_pre - 1, nc_pre - 4)
        # old axon 9 -> index 8 (only axon 4 removed): still alive
        assert np.any(mat[8, :] != 0.0)
