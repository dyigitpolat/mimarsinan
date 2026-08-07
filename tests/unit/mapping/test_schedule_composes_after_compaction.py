"""[U4] Pass composition sizes instances by POST-compaction geometry."""

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
    neural_core_to_soft_core,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.packing.hybrid_segment import _validate_coalescing_budget
from mimarsinan.mapping.packing.schedule_bank_clustered import (
    try_bank_clustered_passes,
)
from mimarsinan.mapping.packing.softcore import (
    compact_soft_core_mapping,
    compacted_core_extent,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.verification.capacity import estimate_cores_needed

IN_FEATURES = 32
OUT_FEATURES = 8
BANK_ROWS = IN_FEATURES + 1  # weight rows + one always-on bias row
DEAD_ROWS = 16

# One resident-capable wide type plus three narrow types the FULL instance
# overflows: elimination is what makes the narrow duplicates usable.
CORES_CLUSTERED = [
    {"max_axons": BANK_ROWS, "max_neurons": OUT_FEATURES, "count": 1},
    {"max_axons": 20, "max_neurons": OUT_FEATURES, "count": 3},
]
# One core that two FULL instances overflow (66 > 34) and two compacted
# instances exactly fill (34 <= 34, 16 <= 16).
CORES_POOL = [{"max_axons": 34, "max_neurons": 16, "count": 1}]


def _bank_instances(*, bank_rows, n_instances=2, producer_id=None):
    """``n_instances`` weight-shared instances of one bank (the wide-conv vehicle)."""
    rng = np.random.default_rng(3)
    bank = WeightBank(
        id=0,
        core_matrix=rng.normal(
            size=(bank_rows, OUT_FEATURES)
        ).astype(np.float32) + 3.0,
    )
    weight_rows = bank_rows - 1
    nodes = []
    specs = []
    for k in range(n_instances):
        upstream = -2 if producer_id is None else producer_id
        sources = [
            IRSource(upstream, k * weight_rows + i) for i in range(weight_rows)
        ] + [IRSource(-3, 0)]
        nodes.append(NeuralCore(
            id=k, name=f"inst{k}",
            input_sources=np.array(sources, dtype=object),
            core_matrix=None, weight_bank_id=0,
            weight_row_slice=(0, OUT_FEATURES), latency=0,
            perceptron_index=0, perceptron_output_column=k,
            layout_softcore_index=k,
        ))
        specs.append(LayoutSoftCoreSpec(
            input_count=bank_rows, output_count=OUT_FEATURES,
            residency_class_id=0, latency_tag=0, segment_id=0, name=f"inst{k}",
        ))
    return bank, nodes, specs


def _bank_graph(*, bank_rows=BANK_ROWS, n_instances=2):
    bank, nodes, specs = _bank_instances(
        bank_rows=bank_rows, n_instances=n_instances,
    )
    outputs = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(OUT_FEATURES)],
        dtype=object,
    )
    return IRGraph(
        nodes=list(nodes), output_sources=outputs, weight_banks={0: bank},
        layout_softcores=specs,
    )


def _mask_bank_rows(graph, dead_rows):
    """Attach the masks ``prune_ir_graph`` leaves on bank-backed instances."""
    rows = graph.weight_banks[0].core_matrix.shape[0]
    for node in graph.nodes:
        node.pruned_row_mask = [i in dead_rows for i in range(rows)]
        node.pruned_col_mask = [False] * OUT_FEATURES
    return graph


def _pruned_bank_graph():
    return _mask_bank_rows(_bank_graph(), set(range(DEAD_ROWS)))


def _compacted_twin():
    """The same instances already materialized at their post-compaction size."""
    return _bank_graph(bank_rows=BANK_ROWS - DEAD_ROWS)


def _strategy(policy):
    return MappingStrategy.resolve(ChipCapabilities(
        allow_scheduling=True, schedule_policy=policy,
    ))


def _stage_shapes(graph, policy, cores_config):
    """(pass count, per-pass placement geometry) of the composed program."""
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph, cores_config=cores_config, strategy=_strategy(policy),
    )
    neural = [s for s in hybrid.stages if s.kind == "neural"]
    geometry = [
        [
            tuple(sorted(
                (int(p["axons"]), int(p["neurons"]),
                 int(p["axon_offset"]), int(p["neuron_offset"]))
                for p in placements
            ))
            for placements in stage.hard_core_mapping
            .soft_core_placements_per_hard_core
        ]
        for stage in neural
    ]
    return len(neural), geometry


class TestPassLadderRespondsToSparsity:
    """The headline: elimination must be able to remove a pass."""

    def test_bank_clustered_chunks_shrink_when_pruned(self):
        full = try_bank_clustered_passes(
            cores=list(_bank_graph().nodes), cores_config=CORES_CLUSTERED,
            weight_banks={}, max_schedule_passes=8,
        )
        pruned = try_bank_clustered_passes(
            cores=list(_pruned_bank_graph().nodes),
            cores_config=CORES_CLUSTERED, weight_banks={},
            max_schedule_passes=8,
        )
        # 33 axons fit only the single wide core -> one instance per pass;
        # 17 axons also fit the narrow triple -> both instances in one pass.
        assert [len(c) for c in full] == [1, 1]
        assert [len(c) for c in pruned] == [2]

    def test_bank_clustered_stage_ladder_shrinks_when_pruned(self):
        full, _ = _stage_shapes(
            _bank_graph(), "bank_clustered", CORES_CLUSTERED,
        )
        pruned, _ = _stage_shapes(
            _pruned_bank_graph(), "bank_clustered", CORES_CLUSTERED,
        )
        assert (full, pruned) == (2, 1)

    def test_capacity_split_ladder_shrinks_when_pruned(self):
        full, _ = _stage_shapes(_bank_graph(), "pool", CORES_POOL)
        pruned, _ = _stage_shapes(_pruned_bank_graph(), "pool", CORES_POOL)
        assert (full, pruned) == (2, 1)

    def test_ladder_responds_through_real_prune_ir_graph(self):
        """End to end: seeds -> prune_ir_graph -> composition."""
        def vehicle(dead_bank_rows):
            host = ComputeOp(
                id=1000, name="host_relay",
                input_sources=np.array(
                    [IRSource(-2, i) for i in range(2 * IN_FEATURES)],
                    dtype=object,
                ),
                op_type="identity",
                input_shape=(2 * IN_FEATURES,),
                output_shape=(2 * IN_FEATURES,),
            )
            bank, nodes, specs = _bank_instances(
                bank_rows=BANK_ROWS, producer_id=host.id,
            )
            outputs = np.array(
                [IRSource(n.id, j) for n in nodes
                 for j in range(OUT_FEATURES)],
                dtype=object,
            )
            graph = IRGraph(
                nodes=[host] + list(nodes), output_sources=outputs,
                weight_banks={0: bank}, layout_softcores=specs,
            )
            seeds = (
                None if not dead_bank_rows
                else {0: (
                    [i in dead_bank_rows for i in range(BANK_ROWS)],
                    [False] * OUT_FEATURES,
                )}
            )
            return prune_ir_graph(graph, initial_pruned_per_bank=seeds)

        full = vehicle(set())
        pruned = vehicle(set(range(DEAD_ROWS)))
        assert [
            sum(n.pruned_row_mask) for n in pruned.get_neural_cores()
        ] == [DEAD_ROWS, DEAD_ROWS]
        assert (
            _stage_shapes(full, "pool", CORES_POOL)[0],
            _stage_shapes(pruned, "pool", CORES_POOL)[0],
        ) == (2, 1)
        assert (
            _stage_shapes(full, "bank_clustered", CORES_CLUSTERED)[0],
            _stage_shapes(pruned, "bank_clustered", CORES_CLUSTERED)[0],
        ) == (2, 1)


class TestNoPreCompactionExtentIsRead:
    """A compacted instance is sized by what it occupies, not what it was."""

    @pytest.mark.parametrize(
        "policy,cores_config",
        [("pool", CORES_POOL), ("bank_clustered", CORES_CLUSTERED)],
    )
    def test_composition_matches_the_already_compacted_twin(
        self, policy, cores_config,
    ):
        masked = _stage_shapes(_pruned_bank_graph(), policy, cores_config)
        twin = _stage_shapes(_compacted_twin(), policy, cores_config)
        assert masked == twin

    def test_capacity_gate_estimate_responds_to_sparsity(self):
        """The gate runs AFTER pruning; it must budget the compacted extent."""
        constraints = {"cores": [{
            "max_axons": BANK_ROWS, "max_neurons": 4 * OUT_FEATURES,
            "count": 8, "has_bias": True,
        }]}
        full = estimate_cores_needed(_bank_graph(n_instances=4), constraints)
        pruned = estimate_cores_needed(
            _mask_bank_rows(
                _bank_graph(n_instances=4), set(range(DEAD_ROWS)),
            ),
            constraints,
        )
        assert full.cores_needed > pruned.cores_needed
        assert pruned.cores_needed == estimate_cores_needed(
            _bank_graph(bank_rows=BANK_ROWS - DEAD_ROWS, n_instances=4),
            constraints,
        ).cores_needed

    def test_coalescing_budget_validates_the_compacted_extent(self):
        """A wide instance that only overflows the coalescing budget before
        elimination must not be refused after it."""
        sources = [IRSource(-2, i) for i in range(40)] + [IRSource(-3, 0)]
        core = NeuralCore(
            id=0, name="wide",
            input_sources=np.array(sources, dtype=object),
            core_matrix=np.ones((41, OUT_FEATURES), dtype=np.float32),
            latency=0, perceptron_index=0,
            pruned_row_mask=[i < 24 for i in range(41)],
            pruned_col_mask=[False] * OUT_FEATURES,
        )
        cores_config = [{"max_axons": 16, "max_neurons": OUT_FEATURES,
                         "count": 2}]
        _validate_coalescing_budget([core], cores_config, False)

        core.pruned_row_mask = [False] * 41
        with pytest.raises(RuntimeError, match="coalescing cores"):
            _validate_coalescing_budget([core], cores_config, False)

    def test_stale_layout_record_is_not_the_extent_authority(self):
        """Owned cores compacted by ``prune_ir_graph`` leave the layout
        record at the pre-elimination extent; sizing must ignore it."""
        nodes = []
        specs = []
        for k in range(2):
            nodes.append(NeuralCore(
                id=k, name=f"owned{k}",
                input_sources=np.array(
                    [IRSource(-2, k * 12 + i) for i in range(12)],
                    dtype=object,
                ),
                core_matrix=np.ones((12, OUT_FEATURES), dtype=np.float32),
                latency=0, perceptron_index=0, layout_softcore_index=k,
                pruned_row_mask=[False] * 12,
                pruned_col_mask=[False] * OUT_FEATURES,
                pre_pruning_row_mask=[i < 8 for i in range(20)],
                pre_pruning_col_mask=[False] * OUT_FEATURES,
            ))
            specs.append(LayoutSoftCoreSpec(
                input_count=20, output_count=OUT_FEATURES,
                residency_class_id=0, latency_tag=0, segment_id=0,
                name=f"owned{k}",
            ))
        outputs = np.array(
            [IRSource(n.id, j) for n in nodes for j in range(OUT_FEATURES)],
            dtype=object,
        )
        graph = IRGraph(
            nodes=nodes, output_sources=outputs, layout_softcores=specs,
        )
        # Stale record: 20 + 20 = 40 > 34 axons (two passes). Truth: 24 <= 34.
        assert _stage_shapes(graph, "pool", CORES_POOL)[0] == 1


class TestUnchangedGeometryIsPreserved:
    """Nothing eliminated -> the composed program is what it always was."""

    def test_unpruned_ladder_is_unchanged(self):
        assert _stage_shapes(_bank_graph(), "pool", CORES_POOL) == (
            2,
            [[((33, 8, 0, 0),)], [((33, 8, 0, 0),)]],
        )
        assert _stage_shapes(
            _bank_graph(), "bank_clustered", CORES_CLUSTERED,
        ) == (2, [[((33, 8, 0, 0),)], [((33, 8, 0, 0),)]])

    def test_all_false_masks_leave_the_extent_alone(self):
        graph = _mask_bank_rows(_bank_graph(), set())
        assert _stage_shapes(graph, "pool", CORES_POOL) == _stage_shapes(
            _bank_graph(), "pool", CORES_POOL,
        )
        for core in graph.nodes:
            assert compacted_core_extent(core) == (BANK_ROWS, OUT_FEATURES)


class TestCompactedExtentMatchesTheCompactor:
    """One rule for post-pruning geometry, shared with the runtime compactor."""

    def _owned(self, n_axons, n_neurons, *, rows=None, cols=None,
               always_on_last=False):
        sources = [IRSource(-2, i) for i in range(n_axons)]
        if always_on_last:
            sources[-1] = IRSource(-3, 0)
        return NeuralCore(
            id=0, name="c",
            input_sources=np.array(sources, dtype=object),
            core_matrix=np.ones((n_axons, n_neurons), dtype=np.float32),
            latency=0,
            pruned_row_mask=rows, pruned_col_mask=cols,
        )

    @pytest.mark.parametrize("core_kwargs", [
        {"n_axons": 6, "n_neurons": 4},
        {"n_axons": 6, "n_neurons": 4,
         "rows": [True, False, True, False, False, False],
         "cols": [False, True, False, False]},
        {"n_axons": 6, "n_neurons": 4,
         "rows": [True] * 5 + [False], "cols": [False] * 4,
         "always_on_last": True},
        # Always-on bias row marked dead is still never dropped.
        {"n_axons": 6, "n_neurons": 4,
         "rows": [True] * 6, "cols": [False] * 4, "always_on_last": True},
        # BIAS_ONLY: every axon dead, bias-driven columns survive.
        {"n_axons": 6, "n_neurons": 4,
         "rows": [True] * 6, "cols": [False, True, False, False]},
    ])
    def test_extent_equals_the_runtime_compactor(self, core_kwargs):
        core = self._owned(**core_kwargs)
        predicted = compacted_core_extent(core)
        soft = neural_core_to_soft_core(core)
        compact_soft_core_mapping([soft], [])
        assert predicted == (
            soft.get_input_count(), soft.get_output_count(),
        )

    def test_mask_length_disagreement_still_fails_loud(self):
        """The extent's defensive branch never softens the IR seam's check."""
        core = self._owned(6, 4, rows=[True] * 3, cols=[False] * 4)
        assert compacted_core_extent(core) == (6, 4)
        with pytest.raises(ValueError, match="mask length mismatch"):
            neural_core_to_soft_core(core)

    def test_extent_of_a_bank_instance_follows_the_bank_masks(self):
        graph = _pruned_bank_graph()
        core = graph.nodes[0]
        assert compacted_core_extent(core) == (
            BANK_ROWS - DEAD_ROWS, OUT_FEATURES,
        )
        soft = neural_core_to_soft_core(core, graph=graph)
        compact_soft_core_mapping([soft], [SpikeSource(core.id, 0, False, False)])
        assert (soft.get_input_count(), soft.get_output_count()) == (
            BANK_ROWS - DEAD_ROWS, OUT_FEATURES,
        )
