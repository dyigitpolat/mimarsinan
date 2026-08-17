"""[memory audit F1/F2] value-flow buffer pruning and shared weight uploads."""

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


def _identity_flow_and_fused():
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 12), nn.GELU(),
        nn.Linear(12, 4),
    ).eval()
    fused = convert_torch_model(
        model, (8,), 4, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for p in fused.get_perceptrons():
        fuse_into_perceptron(p, device="cpu")
    repr_ = fused.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default",
                   max_axons=256, max_neurons=64).map(repr_)
    flow = ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64
    )
    return flow, fused


def _token_bank_hybrid(n_tokens=5):
    rows, cols = 5, 4
    rng = np.random.default_rng(7)
    bank = WeightBank(
        id=0, core_matrix=rng.normal(size=(rows, cols)).astype(np.float64)
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * (rows - 1) + i) for i in range(rows - 1)]
            + [IRSource(-3, 0)], dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"tok{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, cols),
            perceptron_index=0, perceptron_output_column=tok, latency=0,
        ))
    graph = IRGraph(
        nodes=nodes,
        output_sources=np.array(
            [IRSource(n.id, j) for n in nodes for j in range(cols)],
            dtype=object,
        ),
        weight_banks={0: bank},
    )
    strategy = MappingStrategy.resolve(ChipCapabilities(
        allow_scheduling=True,
    ))
    return build_hybrid_hard_core_mapping(
        ir_graph=graph,
        cores_config=[{"max_axons": 8, "max_neurons": 8, "count": 2}],
        strategy=strategy,
    )


class TestCensusBatchResolution:
    """[F3] a DECLARED census bound binds attempt 1; an undeclared row keeps
    the throughput floor (its resolved default only funds the OOM retry)."""

    def test_undeclared_keeps_the_throughput_floor(self):
        from mimarsinan.pipelining.core.simulation_factory import (
            _SIM_EVAL_BATCH_SIZE,
            resolve_census_batch_size,
        )
        assert resolve_census_batch_size(
            128, declared_cap=None, retry_cap=None
        ) == _SIM_EVAL_BATCH_SIZE

    def test_declared_cap_binds_the_first_attempt(self):
        from mimarsinan.pipelining.core.simulation_factory import (
            resolve_census_batch_size,
        )
        assert resolve_census_batch_size(
            128, declared_cap=64, retry_cap=None
        ) == 64

    def test_retry_cap_binds_and_the_tighter_cap_wins(self):
        from mimarsinan.pipelining.core.simulation_factory import (
            resolve_census_batch_size,
        )
        assert resolve_census_batch_size(
            128, declared_cap=None, retry_cap=8
        ) == 8
        assert resolve_census_batch_size(
            128, declared_cap=64, retry_cap=8
        ) == 8

    def test_declared_cap_distinguishes_silence_from_declaration(self):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
        bare = DeploymentPlan.resolve({})
        assert bare.declared_simulation_batch_size is None
        assert bare.simulation_batch_size == 8  # retry fallback only
        declared = DeploymentPlan.resolve({"simulation_batch_size": 64})
        assert declared.declared_simulation_batch_size == 64
        assert declared.simulation_batch_size == 64


class TestStateBufferPruning:
    def test_consumed_entries_are_pruned_and_outputs_exact(self):
        # [F1] intermediate buffers are freed once consumed; outputs exact.
        import mimarsinan.chip_simulation.value_run.value_flow as vf

        flow, fused = _identity_flow_and_fused()
        seen = {"calls": 0, "n": 0}
        # Neural stages decref by source id, compute ops by producer count
        # (op_source_counts); every stage must still decref exactly once, so
        # the count spans BOTH seams.
        original = vf.decref_consumers
        original_op = vf.decref_op_consumers

        def spy(buf, remaining, src_ids, **kw):
            original(buf, remaining, src_ids, **kw)
            seen["calls"] += 1
            seen["n"] = len(buf)

        def spy_op(buf, remaining, op, **kw):
            original_op(buf, remaining, op, **kw)
            seen["calls"] += 1
            seen["n"] = len(buf)

        vf.decref_consumers = spy
        vf.decref_op_consumers = spy_op
        try:
            x = torch.randn(3, 8)
            with torch.no_grad():
                got = flow(x)
                want = fused.double()(x.double())
        finally:
            vf.decref_consumers = original
            vf.decref_op_consumers = original_op
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)
        n_stages = len(flow.hybrid_mapping.stages)
        assert seen["calls"] == n_stages
        assert seen["n"] < n_stages  # consumed nodes were freed

    def test_upload_memo_shares_tensors_for_shared_arrays(self):
        # [F2] cores sharing one deduped ndarray share ONE device tensor —
        # observed in flight on the upload returns (the prepared segments die
        # with the per-forward scope).
        from unit.chip_simulation.value_upload_probe import probe_uploads

        hybrid = _token_bank_hybrid()
        flow = ValueHybridCoreFlow(hybrid, dtype=torch.float64)
        with probe_uploads(flow, keep=True) as probe:
            with torch.no_grad():
                flow(torch.randn(2, 20))
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        head_cores = neural[0].hard_core_mapping.cores
        # Padded grids materialize transiently, so the ndarray they share is
        # the stored bank payload behind one content key.
        assert head_cores[0].core_matrix is None
        assert head_cores[0].core_matrix_key() == head_cores[1].core_matrix_key()
        assert (
            head_cores[0].matrix_placements[0].source
            is head_cores[1].matrix_placements[0].source
        )
        uploaded = {id(core): tensor for core, tensor in probe.kept}
        assert uploaded[id(head_cores[0])] is uploaded[id(head_cores[1])]
