"""[U-LIVE] The dead-read removal's unit is the neuron (the window unit's check, m-1).

A dead read that covers exactly one neuron tile of its producer left that tile unread at the declared geometry, where
the tile is a core of its own and was removed, but not at the unbounded geometry, where the producer is one core whose
other neurons are read: the static check's ``ir-tiling`` then named neurons one IR holds and the other does not (a
wrong certificate of a correct program). The rule: a neuron that no node and no output reads is marked through the
IR's own column compaction (``pruned_col_mask``), and a whole core is removed only where every neuron of it is unread,
so both geometries hold the same neurons. The check's two forms (the dead columns aligned with the producer's second
tile, and across both tiles) hold ``ir-tiling`` at both geometries with equal outputs.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg

from mimarsinan.certification.program_check.report import ConditionResult
from mimarsinan.certification.program_check.unbounded import check_ir_tiling
from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values
from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from spiking_points import declaring_fire_while_integrating, point_of_mode
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.torch_mapping.converter import convert_torch_model

T = 8
#: the dead columns of A (8 neurons, two tiles of 4 at the declared geometry): aligned with its second tile, or across
VARIANTS = {"aligned": [4, 5, 6, 7], "unaligned": [2, 3, 4, 5]}


class Form(nn.Module):
    """e over the input (subsumed), A (8 neurons) and B (3) over e, C over cat(B, A) with A's ``dead`` neurons read at
    zero; the output cat(B, C)."""

    def __init__(self, seed: int, dead):
        super().__init__()
        g = torch.Generator().manual_seed(500 + seed)
        self.e, self.a, self.b, self.c = nn.Linear(4, 4), nn.Linear(4, 8), nn.Linear(4, 3), nn.Linear(11, 2)
        with torch.no_grad():
            for lin in (self.e, self.a, self.b, self.c):
                lin.weight.copy_((torch.rand(lin.weight.shape, generator=g) * 1.6 - 0.5) / lin.in_features ** 0.5)
                lin.bias.copy_(torch.rand(lin.bias.shape, generator=g) * 0.2 - 0.05)
            self.c.weight[:, [3 + k for k in dead]] = 0.0
        self.r = nn.ReLU()

    def forward(self, x):
        e = self.r(self.e(x))
        a, b = self.r(self.a(e)), self.r(self.b(e))
        return torch.cat([b, self.r(self.c(torch.cat([b, a], dim=1)))], dim=1)


def _step():
    cfg = {"spiking_family": "lif", "spiking_variant": "synchronized", "weight_bits": 8, "thresholding_mode": "<=",
           "firing_mode": "Default", "simulation_steps": T, "spike_generation_mode": "Uniform"}
    cfg = declaring_fire_while_integrating(cfg)  # [composed head, rule 12] U-LIF2 W16: the form fires in its window
    fold_spiking_axes(cfg)
    fold_conversion_recipe(cfg, resolve_spiking_point(cfg), explicit_keys=set(cfg))
    step = SoftCoreMappingStep.__new__(SoftCoreMappingStep)
    step.pipeline = SimpleNamespace(config=cfg)
    return step, DeploymentPlan.resolve(cfg), cfg


def _program(seed, dead, max_neurons, prune=False):
    model = Form(seed, dead).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model, (4,), 5, device="cpu", encoding_layer_placement="subsume").eval()
    perceptrons = list(flow.get_perceptrons())
    for p in perceptrons:
        p.set_activation_scale(torch.tensor(1.0))
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    repr_.assign_perceptron_indices()
    step, plan, cfg = _step()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=2048, max_neurons=max_neurons).map(repr_)
    with contextlib.redirect_stdout(io.StringIO()):
        enabled, record = step._apply_depth_balancing_relays(ir, plan)
    quantize_ir_graph(ir, 8, weight_quantization=False)
    IRLatency(ir).calculate()
    if prune:  # IR pruning where Soft Core Mapping runs it (after quantization and latencies)
        ir = prune_ir_graph(ir, simulation_steps=hg.T, spiking_mode="lif")
    step._require_cycle_windows(ir, plan, bool(enabled))
    declare_host_kernel_evaluation(ir, "per_sample")
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    program = hg.Program("tiling", (4,), flow, repr_, ir, hybrid, perceptrons, [1.0] * len(perceptrons), False)
    return ir, dict(record), program, cfg


@pytest.mark.parametrize("prune", [False, True], ids=["unpruned", "ir_pruning"])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_both_geometries_hold_the_same_neurons_and_read_equal(variant, seed, prune):
    tiled, record_t, program_t, cfg = _program(seed, VARIANTS[variant], 4, prune)
    unbounded, _record_u, program_u, _cfg = _program(seed, VARIANTS[variant], 2048, prune)
    result = ConditionResult(condition="ir-tiling", status="checked")
    declaration = SimpleNamespace(windowed=bool(cfg.get("lif_per_hop_retiming")), timed=True, dithered_encode=False)
    check_ir_tiling(tiled, unbounded, declaration=declaration, result=result)
    assert result.violation_count == 0, [str(v)[:300] for v in result.violations[:3]]
    assert record_t["dead_reads_off"] > 0
    torch.manual_seed(9_100 + seed)
    x = 2.0 * torch.rand(8, 4)
    outs = []
    for program in (program_t, program_u):
        dflow = hg.deployed_flow(program, "windowed_lif")
        with torch.no_grad():
            outs.append(executor_output_values(dflow, dflow(x)).double())
    assert torch.equal(outs[0], outs[1])


def test_a_neuron_no_one_reads_is_marked_and_its_core_stays():
    """At the unbounded geometry A stays one core with its unread neurons marked compacted (the IR's own column
    compaction), so the deployed program holds exactly the neurons the declared geometry holds."""
    unbounded, record, _program_u, _cfg = _program(0, VARIANTS["aligned"], 2048)
    a = [n for n in unbounded.nodes if isinstance(n, NeuralCore) and n.get_output_count() == 8]
    assert len(a) == 1
    assert list(np.flatnonzero(a[0].pruned_col_mask)) == VARIANTS["aligned"]
    assert record["unread_neurons_marked"] == 4 and record["unread_cores_removed"] == 0
