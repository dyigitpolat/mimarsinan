"""SCM ↔ HCM parity: same code, same data ⇒ same output.

Both steps now build a ``HybridHardCoreMapping`` from the same IR graph
via ``build_hybrid_hard_core_mapping`` and run that mapping through
``SpikingHybridCoreFlow``. Functional equivalence between SCM and HCM
should therefore be **bit-identical** in every spiking_mode (lif, ttfs,
ttfs_quantized) — anything else means the sharing has regressed.

We also pin down that the shared subsample helper produces the same
indices for any (total, seed, max) triple.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
    platform_constraints,  # fixture
)
from mimarsinan.chip_simulation.test_subsample import (
    compute_test_subsample_indices,
)
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


@pytest.fixture
def _tiny_setup(platform_constraints):
    """Build an IR graph + hybrid mapping the same way SCM and HCM do."""
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["spike_generation_mode"] = "Uniform"
    cfg["thresholding_mode"] = "<"
    cfg["weight_quantization"] = False
    cfg["weight_bits"] = 8
    cfg["simulation_steps"] = 4

    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.normalization = torch.nn.Identity()

    pipeline = MockPipeline(config=cfg)
    pipeline.seed("fused_model", model, step_name="Normalization Fusion")
    pipeline.seed(
        "platform_constraints_resolved",
        platform_constraints,
        step_name="Model Configuration",
    )

    scm = SoftCoreMappingStep(pipeline)
    scm.name = "Soft Core Mapping"
    pipeline.prepare_step(scm)
    scm.run()

    ir_graph = pipeline.cache["Soft Core Mapping.ir_graph"]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=platform_constraints["cores"],
        allow_neuron_splitting=False,
        allow_scheduling=False,
        allow_coalescing=False,
    )
    return cfg, ir_graph, hybrid


def _build_flow(cfg, hybrid):
    return SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid,
        int(cfg["simulation_steps"]),
        None,
        cfg["firing_mode"],
        cfg["spike_generation_mode"],
        cfg["thresholding_mode"],
        spiking_mode=cfg["spiking_mode"],
    )


def test_scm_and_hcm_share_simulation_path(_tiny_setup):
    """SCM and HCM both go through SpikingHybridCoreFlow on the same
    HybridHardCoreMapping; two independent constructions must produce
    identical outputs on the same input.
    """
    cfg, _ir_graph, hybrid = _tiny_setup

    flow_scm = _build_flow(cfg, hybrid).eval()
    flow_hcm = _build_flow(cfg, hybrid).eval()

    torch.manual_seed(0)
    x = torch.randn(2, *cfg["input_shape"])

    with torch.no_grad():
        out_scm = flow_scm(x)
        out_hcm = flow_hcm(x)

    assert torch.equal(out_scm, out_hcm), (
        "SCM and HCM are now expected to be bit-identical (they run the "
        "same SpikingHybridCoreFlow on the same hybrid mapping). "
        f"Max abs diff: {(out_scm - out_hcm).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("spiking_mode", ["lif", "ttfs", "ttfs_quantized"])
def test_scm_and_hcm_parity_across_spiking_modes(_tiny_setup, spiking_mode):
    """Equivalence must hold in every spiking mode."""
    cfg, _ir_graph, hybrid = _tiny_setup
    cfg = dict(cfg)
    cfg["spiking_mode"] = spiking_mode
    if spiking_mode != "lif":
        cfg["firing_mode"] = "TTFS"
        cfg["spike_generation_mode"] = "TTFS"
        cfg["thresholding_mode"] = "<="

    flow_a = _build_flow(cfg, hybrid).eval()
    flow_b = _build_flow(cfg, hybrid).eval()

    torch.manual_seed(0)
    x = torch.randn(2, *cfg["input_shape"])
    with torch.no_grad():
        out_a = flow_a(x)
        out_b = flow_b(x)

    assert torch.equal(out_a, out_b), (
        f"SCM/HCM parity broken for spiking_mode={spiking_mode!r}; "
        f"max abs diff: {(out_a - out_b).abs().max().item():.6e}"
    )


def test_shared_subsample_indices_are_deterministic():
    """SCM/HCM/Sim subsample via :func:`compute_test_subsample_indices`;
    the same (total, seed, max) must yield the same indices in the same
    order so the three accuracy numbers are evaluated on the same
    samples.
    """
    a = compute_test_subsample_indices(total_samples=1000, seed=7, max_samples=50)
    b = compute_test_subsample_indices(total_samples=1000, seed=7, max_samples=50)
    assert a == b
    assert len(a) == 50
    assert len(set(a)) == 50

    c = compute_test_subsample_indices(total_samples=1000, seed=8, max_samples=50)
    assert c != a, "different seeds must produce different subsamples"


def test_shared_subsample_indices_full_range_when_uncapped():
    """``max_samples <= 0`` or ``>= total`` returns the full range so
    every consumer can iterate the result unconditionally."""
    indices = compute_test_subsample_indices(total_samples=10, seed=0, max_samples=0)
    assert indices == list(range(10))

    indices = compute_test_subsample_indices(total_samples=10, seed=0, max_samples=10)
    assert indices == list(range(10))

    indices = compute_test_subsample_indices(total_samples=10, seed=0, max_samples=100)
    assert indices == list(range(10))
