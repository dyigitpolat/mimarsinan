"""§7 rows 6 and 10: the NF twin folds the SAME kernel as the deployment.

Under ``firing_granularity='per_event'`` the NF decomposes its charge through
the mapper's own ``get_effective_weight`` and drives ``lif_serial_fold`` per
cycle, so NF window counts and NF per-cycle RASTERS must equal the packed
hard-core executor's at atol=0. Fold-invariant mapping transforms (output
tiling, identity relays) are admitted and stay exact; transforms that
re-threshold a partial sum refuse.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.models.nn.activations.lif_serial import SerialFoldSlot
from mimarsinan.models.spiking.serial import (
    SerialDecompositionMismatchError,
    SerialFoldUnsupportedError,
)
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

from .test_serial_executors import (
    PER_EVENT_UNBOUNDED,
    T,
    build_chain,
    build_flow,
    run_counts,
)


class _NFModel(nn.Module):
    """The mapper repr as a torch module the NF walk can drive."""

    def __init__(self, repr_, perceptrons):
        super().__init__()
        self.preprocessor = nn.Identity()
        self._repr = repr_
        self.hops = nn.ModuleList(perceptrons)

    def get_mapper_repr(self):
        return self._repr

    def get_perceptrons(self):
        return list(self.hops)

    def forward(self, x):
        return self._repr(x)


def _nf_model(repr_):
    return _NFModel(repr_, list(repr_.get_perceptrons())).eval()


def _nf_rates(repr_, x, *, soma_law=PER_EVENT_UNBOUNDED):
    """The NF output in the WIRE domain the deployed executor decodes to:
    the NF walk returns value-domain magnitudes (rate x the output theta)."""
    model = _nf_model(repr_)
    with torch.no_grad():
        out = chip_aligned_segment_forward(model, x, T, soma_law=soma_law)
    scale = torch.as_tensor(
        model.get_perceptrons()[-1].activation_scale, dtype=out.dtype)
    return (out / scale).to(torch.float64)


def _hcm_rates(hybrid, x, *, law=PER_EVENT_UNBOUNDED):
    flow = build_flow(hybrid, law=law, packed=True)
    out, _ = run_counts(flow, x)
    return out / float(T)


def _sample(n: int = 4):
    torch.manual_seed(3)
    return torch.rand(n, 8) * 0.9


def test_the_gate_arms_for_the_point_through_is_streamed_lif():
    """The per-event point is streamed by construction, so the exact
    (atol=0) NF↔SCM gate arms with no edit to its predicate."""
    from mimarsinan.pipelining.core.nf_scm_parity import nf_scm_parity_enabled

    contract = SpikingDeploymentContract.from_pipeline_config({
        "spiking_family": "lif", "spiking_variant": "streamed",
        "firing_granularity": "per_event", "firing_mode": "Novena",
        "thresholding_mode": "<=", "simulation_steps": T,
    })
    assert contract.soma_law().is_per_event
    assert contract.is_streamed_lif() is True
    assert nf_scm_parity_enabled(contract) is True


def test_nf_equals_packed_hcm_under_the_point():
    repr_, hybrid = build_chain()
    x = _sample()
    nf = _nf_rates(repr_, x)
    hcm = _hcm_rates(hybrid, x)
    assert torch.equal(nf, hcm.to(torch.float64))


def test_nf_equals_packed_hcm_with_output_tiling_and_a_relay():
    """§7 row 10: output tiling is per-neuron disjoint and an identity relay
    maps k events to k spikes, so both are fold-invariant and stay EXACT."""
    repr_, hybrid = build_chain(n_cores=16, max_neurons=3, relay=True)
    tiles = max(
        len(stage.hard_core_mapping.cores)
        for stage in hybrid.stages if stage.hard_core_mapping is not None
    )
    assert tiles >= 2, "fixture must actually tile a perceptron across cores"
    x = _sample()
    assert torch.equal(
        _nf_rates(repr_, x),
        _hcm_rates(hybrid, x).to(torch.float64),
    )


def test_the_witness_multi_spikes_and_the_gate_would_see_a_theta_mutation():
    """Teeth: perturb the deployed threshold and the atol=0 comparison must
    go RED. A gate that stays green under a mutated theta proves nothing."""
    repr_, hybrid = build_chain()
    x = _sample()
    nf = _nf_rates(repr_, x)
    assert torch.equal(nf, _hcm_rates(hybrid, x).to(torch.float64))
    for stage in hybrid.stages:
        if stage.hard_core_mapping is not None:
            for core in stage.hard_core_mapping.cores:
                core.threshold = float(core.threshold) * 2.0
    assert not torch.equal(nf, _hcm_rates(hybrid, x).to(torch.float64))


def test_the_nf_raster_equals_the_deployed_raster_at_atol_zero():
    """The per-CYCLE multiplicity is the load-bearing new degree of freedom:
    two runs with equal window counts but different rhythm are different
    computations at the next hop, so the gate compares rasters, not only
    window counts. The fused segment publishes its OWN output raster, which
    is the last hop's emissions in producer-local time."""
    repr_, hybrid = build_chain()
    x = _sample(2)
    nf_raster = _capture_nf_rasters(repr_, x)
    hcm_raster = _capture_hcm_rasters(hybrid, x)
    assert nf_raster and hcm_raster
    peak = max(float(r.max()) for r in nf_raster)
    assert peak >= 2.0, f"degenerate witness: peak per-cycle emission {peak}"
    nf_out, hcm_out = nf_raster[-1], hcm_raster[-1]
    assert nf_out.shape == hcm_out.shape
    assert torch.equal(nf_out, hcm_out), "NF raster differs from the deployment"
    # Window counts agreeing is strictly weaker than the raster agreeing.
    assert torch.equal(nf_out.sum(dim=0), hcm_out.sum(dim=0))


def test_a_multi_source_hop_refuses_instead_of_guessing_the_slot_order():
    repr_, _ = build_chain()
    model = _nf_model(repr_)
    lif = model.get_perceptrons()[1].activation
    from mimarsinan.spiking.segment_policy_lif_serial import _arm_serial_slot

    with pytest.raises(SerialFoldUnsupportedError, match="ONE upstream event"):
        _arm_serial_slot(PER_EVENT_UNBOUNDED, model.get_perceptrons()[1], lif,
                         [None, None])


def _slot(weight: torch.Tensor):
    return SerialFoldSlot(
        soma_law=PER_EVENT_UNBOUNDED, weight=weight, bias=None, theta=1.0,
        membrane_init=0.0,
    )


def test_a_decomposition_in_the_wrong_order_RAISES_and_never_folds():
    """The NF-order == mapper-order contract is the twin's whole warrant, so
    it must survive ``python -O``: a bare assert would vanish there and the
    twin would silently fold a different event order than the deployment."""
    weight = torch.tensor([[1.0, 2.0, 4.0]])
    events = torch.tensor([[1.0, 1.0, 0.0]])
    slot = _slot(weight.flip(-1))
    slot.feed(events)
    fused = torch.nn.functional.linear(events, weight)
    with pytest.raises(SerialDecompositionMismatchError, match="feature order"):
        slot.run_cycle(fused, 1.0)


def test_a_non_ascending_canonical_order_refuses_the_pass_through_feed(monkeypatch):
    """``feed`` hands the NF feature order straight through as the slot order;
    that is only sound while the canonical order IS ascending."""
    from mimarsinan.models.nn.activations import lif_serial

    monkeypatch.setattr(
        lif_serial, "canonical_slot_order", lambda n: list(reversed(range(n))))
    slot = _slot(torch.tensor([[1.0, 2.0, 4.0]]))
    with pytest.raises(SerialDecompositionMismatchError,
                       match="canonical_slot_order"):
        slot.feed(torch.tensor([[1.0, 1.0, 0.0]]))


def _capture_nf_rasters(repr_, x):
    """Per-hop (T, B, n) NF emission MULTIPLICITY trains, in hop order."""
    model = _nf_model(repr_)
    captured: dict = {}

    def make_hook(index, perceptron):
        def hook(_module, _inp, out):
            scale = torch.as_tensor(
                perceptron.activation_scale, device=out.device, dtype=out.dtype)
            flat = (out / scale.clamp(min=1e-12)).detach().reshape(
                out.shape[0], -1)
            captured.setdefault(index, []).append(flat.to(torch.float64))
        return hook

    handles = [
        p.activation.register_forward_hook(make_hook(i, p))
        for i, p in enumerate(model.get_perceptrons())
    ]
    try:
        with torch.no_grad():
            chip_aligned_segment_forward(
                model, x, T, soma_law=PER_EVENT_UNBOUNDED)
    finally:
        for handle in handles:
            handle.remove()
    return [
        torch.stack(captured[i], dim=0)
        for i in sorted(captured) if len(captured[i]) == T
    ]


def _capture_hcm_rasters(hybrid, x):
    """Per-neural-stage (T, B, n) deployed output rasters, in stage order."""
    flow = build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True)
    rasters: list = []
    flow.stage_raster_recorder = (
        lambda stage, raster: rasters.append(raster.detach().clone())
    )
    with torch.no_grad():
        flow(x)
    flow.stage_raster_recorder = None
    return [r.to(torch.float64) for r in rasters]
