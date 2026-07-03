"""NF↔SCM per-neuron parity gate (rung 1 ↔ rung 2) for analytic TTFS schedules."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

_DEBUG_ENV = "MIMARSINAN_NF_SCM_PARITY_DEBUG"


class NfScmParityError(AssertionError):
    """Per-neuron divergence between the analytical NF and the identity mapping.

    ``mismatch_fraction`` is the measured per-neuron flip fraction (``None`` for
    decision-level and structural raises that have no such scalar).
    """

    def __init__(self, message: str, *, mismatch_fraction: float | None = None):
        super().__init__(message)
        self.mismatch_fraction = mismatch_fraction


def _unify_model_device(model):
    """Place the WHOLE model on one device and return it (``None`` if param-less).

    Prevents a cross-device matmul when the mapper graph left modules on different
    devices; prefers a CUDA device when the model holds any CUDA parameter.
    """
    params = list(model.parameters())
    if not params:
        return None
    device = next(
        (p.device for p in params if p.device.type == "cuda"), params[0].device,
    )
    model.to(device)
    return device


def nf_scm_parity_enabled(contract: Any) -> bool:
    """Whether this mode's NF can be held per-neuron against the deployed executor.

    Continuous ttfs gets the per-neuron gate; cascaded gets a decision-level gate;
    the floor+half-step-bias convention modes (ttfs_quantized, synchronized floor-collapse) are excluded.
    """
    if contract.uses_ttfs_floor_ceil_convention():
        return False
    if contract.is_cascaded():
        return True
    return contract.training_forward_kind() == "analytical_staircase"


def assert_nf_scm_parity_or_raise(
    pipeline,
    model,
    ir_graph,
    samples: torch.Tensor,
    *,
    atol: float = 1e-9,
    max_mismatch_fraction: float = 0.0,
) -> float:
    """Compare per-neuron NF activations against the identity-mapped contract run.

    Both sides are compared in the normalized [0, 1] TTFS domain; ``max_mismatch_fraction``
    budgets the honest mapping-level wire residual. Returns the measured fraction.
    """
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.pipelining.core.simulation_factory import (
        build_identity_mapping_for_pipeline,
    )

    contract = SpikingDeploymentContract.from_pipeline_config(pipeline.config)
    if contract.is_synchronized() and "forward" in getattr(model, "__dict__", {}):
        raise NfScmParityError(
            "NF↔SCM parity: synchronized NF must run the class-level analytical "
            "forward, but the model carries an instance forward override (a "
            "pre-schedule-aware-tuner cascade forward from a legacy cache?). "
            "Re-run TTFS Cycle Fine-Tuning or strip model.__dict__['forward']."
        )
    identity_mapping = build_identity_mapping_for_pipeline(
        ir_graph, pipeline_config=pipeline.config,
    )

    from mimarsinan.mapping.pruning import derive_deployed_neuron_survival

    nf = _capture_nf_normalized(model, samples)
    # Project the NF onto neurons actually deployed after pruning (the pruned ir_graph is the survival authority; no-op when nothing was pruned).
    nf = derive_deployed_neuron_survival(ir_graph).project(nf)
    scm = _collect_scm_normalized(identity_mapping, model, samples, contract)

    shared = sorted(set(nf) & set(scm))
    if not shared:
        raise NfScmParityError(
            "NF↔SCM parity: no comparable perceptrons (no on-chip cores carry "
            "a perceptron_index; run assign_perceptron_indices before mapping)"
        )

    debug = os.environ.get(_DEBUG_ENV) == "1"
    if debug:
        print(
            f"[nf_scm_parity] nf-only perceptrons: {sorted(set(nf) - set(scm))} "
            f"scm-only: {sorted(set(scm) - set(nf))}"
        )

    mismatches, total, worst = compare_normalized_records(
        nf, scm, atol=atol, debug=debug,
    )

    fraction = mismatches / max(total, 1)
    if fraction > max_mismatch_fraction:
        d, pi, s_idx, rank, nf_v, scm_v = worst
        raise NfScmParityError(
            f"NF↔SCM per-neuron parity failed: {mismatches}/{total} values "
            f"differ beyond atol={atol} (fraction {fraction:.4f} > budget "
            f"{max_mismatch_fraction}). Worst: perceptron {pi} sample {s_idx} "
            f"sorted-rank {rank}: nf={nf_v!r} scm={scm_v!r} (|Δ|={d!r})",
            mismatch_fraction=fraction,
        )
    return fraction


def _build_cascaded_identity_executor(pipeline, model, ir_graph):
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.models.spiking.hybrid.identity_flow import (
        build_identity_spiking_flow,
    )

    cfg = pipeline.config
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    return build_identity_spiking_flow(
        cfg["input_shape"],
        ir_graph,
        contract.simulation_steps,
        getattr(model, "preprocessor", None),
        contract.firing_mode,
        contract.spike_generation_mode,
        contract.thresholding_mode,
        spiking_mode=contract.spiking_mode,
        ttfs_cycle_schedule=contract.ttfs_cycle_schedule,
    ).eval()


def assert_cascaded_nf_scm_agreement_or_raise(
    pipeline,
    model,
    ir_graph,
    samples: torch.Tensor,
    *,
    min_agreement: float = 0.98,
) -> float:
    """Decision-level cascaded gate: NF argmax must agree with the identity-mapped executor on ``min_agreement`` of samples.

    Healthy agreement is ~1.0 (driver==executor bit-exact once bias references stay
    live); a wrong-NF-dynamics regression craters it. Returns the measured agreement.
    """
    executor = _build_cascaded_identity_executor(pipeline, model, ir_graph)
    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
        executor = executor.to(device)
    with torch.no_grad():
        nf_pred = model(samples).argmax(dim=1)
        scm_pred = executor(samples).argmax(dim=1)
    agreement = float((nf_pred == scm_pred).double().mean())
    if agreement < float(min_agreement):
        raise NfScmParityError(
            f"NF↔SCM cascaded decision agreement failed: {agreement:.4f} < "
            f"min_agreement={min_agreement} over {int(samples.shape[0])} "
            f"samples (healthy WQ tie-flip residual measures ~0.95; the "
            f"wrong-NF-dynamics incident class craters this)"
        )
    return agreement


def assert_torch_vs_deployed_sim_parity_or_raise(
    model,
    flow,
    samples: torch.Tensor,
    *,
    min_agreement: float = 0.98,
) -> float:
    """Torch↔deployed-sim parity: the NF torch forward must agree with the deployed spiking sim on ``min_agreement`` of samples.

    Healthy agreement is ~1.0 (a single WQ tie-flip per few hundred samples is the
    only expected residual); a fidelity regression craters it. Returns the measured agreement.
    """
    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
        flow = flow.to(device)
    with torch.no_grad():
        torch_pred = model(samples).argmax(dim=1)
        sim_pred = flow(samples).argmax(dim=1)
    agreement = float((torch_pred == sim_pred).double().mean())
    if agreement < float(min_agreement):
        raise NfScmParityError(
            f"torch↔deployed-sim parity failed: {agreement:.4f} < "
            f"min_agreement={min_agreement} over {int(samples.shape[0])} samples. "
            f"The deployed spiking sim diverged from the trained torch cascade — a "
            f"deployment-fidelity regression (expected residual is only the rare WQ "
            f"tie-flip, ~1 sample per few hundred)."
        )
    return agreement


def compare_normalized_records(
    nf: Dict[int, np.ndarray],
    scm: Dict[int, np.ndarray],
    *,
    atol: float,
    debug: bool = False,
):
    """Order-insensitive per-perceptron comparison: ``(mismatches, total, worst)``.

    Each (perceptron, sample) row is compared as a sorted multiset (conv core emission
    order need not match the torch flatten order); positional wiring is enforced transitively.
    """
    total = 0
    mismatches = 0
    worst = None
    for pi in sorted(set(nf) & set(scm)):
        nf_vals, scm_vals = nf[pi], scm[pi]
        if nf_vals.shape != scm_vals.shape:
            raise NfScmParityError(
                f"NF↔SCM parity: perceptron {pi} neuron-count mismatch "
                f"{nf_vals.shape} vs {scm_vals.shape}"
            )
        nf_sorted = np.sort(nf_vals, axis=1)
        scm_sorted = np.sort(scm_vals, axis=1)
        diff = np.abs(nf_sorted - scm_sorted)
        if debug:
            frac_pi = float((diff > atol).mean())
            print(
                f"[nf_scm_parity] perceptron {pi}: shape={nf_vals.shape} "
                f"mismatch={frac_pi:.4%} max|Δ|={float(diff.max()):.4f} "
                f"nf[mean={float(nf_vals.mean()):.4f}] "
                f"scm[mean={float(scm_vals.mean()):.4f}]"
            )
        total += diff.size
        bad = diff > atol
        mismatches += int(bad.sum())
        if bad.any():
            s_idx, rank = np.unravel_index(int(diff.argmax()), diff.shape)
            candidate = (
                float(diff[s_idx, rank]), pi, int(s_idx), int(rank),
                float(nf_sorted[s_idx, rank]), float(scm_sorted[s_idx, rank]),
            )
            if worst is None or candidate[0] > worst[0]:
                worst = candidate
    return mismatches, total, worst


def _capture_nf_normalized(model, samples: torch.Tensor) -> Dict[int, np.ndarray]:
    """Per-perceptron NF outputs over the batch, normalized to [0, 1]."""
    from mimarsinan.models.nn.activations.ttfs_spiking import _channel_broadcast_view

    device = _unify_model_device(model)
    if device is not None:
        samples = samples.to(device)
    perceptrons = list(model.get_perceptrons())
    captured: Dict[int, torch.Tensor] = {}

    def _make_hook(index, perceptron):
        def hook(_module, _inp, out):
            scale = torch.as_tensor(
                perceptron.activation_scale, device=out.device, dtype=out.dtype,
            )
            if scale.dim() == 0:
                normalized = out / scale.clamp(min=1e-12)
            else:
                normalized = out / _channel_broadcast_view(scale, out).clamp(min=1e-12)
            captured[index] = normalized.detach().reshape(out.shape[0], -1)
        return hook

    handles = [
        p.activation.register_forward_hook(_make_hook(i, p))
        for i, p in enumerate(perceptrons)
    ]
    try:
        with torch.no_grad():
            model(samples)
    finally:
        for handle in handles:
            handle.remove()
    return {i: v.cpu().numpy().astype(np.float64) for i, v in captured.items()}


def _collect_scm_normalized(
    identity_mapping,
    model,
    samples: torch.Tensor,
    contract,
) -> Dict[int, np.ndarray]:
    """Per-perceptron contract-runner outputs on the identity mapping.

    Cores group by ``perceptron_index`` and concatenate in IR-id order; psum partials are excluded.
    """
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract

    preprocessor = getattr(model, "preprocessor", None)
    per_sample: List[Dict[int, np.ndarray]] = []
    for i in range(samples.shape[0]):
        x = samples[i : i + 1]
        if preprocessor is not None:
            with torch.no_grad():
                x = preprocessor(x)
        x_np = x.reshape(1, -1).detach().cpu().to(torch.float64).numpy()
        run = run_ttfs_hybrid_contract(
            identity_mapping, x_np, sample_index=i, contract=contract,
        )
        per_sample.append(_group_record_by_perceptron(run.record, identity_mapping))

    grouped: Dict[int, np.ndarray] = {}
    for pi in per_sample[0]:
        grouped[pi] = np.stack([sample_vals[pi] for sample_vals in per_sample])
    return grouped


def _group_record_by_perceptron(record, identity_mapping) -> Dict[int, np.ndarray]:
    # Order a perceptron's tiles by tile_offset (perceptron_output_slice start), not ir_id: id assignment is not monotone in the slice after compaction.
    per_core: Dict[int, tuple[int, int, np.ndarray]] = {}
    for stage_index, segment in record.segments.items():
        stage = identity_mapping.stages[stage_index]
        placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
        for core_record in segment.cores:
            core_placements = placements[core_record.core_index]
            assert len(core_placements) == 1, (
                "NF↔SCM parity gate requires an identity mapping (1 placement/core)"
            )
            placement = core_placements[0]
            assert placement.get("split_group_id") is None, (
                "identity mappings must not contain neuron-split fragments"
            )
            perceptron_index = placement.get("perceptron_index")
            if perceptron_index is None or perceptron_index < 0:
                continue
            if placement.get("psum_role") not in (None, "accum"):
                continue
            values = np.asarray(
                core_record.output_activation[: core_record.n_out_used],
                dtype=np.float64,
            )
            out_slice = placement.get("perceptron_output_slice")
            tile_offset = int(out_slice[0]) if out_slice is not None else 0
            per_core[placement["ir_node_id"]] = (
                int(perceptron_index), tile_offset, values,
            )

    by_perceptron: Dict[int, list[int]] = defaultdict(list)
    for ir_id, (perceptron_index, _, _) in per_core.items():
        by_perceptron[perceptron_index].append(ir_id)
    return {
        pi: np.concatenate(
            [
                per_core[ir_id][2]
                for ir_id in sorted(ir_ids, key=lambda c: (per_core[c][1], c))
            ]
        )
        for pi, ir_ids in by_perceptron.items()
    }
