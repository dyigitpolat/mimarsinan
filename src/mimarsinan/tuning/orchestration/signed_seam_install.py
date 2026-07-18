"""[sigma-in-the-op] Pre-training signed-seam install: quantile sigma folded into the armed op, quantile kappa currencies, consumer bakes (memo sec.10f)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_active


def ensure_offload_negative_boundary(model, trainer, pipeline_config) -> None:
    """[sigma-in-the-op, opt-in] Install the signed-seam lift + quantile
    currencies at the AQ install seam (offload only), pre-training — memo
    sec.10f. Post-training sigma stays banned (the scope law); the SCM skip
    protects the installed composite as the trained function."""
    if not lif_exact_qat_active(pipeline_config):
        return
    if not bool(pipeline_config.get("lif_aq_negative_boundary", False)):
        return
    from mimarsinan.torch_mapping.encoding_layers import (
        encoder_deploys_as_staircase_hop,
    )

    placement = str(pipeline_config.get("encoding_layer_placement", "subsume"))
    if encoder_deploys_as_staircase_hop(placement):
        return
    install_signed_seam_offsets(model, trainer, pipeline_config)


def _signed_seam_quantiles(
    values: torch.Tensor, *, quantile: float,
) -> tuple[float, float]:
    """(sigma, kappa) for a signed seam: sigma = q-quantile of the negative
    magnitude (0 when none), kappa = q-quantile of the SHIFTED values —
    capacity without resolution loss (kappa is a quantile, NEVER the max:
    memo sec.10f, the full-width cover refutation)."""
    v = values.reshape(-1).float()
    neg = (-v).clamp(min=0.0)
    sigma = float(torch.quantile(neg, quantile)) if bool((neg > 0).any()) else 0.0
    kappa = float(torch.quantile(v + sigma, quantile).clamp(min=1e-6))
    return sigma, kappa


def install_signed_seam_offsets(
    model, trainer, pipeline_config, *, quantile: float = 0.99,
) -> int:
    """[sigma-in-the-op, memo sec.10f] Fold the signed-seam lift INTO each
    armed op's function BEFORE training: output_value_offset = sigma-quantile,
    seam currency = kappa-quantile of the shifted range, consumers baked
    (entries via the effective-bias identity; biased host Linears directly)
    so the install is value-preserving in-range and the QAT trains the exact
    deployed composition with the negative band alive. Returns installs."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
    from mimarsinan.mapping.support.negative_boundary import (
        _is_host_node,
        _perceptron_of,
        boundary_consumers,
    )
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias
    from mimarsinan.mapping.support.per_source_scales import (
        compute_per_source_scales,
    )
    from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
    from mimarsinan.spiking.scale_aware_boundaries import (
        propagate_boundary_input_scales,
    )
    from mimarsinan.spiking.segment_forward import (
        AnalyticalSegmentPolicy,
        SegmentForwardDriver,
    )

    batches = [x for x, _ in trainer.iter_validation_batches(2)]
    if not batches:
        return 0
    device = pipeline_config["device"]
    calibration_x = torch.cat(batches, dim=0).to(device)
    repr_ = model.get_mapper_repr()
    driver = SegmentForwardDriver(
        repr_, int(pipeline_config["simulation_steps"]), AnalyticalSegmentPolicy(),
    )
    samples: dict = {}
    with torch.no_grad():
        driver(calibration_x, compute_sample_recorder=samples)

    consumers = repr_.consumer_map()
    installed = 0
    for node, sample in samples.items():
        if not (
            isinstance(node, ComputeOpMapper)
            and getattr(node, "is_wire_value_op", False)
        ):
            continue
        if node.output_value_offset is not None:
            continue  # idempotent: one install owns the seam
        sigma, kappa = _signed_seam_quantiles(sample, quantile=quantile)
        node.output_value_offset = torch.tensor(float(sigma))
        node.boundary_traffic_scale = float(kappa)
        installed += 1

    if not installed:
        return 0
    # One re-propagation: the wrapper s_out, the weight fold, and the entry
    # currencies agree on the lifted kappa (one scale to both walks).
    input_data_scale = ResolvedWorkloadProfile.from_config(
        pipeline_config
    ).input_data_scale
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)

    # Consumer bake AFTER the currencies settle: entry charge shift is
    # W_eff . (sigma / s_out_final) (the T4 identity, buffer units).
    for node in repr_.execution_order():
        if not (
            isinstance(node, ComputeOpMapper)
            and node.output_value_offset is not None
        ):
            continue
        sigma_v = float(node.output_value_offset)
        s_out = float(
            torch.as_tensor(node.output_scale).float().mean()
        ) if node.output_scale is not None else 1.0
        for consumer in boundary_consumers(node, consumers):
            if not _is_host_node(consumer):
                perceptron = _perceptron_of(consumer)
                if perceptron is not None:
                    apply_negative_shift_bias(
                        perceptron, torch.tensor(sigma_v / max(s_out, 1e-12)),
                    )
                continue
            module = getattr(consumer, "module", None)
            if isinstance(module, nn.Linear) and module.bias is not None:
                with torch.no_grad():
                    module.bias.sub_(sigma_v * module.weight.sum(dim=1))
                continue
            raise NotImplementedError(
                "sigma-in-the-op: a lifted seam feeds a host consumer with no "
                f"bias carrier ({type(module).__name__}); extend the bake or "
                "leave this op unlifted."
            )
    return installed
