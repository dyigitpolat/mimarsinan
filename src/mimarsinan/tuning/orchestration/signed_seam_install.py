"""[sigma-in-the-op] Pre-training signed-seam install: quantile sigma folded into the armed op, quantile kappa currencies, consumer bakes (memo sec.10f)."""

from __future__ import annotations

import torch

from mimarsinan.mapping.support.tensor_stats import safe_quantile
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
    sigma = float(safe_quantile(neg, quantile)) if bool((neg > 0).any()) else 0.0
    kappa = float(safe_quantile(v + sigma, quantile).clamp(min=1e-6))
    return sigma, kappa


def _prearm_marked_value_ops(repr_, boundary_table) -> int:
    """[B4 completion, calculus §15.7] Marked-but-unarmed wire-value ops
    (unity-gauge chains — the entry-1 stem class) get wrap slots at the
    PASS-THROUGH currency so sigma can transport: unit per-source scales,
    boundary-table kappa out — currency-inert until the covers lift it."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    armed = 0
    for node in repr_.execution_order():
        if not (
            isinstance(node, ComputeOpMapper)
            and getattr(node, "is_wire_value_op", False)
            and node.per_source_scales is None
        ):
            continue
        kappa = float(boundary_table.get(node, 1.0))
        node.per_source_scales = [torch.ones(1) for _ in node._sources_list]
        node.output_scale = torch.tensor([kappa])
        armed += 1
    return armed


def _bake_walk_feasible(node, consumers) -> bool:
    """Dry-run of the σ consumer walk: True iff every reachable host consumer
    is compensable (bias carrier / shift-invariant / equivariant pass-through).
    Non-compensable chains (the ViT stem's ``cat`` class) are SKIPPED by the
    installer — left trained-clamp — instead of crashing the install
    (σ-scope law: σ belongs to producer→entry edges it can actually reach)."""
    import torch.nn as _nn

    from mimarsinan.mapping.support.negative_boundary import (
        _is_host_node,
        boundary_consumers,
    )

    frontier = list(boundary_consumers(node, consumers))
    while frontier:
        consumer = frontier.pop()
        if not _is_host_node(consumer):
            continue
        module = getattr(consumer, "module", None)
        if isinstance(module, _nn.Linear) and module.bias is not None:
            continue
        if (
            isinstance(module, _nn.MultiheadAttention)
            and module.in_proj_weight is not None
            and module.in_proj_bias is not None
        ):
            continue
        if isinstance(module, _nn.LayerNorm):
            continue
        if _scalar_shift_equivariant(module):
            frontier.extend(boundary_consumers(consumer, consumers))
            continue
        return False
    return True


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
        read_boundary_out_scales,
        stamped_input_boundary_scale,
        verify_boundary_currency_coherence,
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
    _prearm_marked_value_ops(
        repr_,
        read_boundary_out_scales(
            repr_, input_data_scale=stamped_input_boundary_scale(repr_),
        ),
    )
    installed = 0
    skipped: list = []
    for node, sample in samples.items():
        # ONLY armed ops: the wrapper carries the offset into every deployed
        # representation; an unarmed op's offset would exist in the plain
        # forward alone (train/deploy split).
        if not (
            isinstance(node, ComputeOpMapper)
            and getattr(node, "is_wire_value_op", False)
            and node.per_source_scales is not None
        ):
            continue
        if node.output_value_offset is not None:
            continue  # idempotent: one install owns the seam
        if not _bake_walk_feasible(node, consumers):
            skipped.append(getattr(node, "name", None) or type(node.module).__name__)
            continue
        sigma, kappa = _signed_seam_quantiles(sample, quantile=quantile)
        node.output_value_offset = torch.tensor(float(sigma))
        node.boundary_traffic_scale = float(kappa)
        installed += 1

    if skipped:
        print(
            f"[SIGNED-SEAM] {len(skipped)} armed op(s) skipped "
            f"(bake-infeasible consumer chains, left trained-clamp): {skipped}",
            flush=True,
        )
    # One re-propagation whenever ANYTHING is armed (pre-arm included): the
    # wrapper s_out, the weight fold, the consumers' per_source currencies
    # [calculus §16.6], and the entry currencies agree (one scale, one writer).
    input_data_scale = ResolvedWorkloadProfile.from_config(
        pipeline_config
    ).input_data_scale
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
    if not installed:
        verify_boundary_currency_coherence(model)
        return 0

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
        frontier = list(boundary_consumers(node, consumers))
        while frontier:
            consumer = frontier.pop()
            if not _is_host_node(consumer):
                perceptron = _perceptron_of(consumer)
                if perceptron is not None:
                    apply_negative_shift_bias(
                        perceptron, torch.tensor(sigma_v / max(s_out, 1e-12)),
                    )
                continue
            module = getattr(consumer, "module", None)
            if _bake_shift_into_host_bias(module, sigma_v):
                continue
            if isinstance(module, nn.LayerNorm):
                # Shift-INVARIANT (mean-subtracting): a scalar sigma vanishes
                # here — the walk ends with nothing to bake.
                continue
            if _scalar_shift_equivariant(module):
                # f(v + c) = f(v) + c: sigma passes through unchanged.
                frontier.extend(boundary_consumers(consumer, consumers))
                continue
            raise NotImplementedError(
                "sigma-in-the-op: a lifted seam feeds a host consumer with no "
                f"bias carrier ({type(module).__name__}); extend the bake or "
                "leave this op unlifted."
            )
    # The install seam is the last currency writer: certify one-writer
    # coherence LOUD before training adapts to the stamps (calculus §11.2).
    verify_boundary_currency_coherence(model)
    return installed


def _bake_shift_into_host_bias(module, sigma_v: float) -> bool:
    """Subtract the shift response sigma.W.sum(dim=1) from the module's bias so
    f_baked(v + sigma) == f(v); True iff the module is a bias carrier."""
    if isinstance(module, nn.Linear) and module.bias is not None:
        with torch.no_grad():
            module.bias.sub_(sigma_v * module.weight.sum(dim=1))
        return True
    if (
        isinstance(module, nn.MultiheadAttention)
        and module.in_proj_weight is not None
        and module.in_proj_bias is not None
    ):
        # Self-attention: q=k=v arrive from the same lifted seam (the graph
        # dedupes the edges), so ONE packed-in_proj bake covers all three.
        with torch.no_grad():
            module.in_proj_bias.sub_(
                sigma_v * module.in_proj_weight.sum(dim=1)
            )
        return True
    return False


_SHIFT_EQUIVARIANT_ADAPTER_FNS = frozenset({
    # f(v + c) = f(v) + c for a scalar c on the shifted input; "add" passes a
    # single shifted operand's c through; "getitem" selects; "cat" stays OUT
    # (a partial-slice shift is not bias-compensable downstream).
    "mean", "amax", "amin", "flatten", "reshape", "permute", "transpose",
    "add", "getitem",
})


def _scalar_shift_equivariant(module) -> bool:
    """Whether ``f(v + c) = f(v) + c`` for a scalar c: pools/relays and the
    mean/select adapter family qualify; ``sum`` does NOT (it scales c by N)."""
    from mimarsinan.mapping.support.compute_modules import ComputeAdapter

    if isinstance(module, ComputeAdapter):
        return getattr(module.fn, "__name__", "") in _SHIFT_EQUIVARIANT_ADAPTER_FNS
    return isinstance(module, (
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.Identity, nn.Flatten,
    ))
