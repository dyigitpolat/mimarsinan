"""[R3/S2] Calibration-time per-channel theta promotion on matching-axis edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.deployment_contract import (
    SpikingDeploymentContract,
)
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.mapping.channel_axis_walk import (
    channel_aligned_consumer_targets,
    columns_channel_aligned,
)
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper

MIN_THETA = 1e-6

# The per-channel loading quantile: A5 in sync_deployment_exactness.md §6 —
# per-channel theta at q0.99 is the measured best single lever; never load a
# channel at the full quantile (one outlier owns the channel's grid).
PER_CHANNEL_QUANTILE_CAP = 0.99


@dataclass(frozen=True)
class PerChannelThetaReport:
    """Names of promoted / kept-scalar perceptrons (the armed-lever witness)."""

    promoted: tuple = ()
    skipped: tuple = ()


def per_channel_theta_armed(config) -> bool:
    """The ``per_channel_theta`` knob is on AND the mode's decode is
    per-channel-capable: LIF (per-channel rate decode) or synchronized TTFS
    (per-channel theta folds into effective weights; wires stay raw-grid).
    Cascaded keeps its own theta-cotrain mechanism; ttfs_quantized keeps its
    proven scalar full-quantile decode."""
    if not bool(config.get("per_channel_theta", False)):
        return False
    contract = SpikingDeploymentContract.from_pipeline_config(config)
    return is_lif(contract.spiking_mode) or contract.is_synchronized()


def _theta_promotable(perceptron) -> bool:
    """Channels-last linear non-encoding hop of >= 2 channels: the NF broadcast
    and the channel-axis walk (k=1 == last axis) are only defined for these;
    a 1-channel vector is semantically the scalar (and 1-element clamp bounds
    stay loud-rejected by contract)."""
    return (
        not getattr(perceptron, "is_encoding_layer", False)
        and isinstance(perceptron.layer, nn.Linear)
        and int(perceptron.layer.weight.shape[0]) > 1
        and int(getattr(perceptron, "output_channel_axis", 1)) == -1
    )


def _require_consumer_widths(producer, perceptron_targets, module_targets) -> None:
    out_channels = int(producer.layer.weight.shape[0])
    name = getattr(producer, "name", "<unnamed>")
    for consumer in perceptron_targets:
        if int(consumer.layer.weight.shape[1]) != out_channels:
            raise ValueError(
                f"per-channel theta for {name!r}: aligned consumer "
                f"{getattr(consumer, 'name', '<unnamed>')!r} consumes "
                f"{int(consumer.layer.weight.shape[1])} features, not the "
                f"producer's {out_channels} channels"
            )
    for module in module_targets:
        if int(module.weight.shape[1]) != out_channels:
            raise ValueError(
                f"per-channel theta for {name!r}: aligned host module consumes "
                f"{int(module.weight.shape[1])} features, not the producer's "
                f"{out_channels} channels"
            )


def eligible_per_channel_perceptrons(model) -> dict:
    """``{id(perceptron): perceptron}`` for every hop whose channel axis is
    consumed ONLY as linear feature axes (the M4 matching-axis condition), with
    perceptron consumers reached through structural mappers alone.

    A perceptron consumer past a host ComputeOp is a segment-entry boundary
    whose snap normalizer is scalar (today's parity contract) — those producers
    keep the scalar theta, exactly like weight-shared / axis-flipped hops."""
    mapper_repr = model.get_mapper_repr()
    consumers = mapper_repr.consumer_map()
    eligible: dict = {}
    for node in mapper_repr.execution_order():
        if not isinstance(node, PerceptronMapper):
            continue
        perceptron = node.perceptron
        if not _theta_promotable(perceptron):
            continue
        targets = channel_aligned_consumer_targets(
            node,
            consumers,
            consumer_predicate=columns_channel_aligned,
            structural_perceptron_paths=True,
        )
        if targets is None:
            continue
        perceptron_targets, module_targets = targets
        if not perceptron_targets and not module_targets:
            continue
        _require_consumer_widths(perceptron, perceptron_targets, module_targets)
        eligible[id(perceptron)] = perceptron
    return eligible


def per_channel_theta_vector(
    pooled_scale: float, channel_quantiles, *, min_scale: float = MIN_THETA,
) -> torch.Tensor:
    """Per-channel theta: each live channel's own quantile (floored at
    ``min_scale``); dead channels keep the pooled scalar so they stay
    representable."""
    q = torch.as_tensor(list(channel_quantiles), dtype=torch.float32)
    if q.dim() != 1 or q.numel() == 0:
        raise ValueError(
            f"per-channel theta needs a non-empty 1-D quantile vector, got "
            f"shape {tuple(q.shape)}"
        )
    pooled = torch.full_like(q, float(pooled_scale))
    return torch.where(q > 0.0, q.clamp(min=float(min_scale)), pooled)


def promote_per_channel_theta(
    model,
    channel_quantiles,
    pooled_scales,
    *,
    min_scale: float = MIN_THETA,
) -> PerChannelThetaReport:
    """Rebind each eligible hop's ``activation_scale`` to its per-channel theta
    (in-place ``.data`` write: decorator references stay live). Entries align
    with ``model.get_perceptrons()`` order — the activation-analysis layout."""
    perceptrons = list(model.get_perceptrons())
    if len(channel_quantiles) != len(perceptrons) or len(pooled_scales) != len(
        perceptrons
    ):
        raise ValueError(
            "per-channel theta table count mismatch: "
            f"{len(channel_quantiles)} channel rows / {len(pooled_scales)} pooled "
            f"scales for {len(perceptrons)} perceptrons"
        )
    eligible = eligible_per_channel_perceptrons(model)
    promoted: list[str] = []
    skipped: list[str] = []
    for perceptron, quantiles, pooled in zip(
        perceptrons, channel_quantiles, pooled_scales
    ):
        name = str(getattr(perceptron, "name", "<unnamed>"))
        if id(perceptron) not in eligible:
            skipped.append(name)
            continue
        if quantiles is None:
            raise ValueError(
                f"per-channel theta: no channel quantiles captured for the "
                f"eligible hop {name!r}"
            )
        vector = per_channel_theta_vector(
            float(pooled), quantiles, min_scale=min_scale,
        )
        out_channels = int(perceptron.layer.weight.shape[0])
        if int(vector.numel()) != out_channels:
            raise ValueError(
                f"per-channel theta for {name!r}: {int(vector.numel())} channel "
                f"quantiles for {out_channels} output channels — the stats were "
                "captured on the wrong axis"
            )
        perceptron.set_activation_scale(vector)
        promoted.append(name)
    return PerChannelThetaReport(tuple(promoted), tuple(skipped))


def maybe_promote_per_channel_theta(
    config, model, activation_scale_stats, pooled_scales,
) -> PerChannelThetaReport | None:
    """The install-seam hook: promote when armed, no-op (None) otherwise.

    Fails loud when armed without the analysis step's per-channel capture — a
    stale cache resume must never silently install scalar thetas."""
    if not per_channel_theta_armed(config):
        return None
    table = (activation_scale_stats or {}).get("per_channel_quantiles")
    if not table:
        raise ValueError(
            "per_channel_theta is armed but activation_scale_stats carries no "
            "per-channel quantiles; rerun Activation Analysis with the flag armed"
        )
    report = promote_per_channel_theta(
        model, table["channels"], pooled_scales,
    )
    print(
        f"[PerChannelTheta] q={table['quantile']} "
        f"promoted={len(report.promoted)} {list(report.promoted)} "
        f"scalar={len(report.skipped)}",
        flush=True,
    )
    return report
