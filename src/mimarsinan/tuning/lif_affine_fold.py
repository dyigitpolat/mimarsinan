"""Calibration-derived per-channel affine fold absorbing the LIF dead-zone bias (C4)."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.channel_axis_walk import channel_aligned_consumer_targets
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.models.nn.activations.lif import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

# LS gains beyond this window mean the channel statistics are degenerate at the
# coarse grid; the lab estimator clamps rather than trusting them (§4 C4).
GAIN_MIN = 0.25
GAIN_MAX = 4.0

AFFINE_FOLD_FLAG = "_lif_affine_folded"

_VAR_EPS = 1e-10


def fit_channel_affine(
    deployed: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-channel LS affine ``target ~ a*deployed + c``.

    Dead (zero-variance) channels keep unit gain and a mean-matching bias.
    The FULL affine is the only sound estimator at coarse grids — bias-only
    mean matching is refuted (-4.2pp).
    """
    deployed = deployed.double()
    target = target.double()
    var = deployed.var(dim=0, unbiased=False)
    cov = ((deployed - deployed.mean(0)) * (target - target.mean(0))).mean(0)
    a = torch.where(
        var > _VAR_EPS,
        cov / var.clamp(min=_VAR_EPS),
        torch.ones_like(var),
    )
    a = torch.clamp(a, GAIN_MIN, GAIN_MAX)
    c = target.mean(0) - a * deployed.mean(0)
    return a, c


def fold_affine_into_consumer(
    consumer, a: torch.Tensor, c: torch.Tensor,
) -> None:
    """Fold the producer-output affine ``r -> a*r + c`` into the consumer:
    ``W~[:, j] *= a_j``, ``b~ += W~ @ c`` in the effective (wire) domain —
    the negative-shift fold family."""
    transformer = PerceptronTransformer()
    effective_weight = transformer.get_effective_weight(consumer)
    c_in = c.to(effective_weight.dtype).to(effective_weight.device)
    correction = (effective_weight * c_in).sum(dim=-1)
    transformer.apply_effective_bias_transform(consumer, lambda b: b + correction)
    a_in = a.reshape(1, -1)
    transformer.apply_effective_weight_transform(
        consumer, lambda w: w * a_in.to(w.dtype).to(w.device),
    )


def fold_affine_into_readout(perceptron, a: torch.Tensor, c: torch.Tensor) -> None:
    """Fold the terminal affine as ``z~ -> a*z~ + c`` on the wire pre-activation
    (effective row scale + bias) — exact under the membrane-augmented readout,
    where the reported logit IS the unquantized charge."""
    transformer = PerceptronTransformer()
    a_row = a.reshape(-1, 1)
    transformer.apply_effective_weight_transform(
        perceptron, lambda w: w * a_row.to(w.dtype).to(w.device),
    )
    transformer.apply_effective_bias_transform(
        perceptron,
        lambda b: a.to(b.dtype).to(b.device) * b + c.to(b.dtype).to(b.device),
    )


def _safe_theta(perceptron, ref: torch.Tensor) -> torch.Tensor:
    theta = torch.as_tensor(
        perceptron.activation_scale, device=ref.device, dtype=ref.dtype,
    )
    return theta.reshape(-1).clamp(min=1e-12)


def _normalized(perceptron, value: torch.Tensor) -> torch.Tensor | None:
    """Channels-last per-channel rate view: leading axes (batch, patches, ...)
    pool as samples so the per-channel affine matches the consumer's columns."""
    flat = value.detach().reshape(-1, value.shape[-1])
    theta = _safe_theta(perceptron, flat)
    if theta.numel() not in (1, flat.shape[1]):
        return None
    return flat / theta


def float_envelope_rates_by_perceptron(model, x: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Per-perceptron float-envelope rates ``clamp(z/theta, 0, 1)``: the model
    forward with every LIF node bypassed to its float envelope ``clamp(z, 0, theta)``."""

    def _bypass(module, inputs, _output):
        z = inputs[0]
        theta = torch.as_tensor(
            module.activation_scale, device=z.device, dtype=z.dtype,
        ).clamp(min=1e-12)
        return torch.minimum(z.clamp(min=0.0), theta)

    captured: Dict[int, torch.Tensor] = {}

    def _make_capture(perceptron):
        def _capture(_module, _inputs, output):
            normalized = _normalized(perceptron, output)
            if normalized is not None:
                captured[id(perceptron)] = normalized.clamp(0.0, 1.0)
        return _capture

    handles = [
        m.register_forward_hook(_bypass)
        for m in model.modules()
        if isinstance(m, LIFActivation)
    ]
    handles += [
        p.activation.register_forward_hook(_make_capture(p))
        for p in model.get_perceptrons()
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def deployed_rates_by_perceptron(
    model, x: torch.Tensor, simulation_steps: int,
) -> Dict[int, torch.Tensor]:
    """Per-perceptron deployed cycle-accurate rates (counts/T) via the
    chip-aligned segment forward's value recorder."""
    recorder: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(
            model, x, int(simulation_steps), node_value_recorder=recorder,
        )
    out: Dict[int, torch.Tensor] = {}
    for perceptron in model.get_perceptrons():
        value = recorder.get(id(perceptron))
        if value is None:
            continue
        normalized = _normalized(perceptron, value)
        if normalized is not None:
            out[id(perceptron)] = normalized
    return out


def _only_reaches_host_output(node, consumers_map) -> bool:
    """True when the node's decoded value feeds only host-side structural tails
    (no downstream perceptron re-encode, no ComputeOp seam)."""
    frontier = list(consumers_map.get(id(node), []))
    seen: set[int] = set()
    while frontier:
        consumer = frontier.pop()
        if id(consumer) in seen:
            continue
        seen.add(id(consumer))
        if perceptron_of(consumer) is not None or isinstance(consumer, ComputeOpMapper):
            return False
        frontier.extend(consumers_map.get(id(consumer), []))
    return True


def _foldable_perceptron_consumers(node, consumers_map):
    """Consumer perceptrons whose weight columns see the producer's channel
    axis unmediated, discovered by the shared mapper-DAG walk (the M4 SSOT);
    ``(consumers, skip_reason)`` — a reason means the fold is voided."""
    targets = channel_aligned_consumer_targets(node, consumers_map)
    if targets is None:
        return None, "channel_axis_not_preserved"
    perceptron_targets, _module_targets = targets
    if not perceptron_targets:
        # Host-side modules decode at a different currency (theta-scaled
        # values, not normalized rates); the fold honestly leaves them alone.
        return None, "host_compute_consumer"
    return list(perceptron_targets), None


def apply_lif_affine_fold(model, cal_x: torch.Tensor, simulation_steps: int) -> dict:
    """Layer-sequential per-channel affine folds over the mapper graph.

    Each non-encoding producer's deployed rate is refit against the fixed
    float-envelope target AFTER upstream folds landed (the lab's sequential
    estimator); folds land in consumer columns/bias, terminal perceptrons in
    their own rows. Idempotent per producer. Returns a summary report.
    """
    mapper_repr = model.get_mapper_repr()
    exec_order = mapper_repr.execution_order()
    consumers_map = mapper_repr.consumer_map()

    targets = float_envelope_rates_by_perceptron(model, cal_x)

    report: dict = {"folded": 0, "consumer_folds": 0, "readout_folds": 0, "skipped": {}}

    for node in exec_order:
        producer = perceptron_of(node)
        if producer is None or getattr(producer, "is_encoding_layer", False):
            continue
        name = (
            getattr(node, "name", None)
            or getattr(producer, "name", None)
            or f"perceptron@{id(producer):x}"
        )
        if getattr(producer, AFFINE_FOLD_FLAG, False):
            report["skipped"][name] = "already_folded"
            continue
        target = targets.get(id(producer))
        if target is None:
            report["skipped"][name] = "no_float_target"
            continue

        if _only_reaches_host_output(node, consumers_map):
            kind = "readout"
            consumers = []
        else:
            resolved, skip_reason = _foldable_perceptron_consumers(node, consumers_map)
            if resolved is None:
                report["skipped"][name] = skip_reason
                continue
            if any(
                p.layer.weight.shape[1] != target.shape[1] for p in resolved
            ):
                report["skipped"][name] = "consumer_width_mismatch"
                continue
            kind = "consumer"
            consumers = resolved

        deployed = deployed_rates_by_perceptron(model, cal_x, simulation_steps)
        r_dep = deployed.get(id(producer))
        if r_dep is None or r_dep.shape != target.shape:
            report["skipped"][name] = "no_deployed_rates"
            continue

        a, c = fit_channel_affine(r_dep, target)
        if kind == "readout":
            if producer.layer.weight.dim() != 2:
                report["skipped"][name] = "non_2d_readout"
                continue
            fold_affine_into_readout(producer, a, c)
            report["readout_folds"] += 1
        else:
            for consumer in consumers:
                fold_affine_into_consumer(consumer, a, c)
            report["consumer_folds"] += 1
        setattr(producer, AFFINE_FOLD_FLAG, True)
        report["folded"] += 1

    return report
