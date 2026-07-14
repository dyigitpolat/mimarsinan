"""Promote per-perceptron ``activation_scale`` (theta) to a trainable per-output-channel Parameter for deployed-cascade fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.mappers.scale_propagation import arm_compute_op_wrap_slots
from mimarsinan.spiking.per_channel_theta import eligible_per_channel_perceptrons


def _scale_carrying_nodes(perceptron):
    """Activation modules under ``perceptron`` (excluding the perceptron itself) that
    carry an ``activation_scale`` — e.g. ``TTFSActivation`` and a blend target."""
    for module in perceptron.modules():
        if module is perceptron:
            continue
        if isinstance(getattr(module, "activation_scale", None), torch.Tensor):
            yield module


def _rebind_activation_scale(perceptron, param: nn.Parameter) -> None:
    """Rebind theta on the perceptron AND every node referencing it, so the
    forward reads exactly the Parameter the optimizer trains."""
    perceptron.activation_scale = param
    for node in _scale_carrying_nodes(perceptron):
        node.activation_scale = param


def _per_channel_vector(perceptron) -> torch.Tensor:
    scale = perceptron.activation_scale.detach()
    out_dim = int(perceptron.layer.weight.shape[0])
    if scale.dim() == 0:
        return scale * torch.ones(out_dim, dtype=scale.dtype, device=scale.device)
    return scale.clone()


def promote_activation_scale_per_channel(model, *, skip_encoding: bool = True) -> list:
    """Rebind each non-encoding perceptron's ``activation_scale`` to a per-channel
    ``requires_grad`` Parameter, on the perceptron and every node that references
    it. Idempotent: a second call re-syncs node references. Returns the new params.
    """
    params = []
    for perceptron in model.get_perceptrons():
        if skip_encoding and getattr(perceptron, "is_encoding_layer", False):
            continue
        param = nn.Parameter(
            _per_channel_vector(perceptron).contiguous(), requires_grad=True,
        )
        _rebind_activation_scale(perceptron, param)
        params.append(param)
    return params


def promote_theta_for_exact_qat(model, *, per_channel: bool = True) -> dict:
    """[lif_exact_qat_program §6.2] Trainable theta under the R3 seam constraints:
    per-channel ONLY on matching-axis-eligible hops (exact on-chip export via the
    ``per_input_scales`` fold), scalar-trainable on externally-consumed hops
    (host boundary mean-collapse seams), encoder frozen. Arms the deployed
    ComputeOp wrap slots when any hop goes per-channel. Returns the witness
    report ``{"per_channel": [...], "scalar": [...], "params": [...]}``.

    ``per_channel=False`` forces SCALAR theta everywhere (no ComputeOp wrap): the
    synchronized-ttfs mapper forward does not route per-channel theta through the
    wrap (shape mismatch), so sync trains a scalar-per-perceptron theta.
    """
    eligible = eligible_per_channel_perceptrons(model) if per_channel else set()
    per_channel_names: list[str] = []
    scalar: list[str] = []
    params: list[nn.Parameter] = []
    for perceptron in model.get_perceptrons():
        if getattr(perceptron, "is_encoding_layer", False):
            continue
        name = str(getattr(perceptron, "name", "<unnamed>"))
        if id(perceptron) in eligible:
            param = nn.Parameter(
                _per_channel_vector(perceptron).contiguous(), requires_grad=True,
            )
            per_channel_names.append(name)
        else:
            scale = perceptron.activation_scale.detach()
            param = nn.Parameter(
                scale.reshape(-1).mean().clone(), requires_grad=True,
            )
            scalar.append(name)
        _rebind_activation_scale(perceptron, param)
        params.append(param)
    if per_channel_names:
        # The deployed ScaleNormalizingWrapper and the NF twin must agree on the
        # per-channel host decode from the install seam on (per_channel_theta).
        arm_compute_op_wrap_slots(model.get_mapper_repr())
    return {"per_channel": per_channel_names, "scalar": scalar, "params": params}


def emit_theta_install_witness(reporter, tag, report, *, extra=None) -> dict:
    """The shared theta-aware install witness: build the witness from a promote
    ``report``, print the loud ``[tag]`` line, and emit the reporter event (kind
    = the tag's lowercased snake form). ``extra`` merges mode-specific fields
    (e.g. LIF's fold/entry/retime, computed after promote). Reused by every mode
    so the witness/telemetry is one mechanism, not per-mode duplication."""
    witness = {
        "installed": True,
        "theta_per_channel": len(report["per_channel"]),
        "theta_scalar": len(report["scalar"]),
        **(extra or {}),
    }
    detail = " ".join(f"{k}={v}" for k, v in (extra or {}).items())
    print(
        f"[{tag}] theta in-loop: per_channel={witness['theta_per_channel']} "
        f"{report['per_channel']} scalar={witness['theta_scalar']} "
        f"{report['scalar']}" + (f" {detail}" if detail else ""),
        flush=True,
    )
    emit_reporter_event(reporter, tag.lower().replace("-", "_"), witness)
    return witness


def install_exact_qat_theta(model, reporter, tag, *, per_channel=True, extra=None):
    """Promote theta in-loop then emit the witness — the atomic theta-aware
    install for modes with no post-promote extras (ttfsq/sync/casc). LIF, which
    computes fold/entry extras between promote and witness, calls
    ``promote_theta_for_exact_qat`` + ``emit_theta_install_witness`` directly."""
    report = promote_theta_for_exact_qat(model, per_channel=per_channel)
    witness = emit_theta_install_witness(reporter, tag, report, extra=extra)
    return report, witness
