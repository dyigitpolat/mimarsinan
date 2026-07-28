"""Seam-certificate auditor: per-edge B/C/G classification of a deployment graph (spiking_deployment_calculus.md sec.9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mimarsinan.mapping.support.tensor_stats import safe_quantile
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.wire_semantics import lif_count_staircase
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.scale_aware_boundaries import (
    read_boundary_out_scales,
    stamped_input_boundary_scale,
)
from mimarsinan.spiking.segment_partition import (
    partition_spike_segments,
    perceptron_of,
)

_CURRENCY_RTOL = 1e-3
_TWIN_RTOL = 1e-4
_BOUNDARY_OOB_C = 0.01
_BOUNDARY_BIAS_B_FACTOR = 3.0
_EFF_LEVELS_STARVED = 2.0
_SAMPLE_CAP = 1_000_000


@dataclass(frozen=True)
class SeamCertificate:
    """One edge/node certificate: measured defect + its taxonomy class."""

    site: str
    kind: str  # 'currency' | 'boundary' | 'host_twin' | 'kernel'
    classification: str  # 'B' | 'C' | 'G' | 'ok'
    delta_mean: float
    delta_max: float
    grid_bound: float
    kappa: float
    sigma: float = 0.0
    oob_fraction: float = 0.0
    eff_levels: float | None = None
    note: str = ""


@dataclass(frozen=True)
class SeamAuditLedger:
    certificates: tuple[SeamCertificate, ...]

    @property
    def type_b(self) -> tuple[SeamCertificate, ...]:
        return tuple(c for c in self.certificates if c.classification == "B")

    def by_kind(self, kind: str) -> list[SeamCertificate]:
        return [c for c in self.certificates if c.kind == kind]


def _scalar(value) -> float:
    return float(torch.as_tensor(value).detach().to(torch.float64).mean())


def _value_walk(repr_, x):
    """Eviction-free analytical composition: the auditor needs every node value."""
    repr_._ensure_exec_graph()
    exec_order, deps = repr_._exec_order, repr_._deps
    values: dict = {}
    with torch.no_grad():
        for node in exec_order:
            d = deps.get(node, [])
            if len(d) == 0:
                values[node] = node.forward(x)
            elif len(d) == 1:
                values[node] = node.forward(values[d[0]])
            else:
                values[node] = node.forward(tuple(values[dep] for dep in d))
    return exec_order, deps, values


def _capture_preactivations(repr_, run_walk):
    """{perceptron: pre-activation tensor} captured via pre-hooks during the walk."""
    captured: dict = {}
    handles = []
    for p in repr_.get_perceptrons():
        act = getattr(p, "activation", None)
        if act is None:
            continue

        def _hook(_module, inputs, _p=p):
            captured[_p] = inputs[0].detach()

        handles.append(act.register_forward_pre_hook(_hook))
    try:
        result = run_walk()
    finally:
        for h in handles:
            h.remove()
    return captured, result


def _entry_reference(perceptron, v: torch.Tensor) -> torch.Tensor:
    """The consumer's trained entry function (identity when none installed)."""
    entry = getattr(perceptron, "input_activation", None)
    if entry is None or isinstance(entry, nn.Identity):
        return v
    with torch.no_grad():
        return entry(v)


def _boundary_certificate(site, v, kappa, sigma, T, perceptron) -> SeamCertificate:
    bound = kappa / (2 * T)
    with torch.no_grad():
        shifted = v + sigma
        oob = float(((shifted < 0) | (shifted > kappa)).float().mean())
        rate = (shifted / kappa).clamp(0.0, 1.0)
        deployed = torch.round(rate * T) / T * kappa - sigma
        delta = deployed - _entry_reference(perceptron, v)
        bias = abs(float(delta.mean()))
        delta_max = float(delta.abs().max())
    if oob > _BOUNDARY_OOB_C:
        cls, note = "C", f"out-of-band mass {oob:.3f} (coverage, not convention)"
    elif bias > _BOUNDARY_BIAS_B_FACTOR * bound:
        cls, note = "B", f"in-band systematic bias {bias:.4g} > {_BOUNDARY_BIAS_B_FACTOR}x grid bound"
    elif delta_max > 0.0:
        cls, note = "G", "grid noise within budget"
    else:
        cls, note = "ok", "entry identity exact"
    return SeamCertificate(
        site, "boundary", cls, note=note, delta_mean=bias, delta_max=delta_max,
        kappa=kappa, grid_bound=bound, sigma=sigma, oob_fraction=oob,
    )


def _currency_certificate(site, perceptron, expected: float, T) -> SeamCertificate | None:
    stamps = []
    fold = getattr(perceptron, "per_input_scales", None)
    if fold is not None:
        stamps.append(("per_input_scales", _scalar(fold)))
    entry_scale = getattr(perceptron, "input_activation_scale", None)
    if entry_scale is not None:
        stamps.append(("input_activation_scale", _scalar(entry_scale)))
    if not stamps:
        return None
    mismatches = [
        (name, value) for name, value in stamps
        if abs(value - expected) > _CURRENCY_RTOL * max(abs(expected), 1e-12)
        # An unstamped-at-1.0 entry scale is the pre-propagation state, not V-B.
        and not (name == "input_activation_scale" and value == 1.0)
    ]
    if mismatches:
        name, value = mismatches[0]
        cls, note = "B", f"currency mismatch: {name}={value:.4g} vs producer fold {expected:.4g}"
        delta = abs(value - expected) / max(abs(expected), 1e-12)
    else:
        cls, note, delta = "ok", "consumer folds agree with the producer currency", 0.0
    return SeamCertificate(
        site, "currency", cls, note=note, delta_mean=delta, delta_max=delta,
        kappa=expected, grid_bound=expected / (2 * T),
    )


def _host_twin_certificate(site, node, deps, values, table, T) -> SeamCertificate | None:
    kappa = float(table.get(node, 1.0))
    armed = node.per_source_scales is not None and node.output_scale is not None
    offset = node.output_value_offset
    if offset is not None and not armed:
        return SeamCertificate(
            site, "host_twin", "B",
            note="unarmed op carries output_value_offset (armed-only stamping law)",
            delta_mean=float("inf"), delta_max=float("inf"), kappa=kappa,
            grid_bound=kappa / (2 * T), sigma=_scalar(offset),
        )
    if not armed:
        return None
    d = deps.get(node, [])
    with torch.no_grad():
        wire_inputs = [values[dep] / max(float(table.get(dep, 1.0)), 1e-12) for dep in d]
        wire_arg = wire_inputs[0] if len(wire_inputs) == 1 else tuple(wire_inputs)
        decoded = node.forward_scale_normalized(wire_arg) * kappa
        reference = values[node]
        rel = float((decoded - reference).abs().mean()) / (
            float(reference.abs().mean()) + 1e-12
        )
        delta_max = float((decoded - reference).abs().max())
    cls, note = (
        ("B", f"wire twin diverges from value twin (rel {rel:.4g})")
        if rel > _TWIN_RTOL else ("ok", "wire twin == value twin")
    )
    return SeamCertificate(
        site, "host_twin", cls, note=note, delta_mean=rel, delta_max=delta_max,
        kappa=kappa, grid_bound=kappa / (2 * T),
        sigma=_scalar(offset) if offset is not None else 0.0,
    )


def _embedded_lif(activation):
    """Sanctioned unwrap, else a modules() scan: an UNKNOWN wrapper around a
    LIF must still be audited (skipping would be an E4 false negative)."""
    lif = unwrap_lif_activation(activation)
    if lif is not None or not isinstance(activation, nn.Module):
        return lif
    return next(
        (m for m in activation.modules() if isinstance(m, LIFActivation)), None,
    )


def _kernel_certificate(site, perceptron, z, T, quantile) -> SeamCertificate | None:
    lif = _embedded_lif(getattr(perceptron, "activation", None))
    if lif is None:
        return None
    theta_t = torch.as_tensor(lif.activation_scale).detach()
    theta = max(_scalar(theta_t), 1e-12)
    grid_step = theta / T
    with torch.no_grad():
        actual = perceptron.activation(z)
        staircase = lif_count_staircase(
            z, theta_t.to(z.dtype), T, compare_mode=lif.thresholding_mode,
        )
        delta = (actual - staircase).abs()
        mean_delta = float(delta.mean())
        delta_max = float(delta.max())
        eff = float(
            T * safe_quantile(actual.abs().float(), quantile, limit=_SAMPLE_CAP) / theta
        )
    if mean_delta > grid_step:
        cls, note = "B", (
            f"activation is not its count staircase (mean |delta| "
            f"{mean_delta:.4g} > one grid step {grid_step:.4g})"
        )
    elif eff < _EFF_LEVELS_STARVED:
        cls, note = "C", f"starved kernel: eff_levels {eff:.2f}"
    else:
        cls, note = "ok", "kernel == count staircase (ties within one level)"
    return SeamCertificate(
        site, "kernel", cls, note=note, delta_mean=mean_delta, delta_max=delta_max,
        kappa=theta, grid_bound=grid_step / 2, eff_levels=eff,
    )


def audit_model(model_repr_or_model, T: int, x, *, quantile: float = 0.99) -> SeamAuditLedger:
    """Audit every seam of the graph on one input batch; returns the ledger."""
    repr_ = (
        model_repr_or_model
        if hasattr(model_repr_or_model, "_ensure_exec_graph")
        else model_repr_or_model.get_mapper_repr()
    )
    preacts, (exec_order, deps, values) = _capture_preactivations(
        repr_, lambda: _value_walk(repr_, x),
    )
    table = read_boundary_out_scales(
        repr_, input_data_scale=stamped_input_boundary_scale(repr_),
    )
    seg_of, produces = partition_spike_segments(exec_order, deps)
    site_of = {n: f"{i:02d}:{type(n).__name__}" for i, n in enumerate(exec_order)}

    certificates: list[SeamCertificate] = []
    seen_kernels: set[int] = set()
    for node in exec_order:
        if isinstance(node, ComputeOpMapper):
            cert = _host_twin_certificate(site_of[node], node, deps, values, table, T)
            if cert is not None:
                certificates.append(cert)
        p = perceptron_of(node)
        if p is None or not produces.get(node, False):
            continue
        boundary_deps = [
            dep for dep in deps.get(node, [])
            if not (produces.get(dep, False) and seg_of.get(dep) == seg_of.get(node))
        ]
        if boundary_deps:
            expected = sum(float(table.get(dep, 1.0)) for dep in boundary_deps) / len(
                boundary_deps
            )
            cert = _currency_certificate(site_of[node], p, expected, T)
            if cert is not None:
                certificates.append(cert)
        for dep in boundary_deps:
            kappa = max(float(table.get(dep, 1.0)), 1e-12)
            sigma_v = getattr(dep, "_negative_shift", None)
            sigma = _scalar(sigma_v) if sigma_v is not None else 0.0
            certificates.append(_boundary_certificate(
                f"{site_of[dep]}->{site_of[node]}", values[dep], kappa, sigma, T, p,
            ))
        if id(p) not in seen_kernels and p in preacts:
            seen_kernels.add(id(p))
            cert = _kernel_certificate(site_of[node], p, preacts[p], T, quantile)
            if cert is not None:
                certificates.append(cert)
    return SeamAuditLedger(certificates=tuple(certificates))
