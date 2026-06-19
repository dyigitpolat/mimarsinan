"""Reusable TTFS-cascade toy builders for isolated spiking tests.

These build *converted* model flows whose perceptrons carry genuine
``TTFSActivation`` spike nodes at a chosen timing resolution ``S``, so a test can
drive the deployed single-spike cascade (``cascade_forward``) — used to study the
spike encode/decode behaviour at segment boundaries.

Two structural shapes, selected by ``host_ops``:

* ``host_ops=False`` — one deep cascade segment (``depth`` stacked Linear+ReLU).
* ``host_ops=True``  — value-domain compute ops (``*``/``+``) interleaved between
  stages, which the converter emits as host ``ComputeOpMapper`` nodes. Each cut
  forces a spike *decode -> host compute -> re-encode* boundary, so tests can
  study spike encode/decode effects across multiple sequential neural segments
  (the regime where the cascade<->staircase divergence compounds, §2.2).

``activation_scale`` is calibrated once (per-perceptron max of a value-domain
pass) so the normalized value spans the staircase's [0, 1] band.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SingleSegmentMLP(nn.Module):
    """``depth`` stacked Linear+ReLU -> one cascade segment (no host ops)."""

    def __init__(self, depth: int, width: int, in_dim: int, out_dim: int):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class _HostOpMLP(nn.Module):
    """Linear+ReLU stages separated by value-domain host compute ops.

    Each ``h * w (+ b)`` between stages converts to a ``ComputeOpMapper`` (host /
    value domain), cutting the graph into sequential neural segments with a spike
    decode/re-encode boundary at every cut.
    """

    def __init__(self, depth: int, width: int, in_dim: int, out_dim: int):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:  # host compute op between segments
                x = x * 0.7 + 0.05
        return x


def install_ttfs_nodes(flow, S: int):
    """(Re-)install a fresh ``TTFSActivation`` at resolution ``S`` on every
    perceptron of the converted flow."""
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=S,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    return flow.double()


def _calibrate_scales(flow, x):
    """Set each perceptron's ``activation_scale`` to the max magnitude of its
    value-domain output, so normalized values span the staircase's [0, 1] band."""
    caps: dict = {}
    handles = []
    for p in flow.get_perceptrons():
        def hook(_m, _i, out, p=p):
            caps[p] = max(caps.get(p, 0.0), float(out.detach().abs().max()))
        handles.append(p.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(max(caps[p], 1e-3), dtype=torch.float64))


def build_cascade_flow(
    *,
    host_ops: bool = False,
    depth: int = 3,
    width: int = 8,
    in_dim: int = 8,
    out_dim: int = 4,
    S: int = 4,
    seed: int = 0,
    n_calib: int = 64,
):
    """Build a converted TTFS-cascade flow for isolated tests.

    Returns ``(flow, calib_x)``: ``flow.get_mapper_repr()`` drives the genuine
    cascade; ``calib_x`` is the in-[0,1] batch used to calibrate the (fixed)
    activation scales — reuse it (or any in-[0,1] input) for forwards.
    """
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(seed)
    ctor = _HostOpMLP if host_ops else _SingleSegmentMLP
    base = ctor(depth, width, in_dim, out_dim)
    for m in base.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.4, 0.4)
            nn.init.uniform_(m.bias, -0.05, 0.05)

    calib_x = torch.rand(n_calib, in_dim, dtype=torch.float64)
    flow = convert_torch_model(base, (in_dim,), out_dim, device="cpu")
    _calibrate_scales(flow, calib_x)
    install_ttfs_nodes(flow, S)
    return flow, calib_x


def segment_count(flow) -> int:
    """Number of distinct neural (spike) segments in the converted flow."""
    from mimarsinan.spiking.segment_partition import partition_perceptron_segments

    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    seg_of = partition_perceptron_segments(repr_._exec_order, repr_._deps)
    return len(set(seg_of.values()))


def cascade_forward(flow, x, S, *, grad: bool = False, surrogate_temp: float | None = None):
    """Genuine single-spike ramp-integrate cascade (the deployed dynamics).

    ``surrogate_temp`` (None = the historical severed contract) enables the
    offload-boundary straight-through estimator so the genuine backward flows
    through every segment; the forward is unchanged."""
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    drv = TTFSSegmentForward(flow.get_mapper_repr(), S, boundary_surrogate_temp=surrogate_temp)
    if grad:
        return drv(x.double())
    with torch.no_grad():
        return drv(x.double())
