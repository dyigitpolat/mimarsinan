"""[U-LIF D1, D2] The exact-QAT network's host-to-chip entries as deployed: the entry encode on every host-fed segment
entry, at a gauge covering the calibration quantile of its entry values; no input grid on a host-side encoder.

The deployed program encodes every value a host node hands to a core: it clips at the entry gauge and rounds to the
window's grid. Under ``subsume`` the exact-QAT network modelled that encode only on the encoders, whose consumers read a
staircase already on its grid; a core reading a host op (a residual add, its ReLU, a host convolution) read an unclipped
float, so the LIF swap was the first stage to run the deployed composition (lif_diag D2: C5 untuned -9.22 points, tuned
-2.70). The encode now sits on every segment entry a host value feeds, whatever the placement, so the quantized network
IS the deployed composition and the QAT trains through it. The entry gauge of a residual join was the mean of its
sources' scales, which the join's sum exceeds; each host value node that feeds an entry now presents the calibration
quantile of its values (``ComputeOpMapper.boundary_traffic_scale``, the observed-traffic lift both walks and the weight
fold read). A subsumed encoder runs on the host, where no chip constraint asks for an unsigned grid: it reads the
normalized image whole (lif_diag D1: the clamp to [0, s] zeroed 53.5 percent of C5's input, 63.60 to 31.48).
"""

from __future__ import annotations

import torch

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.spiking.distribution_matching import FANIN_TRAFFIC_QUANTILE
from mimarsinan.spiking.scale_aware_boundaries import establish_wire_gauge, read_boundary_out_scales
from mimarsinan.spiking.segment_forward import AnalyticalSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.torch_mapping.encoding_layers import encoder_deploys_as_staircase_hop, host_fed_segment_entries
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    deployed_negative_boundary_seam,
    install_lif_input_quantizer,
    read_calibration_batch,
)
from mimarsinan.tuning.orchestration.negative_boundary_seam import CALIBRATION_CHUNK, _calibration_mode

__all__ = ["ENTRY_GAUGE_QUANTILE", "calibrate_segment_entry_gauges", "install_deployed_entry_encode"]

#: The calibration quantile of a host-fed entry's positive values its gauge covers (the fan-in traffic law's).
ENTRY_GAUGE_QUANTILE = FANIN_TRAFFIC_QUANTILE


def _gauged_by_its_values(node) -> bool:
    """A host value node whose gauge the entry calibration sets: a host op the negative-boundary policy left as it was
    (a lift carries its own cover; a shifted boundary is encoded at the shift's gauge)."""
    return (isinstance(node, ComputeOpMapper) and getattr(node, "boundary_traffic_scale", None) is None
            and getattr(node, "output_value_offset", None) is None and getattr(node, "_negative_shift", None) is None
            and not getattr(node, "negative_shift_at_seam", False))


def calibrate_segment_entry_gauges(model, calibration_x: torch.Tensor, *, simulation_steps: int,
                                   input_data_scale: float, quantile: float = ENTRY_GAUGE_QUANTILE) -> dict:
    """Lift each host value node feeding a segment entry to the ``quantile`` of its positive values on the calibration
    batch where that exceeds its propagated gauge, and re-establish the gauge; returns ``{name: (before, after)}``."""
    repr_ = model.get_mapper_repr()
    producers = {id(src): src for _, src in host_fed_segment_entries(repr_) if _gauged_by_its_values(src)}
    if not producers:
        return {}
    samples: dict = {}
    driver = SegmentForwardDriver(repr_, int(simulation_steps), AnalyticalSegmentPolicy())
    with _calibration_mode(model):
        for chunk in calibration_x.split(CALIBRATION_CHUNK):
            driver(chunk, compute_sample_recorder=samples)
    table = read_boundary_out_scales(repr_, input_data_scale=float(input_data_scale))
    lifted = {}
    for node in producers.values():
        values = samples.get(node)
        positive = None if values is None else values.float().flatten().clamp(min=0.0)
        if positive is None or not bool((positive > 0).any()):
            continue
        observed = float(safe_quantile(positive[positive > 0], float(quantile)))
        before = float(torch.as_tensor(table.get(node, input_data_scale)).float().mean())
        if observed > before:
            node.boundary_traffic_scale = observed
            lifted[str(getattr(node, "name", None) or type(node.module).__name__)] = (before, observed)
    if lifted:
        establish_wire_gauge(model, input_data_scale=float(input_data_scale))
        print(f"[U-LIF ENTRY GAUGE] {len(lifted)} host-fed segment entry gauge(s) lifted to the {quantile} quantile of "
              f"their values (name, before, after): {[(k, round(a, 6), round(b, 6)) for k, (a, b) in lifted.items()]}",
              flush=True)
    return lifted


def install_deployed_entry_encode(model, pipeline_config, *, trainer=None) -> int:
    """The exact-QAT install of the deployed entry encode (the arm's predicate is the caller's): returns installs.

    Under ``offload`` the entry carriers are the QAT's own trained clamp before the negative-boundary policy decides its
    scope (review M1, its lift the opt-in's); under ``subsume`` the policy decides first, so a signed host boundary is
    lifted in its op and the encode then models the lifted value. The gauge seam runs on every wire-currency graph."""
    simulation_steps = int(pipeline_config["simulation_steps"])
    placement = str(pipeline_config.get("encoding_layer_placement", "subsume"))
    repr_ = getattr(model, "get_mapper_repr", lambda: None)()
    if repr_ is None:
        return 0  # no mapper graph, no segment entries; a subsumed encoder takes no input grid
    entries = host_fed_segment_entries(repr_)
    host_side_encoders = encoder_deploys_as_staircase_hop(placement)
    installed = 0
    if not host_side_encoders:
        installed += sum(int(install_lif_input_quantizer(p, simulation_steps)) for p, _ in entries)
    input_data_scale = float(ResolvedWorkloadProfile.from_config(pipeline_config).input_data_scale)
    seam = deployed_negative_boundary_seam(model, trainer, pipeline_config)
    calibration_x = None if trainer is None else read_calibration_batch(model, trainer)

    def boundary_policy(graph) -> None:
        if seam is not None:
            seam(graph)
        if calibration_x is not None:
            calibrate_segment_entry_gauges(graph, calibration_x, simulation_steps=simulation_steps,
                                           input_data_scale=input_data_scale)

    # [U-SC S-1] the first stage that walks the deployed composition establishes the gauge on every wire-currency
    # graph; the quantizers read the entry scales in place, so they follow every later re-derivation.
    establish_deployed_wire_gauge(model, input_data_scale=input_data_scale, negative_boundary=boundary_policy)
    if host_side_encoders:
        # A boundary Soft Core Mapping's shift carries is encoded shifted at the deployed sites, which the sigma-free
        # entry encode does not model: its entries keep the walk's own encode.
        installed += sum(int(install_lif_input_quantizer(p, simulation_steps)) for p, src in entries
                         if getattr(src, "_negative_shift", None) is None)
    return installed
