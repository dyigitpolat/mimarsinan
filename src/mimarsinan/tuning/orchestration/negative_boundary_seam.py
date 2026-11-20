"""[U-SC2 A-2, D-USC-2] The negative-value boundary policy at the gauge seam.

Soft Core Mapping makes every negative ComputeOp -> core boundary encodable (``ensure_negative_boundary_policy``), but
only there, so every walk before it (the exact-QAT AQ D-hat, the LIF finalize twin, LIF Adaptation's retention metric)
clamped a signed host chain at the [0, 1] spike encode: on SD-10c's cached S4 model the walk agreed with the model's own
forward on 1 of 64 E01 images and read chance. The seam now applies the policy before any walk reads the composition,
on the policy's ONE scope (``encoded_untrained_boundary``: encoded by a core, not the QAT's trained clamp, not lifted),
by its own mechanism switch (``negative_value_shift``):

* shift (the default): each signed boundary's lift is folded into the op's own function before training, sigma in the
  op and its cover kappa as the op's emitted gauge (``ScaleNormalizingWrapper``: ``(f + sigma) / kappa``), every
  consumer baked (``B' = B - W sigma``), so the plain forward, every walk and the deployed program carry one boundary;
  sigma and kappa follow the signed-seam capacity law (quantiles of the calibrated boundary values, never the extremes:
  conversion_boundary_algebra.md, 10f and 10g). Soft Core Mapping's own shift alone (sigma = -min, the gauge unchanged)
  moves the whole band above the encode's top: measured on S4, 0 of 64 against the walk's 2 unshifted
  (final_units/usc2/probes/out/s4_policy_variants_base_scm.json), so the lift carries its cover;
* subsume-forward: Soft Core Mapping's own ``subsume_forward_negative_boundaries``, unchanged.

A lift reaches the zero padding of the convolutions that read the op (U-SC2 finding: a padded tap reads no sigma while
the bake compensates the whole kernel): the padding moves into the op's output (``ZeroPaddedOutput``), the same function,
and the convolution reads it unpadded. A boundary whose op cannot carry the lift (not a wire value op, a consumer chain no
bias can compensate, a core another lift reaches, or a padded read the padding cannot move to) gets Soft Core Mapping's
own policy here, on the same calibration batch through its own LIF calibration walk (review M2: its shift and bake, or
its refusal by name), so every walk reads the boundary Soft Core Mapping deploys; the op is stamped
``negative_shift_at_seam``. Soft Core Mapping's later call skips every lifted or seam-shifted boundary (the trained
boundary of the sigma-scope law; the shift still travels to the program) and stamps the rest as before. The lift
calibration is the value walk on the calibration batch, in eval mode, with the run's random state preserved, so a graph
with nothing to lift is left as it was. Every lift and seam shift is printed by name.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.scale_propagation import arm_wrap_slots
from mimarsinan.chip_simulation.spiking_point import SpikingPoint
from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_point
from mimarsinan.mapping.support.negative_boundary import (
    CALIBRATION_CHUNK,
    NEGATIVE_TOLERANCE,
    _is_host_node,
    _perceptron_of,
    boundary_consumers,
    calibrated_compute_op_minima,
    encoded_untrained_boundary,
    lossy_negative_boundaries,
    subsume_forward_negative_boundaries,
)
from mimarsinan.mapping.support.negative_shift import apply_negative_value_shifts
from mimarsinan.mapping.support.padded_output import ZeroPaddedOutput
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.received_shift import bake_received_shift, correct_for_received_shift, every_axon_shift
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales, read_boundary_out_scales
from mimarsinan.spiking.segment_forward import AnalyticalSegmentPolicy, SegmentForwardDriver
from mimarsinan.tuning.orchestration.signed_seam_install import (
    _bake_shift_into_host_bias,
    _lift_admits,
    _lifts_admitted,
    _signed_seam_quantiles,
)

__all__ = ["CALIBRATION_CHUNK", "NegativeBoundarySeam", "lift_negative_boundaries", "negative_boundary_candidates"]

#: The signed-seam capacity law's quantile (``install_signed_seam_offsets``' default): sigma and kappa cover the band.
SEAM_QUANTILE = 0.99


def negative_boundary_candidates(model) -> list:
    """Host ops the policy could lift: on the ONE scope, before any calibration (a graph with none is never walked)."""
    repr_ = model.get_mapper_repr()
    consumers = repr_.consumer_map()
    return [n for n in repr_.execution_order()
            if isinstance(n, ComputeOpMapper) and encoded_untrained_boundary(n, consumers)]


@contextlib.contextmanager
def _calibration_mode(model):
    """Eval mode (no running-statistic update, the deployed function) and the run's random state preserved."""
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            yield
    finally:
        for m, training in modes:
            m.training = training


def _lift_shift(op, sigma: float) -> torch.Tensor:
    """The lift as an entry core's axons receive it: sigma over the op's emitted gauge kappa (wire units)."""
    kappa = float(torch.as_tensor(op.output_scale).float().mean())
    return torch.tensor(sigma / max(kappa, 1e-12))


def _bake(op, sigma: float, consumers) -> None:
    """[U-RSHIFT, kept for its importers] The lift over every axon of each consumer today's admission reaches, through
    the one writer: an entry core's effective bias (``W_eff . sigma / kappa``), a host bias carrier's bias; a LayerNorm
    absorbs it. The seam itself bakes each reader's received field (``bake_received_shift``)."""
    frontier = list(boundary_consumers(op, consumers))
    while frontier:
        consumer = frontier.pop()
        if not _is_host_node(consumer):
            perceptron = _perceptron_of(consumer)
            if perceptron is not None:
                correct_for_received_shift(perceptron, every_axon_shift(perceptron, _lift_shift(op, sigma)))
            continue
        module = getattr(consumer, "module", None)
        if _bake_shift_into_host_bias(module, sigma) or isinstance(module, torch.nn.LayerNorm):
            continue
        if _lift_admits(module):
            frontier.extend(boundary_consumers(consumer, consumers))


def _entries(op, consumers) -> list:
    """The core perceptrons the lift reaches through the host ops a lift may cross (today's admission): the mechanism's
    choice reads them (a core another lift reaches takes Soft Core Mapping's policy), never a correction."""
    found, frontier = [], list(boundary_consumers(op, consumers))
    while frontier:
        consumer = frontier.pop()
        if not _is_host_node(consumer):
            perceptron = _perceptron_of(consumer)
            if perceptron is not None:
                found.append(perceptron)
        elif _lift_admits(getattr(consumer, "module", None)):
            frontier.extend(boundary_consumers(consumer, consumers))
    return found


def _padded_reads(op, consumers) -> list:
    """The core mappers the lift reaches that pad their input (a zero-padded convolution): their padded taps read no
    sigma."""
    found, frontier = [], list(boundary_consumers(op, consumers))
    while frontier:
        consumer = frontier.pop()
        if not _is_host_node(consumer):
            if any(int(p) for p in getattr(consumer, "padding", ()) or ()):
                found.append(consumer)
        elif _lift_admits(getattr(consumer, "module", None)):
            frontier.extend(boundary_consumers(consumer, consumers))
    return found


def _carry_padding(op, consumers) -> bool:
    """Move the zero padding of the convolutions reading ``op`` into ``op``'s own output (``ZeroPaddedOutput``), so the
    lift reaches the padded zeros; True when nothing pads or the padding moved, False when it cannot (a padded read
    that is not a direct convolution read of a channel-first output, or reads padding differently)."""
    padded = _padded_reads(op, consumers)
    if not padded:
        return True
    direct = consumers.get(id(op), [])
    shape = tuple(getattr(op, "output_shape", None) or ())
    pads = {tuple(c.padding) for c in direct if isinstance(c, Conv2DPerceptronMapper)}
    if (len(shape) != 3 or len(pads) != 1 or op.module_kwargs or op.output_index is not None
            or any(not isinstance(c, Conv2DPerceptronMapper) for c in direct)
            or any(c not in direct for c in padded)):
        return False
    (pad,) = pads
    op.module = ZeroPaddedOutput(op.module, pad)
    op.output_shape = (shape[0], shape[1] + 2 * pad[0], shape[2] + 2 * pad[1])
    for conv in direct:
        conv.padding = (0, 0)
    return True


def _shift_left(model, left: list, calibration_x: torch.Tensor, *, simulation_steps: int, point: SpikingPoint) -> list:
    """[review M2] Soft Core Mapping's own policy on the boundaries no lift carries: its minima (its calibration walk at
    the run's point, in its chunks), its shift and bake (``apply_negative_value_shifts``, refusing a topology it cannot bake by
    name, as at Soft Core Mapping), and its recheck that no encoded boundary stays negative. Returns the ops shifted."""
    forward = calibration_forward_for_point(point)
    minima = {op: m for op, m in calibrated_compute_op_minima(model, calibration_x, int(simulation_steps),
                                                               forward_fn=forward).items() if op in left}
    shifted = list(apply_negative_value_shifts(model, minima))
    remaining = lossy_negative_boundaries(model, minima)
    if remaining:
        raise NotImplementedError(
            f"negative-boundary seam: Soft Core Mapping's calibrated shift left {len(remaining)} boundary/boundaries "
            f"no lift carries negative while an on-chip segment encodes them: "
            f"{[getattr(op, 'name', None) or repr(op) for op in remaining]}")
    for op in shifted:
        op.negative_shift_at_seam = True
    return shifted


def lift_negative_boundaries(model, calibration_x: torch.Tensor, *, simulation_steps: int, input_data_scale: float,
                             point: SpikingPoint, shift_enabled: bool = True, quantile: float = SEAM_QUANTILE) -> list:
    """Apply the negative-value boundary policy before any walk: returns the names of the boundaries no lift carries,
    each given Soft Core Mapping's own policy here (empty when every signed boundary was lifted, or none was signed)."""
    if not negative_boundary_candidates(model):
        return []
    repr_ = model.get_mapper_repr()
    samples: dict = {}
    driver = SegmentForwardDriver(repr_, int(simulation_steps), AnalyticalSegmentPolicy())
    with _calibration_mode(model):
        # In chunks: one pass over Soft Core Mapping's whole calibration batch (2 x 1,024 E01 images) held every host
        # value of a 110-layer program at once and ran S4's session past its 28G class (U-SC2 A-3).
        for chunk in calibration_x.split(CALIBRATION_CHUNK):
            driver(chunk, compute_sample_recorder=samples)
    consumers = repr_.consumer_map()
    signed = {op: v for op, v in samples.items()
              if float(v.min()) < -NEGATIVE_TOLERANCE and encoded_untrained_boundary(op, consumers)}
    if not signed:
        return []
    if not shift_enabled:
        if subsume_forward_negative_boundaries(model, {op: v.min().reshape(1) for op, v in signed.items()}):
            compute_per_source_scales(repr_)
            propagate_boundary_input_scales(model, input_data_scale=float(input_data_scale))
        return []
    table = read_boundary_out_scales(repr_, input_data_scale=float(input_data_scale))
    admitted = _lifts_admitted(repr_, list(signed), consumers)
    lifted, left = {}, []
    reached: dict = {}
    for op in signed:
        for p in _entries(op, consumers):
            reached[id(p)] = reached.get(id(p), 0) + 1
    for op, values in signed.items():
        entries = _entries(op, consumers)
        # A carrier is a wire value op whose consumers all compensate, reaching cores no other lift or earlier bake
        # reaches (an entry's bake is once per core).
        if (not getattr(op, "is_wire_value_op", False) or id(op) not in admitted
                or any(reached[id(p)] > 1 or getattr(p, "_neg_shift_baked", False) for p in entries)
                or not _carry_padding(op, consumers)):
            left.append(getattr(op, "name", None) or type(op.module).__name__)
            continue
        sigma, kappa = _signed_seam_quantiles(values, quantile=quantile)
        if op.per_source_scales is None:  # a marked value op on a unit-gauge chain: armed at its pass-through currency
            arm_wrap_slots(op, [1.0] * len(op._sources_list), float(table.get(op, 1.0)))
        op.output_value_offset = torch.tensor(float(sigma))
        op.boundary_traffic_scale = float(kappa)
        lifted[op] = float(sigma)
    if lifted:
        compute_per_source_scales(repr_)
        propagate_boundary_input_scales(model, input_data_scale=float(input_data_scale))
        bake_received_shift(repr_, {op: _lift_shift(op, sigma) for op, sigma in lifted.items()})
        print(f"[NEGATIVE-BOUNDARY SEAM] lifted {len(lifted)} signed boundary/boundaries (name, sigma, kappa): "
              f"{[(op.name, round(s_, 6), round(float(op.boundary_traffic_scale), 6)) for op, s_ in lifted.items()]}",
              flush=True)
    if left:
        left_ops = [op for op in signed if op not in lifted]
        with _calibration_mode(model):
            shifted = _shift_left(model, left_ops, calibration_x, simulation_steps=simulation_steps, point=point)
        print(f"[NEGATIVE-BOUNDARY SEAM] {len(left)} signed boundary/boundaries no lift carries: {left}; Soft Core "
              f"Mapping's shift applied at the seam to {[(op.name, op._negative_shift.tolist()) for op in shifted]}",
              flush=True)
    return left


@dataclass(frozen=True)
class NegativeBoundarySeam:
    """The policy as the gauge seam calls it: the calibration batch, the run's switches and [U-LIF2 r2] the run's resolved
    point (the record Soft Core Mapping's walk at the seam is built from), bound once."""

    calibration_x: torch.Tensor
    simulation_steps: int
    input_data_scale: float
    point: SpikingPoint
    shift_enabled: bool = True

    def __call__(self, model) -> list:
        return lift_negative_boundaries(model, self.calibration_x, simulation_steps=self.simulation_steps,
                                        input_data_scale=self.input_data_scale, point=self.point,
                                        shift_enabled=self.shift_enabled)
