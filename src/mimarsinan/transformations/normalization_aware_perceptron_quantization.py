from mimarsinan.transformations.transformation_utils import *

from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive
from mimarsinan.transformations.perceptron.bias_saturation import (
    clip_off_saturated_effective_bias,
)
from mimarsinan.transformations.perceptron.input_shift import recorded_input_shift
from mimarsinan.transformations.perceptron.neuron_scales import is_per_neuron_scale, row_peaks, scale_view
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import output_axes
from mimarsinan.transformations.quantization_bounds import layer_weight_bits
from mimarsinan.models.perceptron_mixer.membrane_lattice import record_membrane_lattice_kind

import torch

class NormalizationAwarePerceptronQuantization:
    """Normalization-aware per-perceptron weight quantization.

    Interpolates in weight-value space between the FP and fully-quantized
    effective weights (``rate=0`` identity, ``rate=1`` == legacy full quant).
    ``parameter_scale`` is always the full-range (``rate=1``) scale.

    ``two_scale=True`` derives the weight grid ``s_w`` from ``max|w|`` alone, and
    the effective bias lands where the platform delivers it:

    * bias rows (param-encoded, bias-row splitting; the default): ``r`` always-on
      rows of ±q_max weights, so the bias grid is integer-ratio-snapped to the
      weight grid (``bias_scale = parameter_scale / r``, ``r = ceil(max|b| s_w /
      q_max)``) and every row carries an equal share;
    * an on-chip bias register (``register_bias=True``): the bias is coded on the
      weight lattice itself, ``round(b s_w)``, within the span ``[-(q_max+1) r,
      q_max r]`` the export allocates (it never binds here), so ``bias_scale ==
      parameter_scale``: a coarser grid would buy no register and cost up to
      ``r / 2`` weight steps per neuron (phase C defect D-WQ, cause A).

    Either way the quantized bias is an exact integer on the weight-grid lattice —
    the torch<->chip parity contract (``wq_cascade_crater_repair.md`` §4.3/§5) —
    and the perceptron records the kind of membrane lattice it lives on (the
    lattice itself, at ``s_w``, is derived from the persisted scale at every
    arming: ``models.perceptron_mixer.membrane_lattice``).

    A perceptron whose bias compensates a recorded input shift (``input_shift``,
    the negative-boundary bake ``B - W s``) has the compensation re-derived on
    the weights each projection installs, before its bias is coded, on every
    route and at every rate (D-WQ, cause B). On the shared grid the scale is set
    before that compensation, so a compensated bias beyond the grid's top
    saturates as any bias there does.

    An always-on row carries its drive ``d`` (``transformations.perceptron.bias_drive``: the top code value of its
    reader's input grid on the integer value path, else 1.0): the row's weight is ``b / d``, so the shared grid sizes
    ``max(max|w|, max|b| / d)``, the rows count ``ceil(max|b| s_w / (d q_max))`` and the projected bias is a row code
    times ``d / s``.

    Declared value-domain grids (Wave 2): ``fractional_scale`` takes the scale
    ``q_max / max`` instead of its integer floor (the law's multiplier carries
    it); ``float_bias`` derives the weight grid from ``max|w|`` alone and leaves
    the bias float (the readout law rounds it at its own resolution).

    [U-PCH] ``per_neuron`` (a chip declaring a threshold register per neuron, ``weight_scale_granularity
    per_neuron``) sizes each grid per output neuron: ``s_c`` from row ``c``'s peaks, ``theta_c = s_c``, a membrane
    lattice per neuron; the bias rows a param-encoded bias takes are shared by the perceptron's neurons, so their
    count is the largest neuron's.
    """

    def __init__(self, bits, device, rate=1.0, two_scale=False, fractional_scale=False, float_bias=False,
                 bias_rows_floor=None, register_bias=False, per_neuron=False):
        self.device = device
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.rate = rate
        self.two_scale = bool(two_scale)
        self.fractional_scale = bool(fractional_scale)
        self.float_bias = bool(float_bias)
        # Bias-row splitting's declared k: a FLOOR under the computed bound,
        # never a cap — a bias the declared k outgrows buys rows rather than
        # saturating at the register edge.
        self.bias_rows_floor = None if bias_rows_floor is None else int(bias_rows_floor)
        # The two-scale bias route: an on-chip register (the weight lattice) or bias rows (the r-multiple grid).
        self.register_bias = bool(register_bias)
        self.per_neuron = bool(per_neuron)

    def transform(self, perceptron):
        # [WI-5g] a perceptron on a core type of its own weight width quantizes on that width's grid.
        bits = layer_weight_bits(perceptron, self.bits)
        projection = self if bits == self.bits else NormalizationAwarePerceptronQuantization(
            bits, self.device, self.rate, two_scale=self.two_scale, fractional_scale=self.fractional_scale,
            float_bias=self.float_bias, register_bias=self.register_bias, per_neuron=self.per_neuron)
        transformer = PerceptronTransformer(positions=True)  # [U-PFOLD] a fold per position: each weight row projected
        with transformer.deferred_finite_checks(perceptron):
            projection._transform(perceptron, transformer)

    def _grid_scale(self, peak):
        """The weight grid's scale for ``peak = max|·|``: the integer floor (chip threshold register) or
        ``q_max / peak`` under a declared fractional rule."""
        quotient = self.q_max / torch.clamp(peak, min=1e-12)
        return quotient if self.fractional_scale else torch.clamp(torch.floor(quotient), min=1.0)

    def _peak(self, t, axis):
        """``max |t|``: over the whole tensor, or per output neuron on ``axis`` (``per_neuron``; [U-PFOLD r2] the axis
        the reader declares for the view ``t``)."""
        return row_peaks(t, axis) if self.per_neuron else torch.max(torch.abs(t))

    def _transform(self, perceptron, transformer):
        if self.float_bias:
            # No bias grid exists, so no bias mass can starve it: the projection leaves the bias as trained.
            self._transform_float_bias(perceptron, transformer)
            return
        # The scale must never be set by functionally-unobservable bias mass:
        # a constant-OFF channel's outlier bias starves the shared weight grid
        # (the t01_19/t0_03 WQ-entry crater). Idempotent, function-preserving.
        clip_off_saturated_effective_bias(perceptron, transformer=transformer)
        axes = output_axes(perceptron, transformer.positions)  # [U-PFOLD r2] each view's output axis, the reader's
        w = transformer.get_effective_weight(perceptron)
        b = transformer.get_effective_bias(perceptron)
        # The always-on row carries its drive (its reader's grid top on the integer value path, else 1.0): its
        # weight is the bias over the drive, which is what the shared grid and the rows must hold.
        drive = bias_row_drive(perceptron)

        # All-tensor scale chain: this runs per perceptron per optimizer step,
        # and a python max()/CPU-tensor bound forces a GPU sync per call.
        w_max = self._peak(w, axes.weight)
        b_max = self._peak(b, axes.bias)
        if drive != 1.0:
            b_max = b_max / drive
        if torch.is_tensor(b_max) and b_max.device != w_max.device:
            b_max = b_max.to(w_max.device)

        # A bias-free BN-paired layer has a derived effective bias that IS
        # realizable (through the normalization's beta seam), so it takes the
        # two-scale path too; only a perceptron with no seam at all -- whose
        # effective bias is structurally zero -- falls back to the weight grid.
        if self.two_scale and transformer.effective_bias_is_writable(perceptron):
            self._transform_two_scale(perceptron, transformer, w_max, b_max, drive)
            return

        p_max = torch.clamp(torch.maximum(w_max, b_max), min=1e-12)

        # [nevresim parity] the deployed threshold register is INTEGER
        # (threshold_t=int; theta == this scale after chip quantization):
        # snap the shared grid to the integer lattice so the trained function
        # IS the chip function — a fractional scale would be truncated at
        # emit and deploy a different fire condition than the SSOT simulates.
        # FLOOR, not round: p_max*scale <= q_max keeps every quantized
        # parameter inside its register (no clip saturation, relays keep
        # their strict-comparator margin).
        scale = self._grid_scale(p_max)

        # set_parameter_scale re-declares the shared grid (bias_scale follows).
        perceptron.set_parameter_scale(scale)
        record_membrane_lattice_kind(perceptron)

        self._project_weights(perceptron, transformer, scale)
        # The row's code is round(b s / drive): the bias on the grid of step drive / s.
        transformer.apply_effective_bias_transform(perceptron, self._quantize_param_fn(
            scale / drive if drive != 1.0 else scale, output_axes(perceptron, transformer.positions).bias))

    def _project_weights(self, perceptron, transformer, scale) -> None:
        """The weight write on ``scale``'s grid; a recorded input shift's compensation follows the weights it
        installs (``B' <- B' - (W~_after - W~_before) s``, through the received shift's one writer)."""
        # mapping.support imports transformations at its own import: a module-scope import here is a cycle.
        from mimarsinan.mapping.support.received_shift import compensate_projected_weights

        shifted = recorded_input_shift(perceptron) is not None
        before = transformer.get_effective_weight(perceptron).clone() if shifted else None
        transformer.apply_effective_weight_transform(
            perceptron, self._quantize_param_fn(scale, output_axes(perceptron, transformer.positions).weight))
        if before is not None:
            compensate_projected_weights(perceptron, transformer, before)

    def _transform_two_scale(self, perceptron, transformer, w_max, b_max, drive=1.0):
        # [nevresim parity] integer-lattice weight grid (see single-scale note).
        weight_scale = self._grid_scale(w_max)
        # A compensated bias is the one the bias grid is sized for: its weights are projected first.
        projected = recorded_input_shift(perceptron) is not None
        if projected:
            self._project_weights(perceptron, transformer, weight_scale)
            b_max = self._peak(transformer.get_effective_bias(perceptron),
                               output_axes(perceptron, transformer.positions).bias).to(w_max.device)
            if drive != 1.0:
                b_max = b_max / drive
        if self.register_bias and drive != 1.0:
            raise ValueError("an on-chip bias register has no always-on row, so no row drive: the WQ step stamps none")
        # r whole weight-grid steps cover the bias: r IS the bias-row count a
        # param-encoded platform maps and the span (±q_max r, weight-grid units)
        # an on-chip register holds, so the formula lives in the bias-row SSOT
        # (mapping's import of transformations makes a module-scope import a cycle).
        from mimarsinan.mapping.support.bias_rows import bias_row_demand

        # tol=0: this ratio predates bias-row splitting and its grid stays
        # bit-identical (a tolerance here moved a t0_29 lineage read 0.9919 ->
        # 0.9912 before it was caught).
        ratio = torch.clamp(
            torch.ceil(bias_row_demand(b_max, weight_scale, self.q_max)), min=1.0,
        )
        if self.bias_rows_floor is not None:
            ratio = torch.clamp(ratio, min=float(self.bias_rows_floor))
        if not self.register_bias and is_per_neuron_scale(ratio):
            ratio = ratio.max()  # the always-on rows are every neuron's: as many as its largest neuron needs
        axes = output_axes(perceptron, transformer.positions)  # [U-PFOLD r2] each view's output axis, the reader's
        if self.register_bias:
            # The register: round(b s_w) on the weight lattice within the span ±q_max r (never binding here).
            bias_scale, bias_quantize = weight_scale, self._quantize_param_fn(weight_scale, axes.bias, span=ratio)
        else:
            # The rows: an integer-ratio-snapped grid, so `bias * weight_scale = r * bias_int` exactly and each of
            # the r rows emits bias_int (the lattice the chip export emits and the NF<->SCM parity consumes). Rows
            # driven at `drive` hold the bias over the drive: `(bias / drive) * weight_scale = r * bias_int`.
            bias_scale = weight_scale / ratio
            bias_quantize = self._quantize_param_fn(bias_scale / drive if drive != 1.0 else bias_scale, axes.bias)

        perceptron.set_parameter_scale(weight_scale)
        perceptron.set_bias_scale(bias_scale)
        record_membrane_lattice_kind(perceptron)

        if not projected:
            transformer.apply_effective_weight_transform(
                perceptron, self._quantize_param_fn(weight_scale, axes.weight)
            )
        transformer.apply_effective_bias_transform(perceptron, bias_quantize)

    def _transform_float_bias(self, perceptron, transformer):
        weight_scale = self._grid_scale(self._peak(transformer.get_effective_weight(perceptron),
                                                   output_axes(perceptron, transformer.positions).weight))
        perceptron.set_parameter_scale(weight_scale)
        record_membrane_lattice_kind(perceptron)
        self._project_weights(perceptron, transformer, weight_scale)

    def _quantize_param_fn(self, scale, axis, span=None):
        """Round onto ``scale``'s grid, clamped to the register (``±q_max``, or ``±q_max span`` codes), mixed at the
        rate; a per-neuron scale and span read each output neuron on its own (``scale_view``), on ``axis``, the output
        axis the reader declares for the view the transform receives."""
        rate = float(self.rate)
        q_min = float(self.q_min) if span is None else self.q_min * span
        q_max = float(self.q_max) if span is None else self.q_max * span

        def quantize_param(param):
            s = scale_view(scale, param, axis)
            scaled = param * s
            quantized = torch.round(scaled).clamp_(min=scale_view(q_min, param, axis),
                                                   max=scale_view(q_max, param, axis))
            rescaled = quantized / s
            if rate >= 1.0:
                return rescaled
            if rate <= 0.0:
                return param
            return rate * rescaled + (1.0 - rate) * param

        return quantize_param


COMPARATOR_HALF_STEP_ARM_RATIO = 0.5
"""[G6] erosion ratios >= ~0.5 predict the baked half-step erodes on the bias
lattice (sync memo §5: measured 0.21-1.41 at 5 bits) — arm the E3 comparator."""


def half_step_bias_lattice_erosion_ratio(perceptron, simulation_steps) -> float:
    """[G6/E3] ``g_b / (1/(2S))``: the effective-bias grid step (``1/bias_scale``,
    already θ-normalized) over the normalized half-step ``1/(2S)``."""
    return float(2 * int(simulation_steps)) / float(perceptron.bias_scale)


def comparator_half_step_advisory(
    model, simulation_steps, *, arm_threshold: float = COMPARATOR_HALF_STEP_ARM_RATIO,
) -> dict:
    """Per-perceptron erosion ratios at projection + the arming verdict
    (``armed`` when the WORST hop's ratio reaches ``arm_threshold``)."""
    ratios = [
        half_step_bias_lattice_erosion_ratio(p, simulation_steps)
        for p in model.get_perceptrons()
    ]
    max_ratio = max(ratios, default=0.0)
    return {
        "ratios": ratios,
        "max_ratio": max_ratio,
        "arm_threshold": float(arm_threshold),
        "armed": bool(max_ratio >= float(arm_threshold)),
    }
