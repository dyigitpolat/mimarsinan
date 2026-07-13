from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.models.nn.layers import *
from mimarsinan.models.nn.activations import LeakyGradReLU
from mimarsinan.models.nn.decorators.clamp_quantize import (
    LIFCountStaircaseDecorator,
    TTFSCeilStaircaseDecorator,
)
from mimarsinan.models.nn.decorators.rate_buffer import RateBuffer
from mimarsinan.tuning.orchestration.frontier import frontier_position
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    lif_exact_qat_active,
    mark_lif_exact_qat,
)
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

import torch.nn as nn

_SYNC_EXACT_QAT_ATTR = "_mbh_sync_exact_qat"
_SYNC_GRID_SNAP_ATTR = "_mbh_sync_grid_snap_installed"

HOP_DEPTH_ATTR = "_mbh_aq_hop_depth"
"""Per-perceptron cascade-hop depth stamped for the staged AQ install ([5v B1])."""


def hop_frontier(rate, n_levels: int) -> int:
    """Installed hop-depth count at ``rate``: the frontier-geometry SSOT mapping."""
    return frontier_position(rate, n_levels)


def sync_exact_qat_active(pipeline_config) -> bool:
    """The ``sync_exact_qat`` recipe knob is on AND the resolved mode is synchronized ttfs_cycle (T6)."""
    if not bool(pipeline_config.get("sync_exact_qat", False)):
        return False
    # Lazy: chip_simulation has a fragile import cycle with tuning at init time.
    from mimarsinan.chip_simulation.deployment_contract import (
        SpikingDeploymentContract,
    )

    return SpikingDeploymentContract.from_pipeline_config(pipeline_config).is_synchronized()


def mark_sync_exact_qat(perceptron) -> None:
    """Persistently mark ``perceptron`` as trained through the exact deployed ceil kernel."""
    setattr(perceptron, _SYNC_EXACT_QAT_ATTR, True)


def model_trained_sync_exact(model) -> bool:
    """[MBH T6] Whether the model's QAT endpoint was the exact deployed ceil kernel (all-or-none).

    Mixed marking fails loud: the mapping-time half-step bias compensation is a
    per-model decision, so a partially exact-trained model has no sound bake."""
    flags = [
        bool(getattr(p, _SYNC_EXACT_QAT_ATTR, False)) for p in model.get_perceptrons()
    ]
    if not flags or not any(flags):
        return False
    assert all(flags), (
        "MBH sync-exact QAT marker is inconsistent: "
        f"{sum(flags)}/{len(flags)} perceptrons are marked; the exact-kernel "
        "endpoint must cover every perceptron or none."
    )
    return True


def install_sync_entry_grid_snap(model, pipeline_config) -> int:
    """[MBH T6] Install the deployed per-stage input grid snap on segment-entry perceptrons.

    Boundary input scales are propagated first so each snap normalizes by the
    deployed consumer scale; the quantizer keeps the live ``input_activation_scale``
    Parameter, so later re-propagation stays coherent. Idempotent; returns the
    number of quantizers installed (0 when the knob/mode gate is off)."""
    if not sync_exact_qat_active(pipeline_config):
        return 0
    from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
    # Lazy: the spiking package init pulls chip_simulation (import cycle).
    from mimarsinan.spiking.scale_aware_boundaries import (
        propagate_boundary_input_scales,
    )
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons

    propagate_boundary_input_scales(
        model,
        input_data_scale=ResolvedWorkloadProfile.from_config(
            pipeline_config
        ).input_data_scale,
    )
    installed = 0
    for perceptron in segment_entry_perceptrons(model.get_mapper_repr()):
        if getattr(perceptron, _SYNC_GRID_SNAP_ATTR, False):
            continue
        quantizer = TTFSInputGridQuantizer(
            T=int(pipeline_config["simulation_steps"]),
            activation_scale=perceptron.input_activation_scale,
        )
        perceptron.append_input_wire_op(quantizer)
        setattr(perceptron, _SYNC_GRID_SNAP_ATTR, True)
        installed += 1
    return installed


class AdaptationManager(nn.Module):
    def __init__(self):
        super(AdaptationManager, self).__init__()

        self.activation_adaptation_rate = 0.0
        self.clamp_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0
        self.scale_rate = 0.0
        self.pruning_rate = 0.0

        self.noise_rate = 0.0

        self.lif_active = False
        self.ttfs_active = False

        # [5v B1] hop-staged sync AQ install: None = monolithic; an int arms
        # the depth frontier (hops below ceil(rate*n) install at full rate).
        self.quantization_hop_levels = None

        self._rate_buffers = {}

    def bind_rate_buffer(self, rate_attr):
        """Return the shared ``RateBuffer`` for ``rate_attr``, creating it once.

        Seeded from the field's current float so the first rebuild is value-
        equivalent to the float it replaces.
        """
        buffers = getattr(self, "_rate_buffers", None)
        if buffers is None:
            buffers = {}
            self._rate_buffers = buffers
        buffer = buffers.get(rate_attr)
        if buffer is None:
            buffer = RateBuffer()
            buffer.set(float(getattr(self, rate_attr)))
            buffers[rate_attr] = buffer
        return buffer

    def _rate_buffer(self, rate_attr):
        return getattr(self, "_rate_buffers", {}).get(rate_attr)

    def _rate_carrier(self, rate_attr):
        """The decorator's rate source: the bound buffer if any, else the float."""
        buffer = self._rate_buffer(rate_attr)
        return buffer if buffer is not None else getattr(self, rate_attr)

    def _rate_value(self, rate_attr) -> float:
        """The current rate as a float (reads through a bound buffer)."""
        carrier = self._rate_carrier(rate_attr)
        return float(carrier)

    def _rate_is_active(self, rate_attr, float_active):
        """Whether to include ``rate_attr``'s decorator (gates the rebuild stack).

        A bound buffer counts as active even at alpha 0.0 so the buffer-backed
        stack stays stable across the whole ramp (no rebuild); no buffer falls back
        to the float predicate (byte-identical)."""
        if self._rate_buffer(rate_attr) is not None:
            return True
        return float_active

    def update_activation(self, pipeline_config, perceptron):
        from mimarsinan.chip_simulation.spiking_semantics import (
            is_lif,
            requires_ttfs_firing,
        )

        spiking_mode = pipeline_config.get("spiking_mode", "lif")
        use_ttfs = requires_ttfs_firing(spiking_mode)
        runtime_subsumed = (
            getattr(self, "lif_active", False)
            or getattr(self, "ttfs_active", False)
        )
        subsumes_decorators = runtime_subsumed or is_lif(spiking_mode)
        # [lif_exact_qat_program §6.1(2)] the exact arm un-subsumes ONLY the
        # quantization decorator pre-finalize: the AQ stage hosts the staircase
        # QAT; once LIF activations install (lif_active), the node subsumes it
        # again (re-quantizing on-grid values would drop a level at strict '<').
        quantization_subsumed = runtime_subsumed or (
            is_lif(spiking_mode) and not lif_exact_qat_active(pipeline_config)
        )
        decorators = []
        if self._rate_is_active("activation_adaptation_rate", self.activation_adaptation_rate > 0):
            decorators.append(
                self.get_rate_adjusted_activation_replacement_decorator(perceptron))
        if not subsumes_decorators and self._rate_is_active("clamp_rate", self.clamp_rate != 0.0):
            decorators.append(self.get_rate_adjusted_clamp_decorator(perceptron))
        if not quantization_subsumed and self._rate_is_active("quantization_rate", self.quantization_rate != 0.0):
            quant_decorator = self.get_rate_adjusted_quantization_decorator(
                pipeline_config, perceptron)
            if quant_decorator is not None:
                decorators.append(quant_decorator)
        if not subsumes_decorators and not use_ttfs and self.shift_rate != 0.0:
            decorators.append(self.get_shift_decorator(pipeline_config, perceptron))

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

        if self.noise_rate > 0:
            target_noise_amount = (1.0 / (pipeline_config['target_tq'] * 2.5))
            perceptron.set_regularization(
                NoisyDropout(torch.tensor(0.0), self.noise_rate, target_noise_amount * perceptron.activation_scale)
            )
        else:
            perceptron.set_regularization(nn.Identity())

    def get_rate_adjusted_activation_replacement_decorator(self, perceptron):
        """Gradually blend the base activation toward LeakyGradReLU (chip ReLU)."""
        return RateAdjustedDecorator(
            self._rate_carrier("activation_adaptation_rate"),
            ActivationReplacementDecorator(LeakyGradReLU()),
            MixAdjustmentStrategy())

    def get_rate_adjusted_clamp_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self._rate_carrier("clamp_rate"),
            ClampDecorator(torch.tensor(0.0), perceptron.activation_scale),
            MixAdjustmentStrategy())

    def get_shift_decorator(self, pipeline_config, perceptron):
        shift_amount = calculate_activation_shift(pipeline_config["target_tq"], perceptron.activation_scale) * self.shift_rate
        return ShiftDecorator(shift_amount)
    
    def get_rate_adjusted_quantization_decorator(self, pipeline_config, perceptron):
        from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

        if lif_exact_qat_active(pipeline_config):
            # [lif_exact_qat_program §6.1] the QAT endpoint is the deployed LIF
            # count staircase itself (theta in-loop rides the same Parameter);
            # the marker is per-MODEL and owns the half-step fold (P-L6).
            mark_lif_exact_qat(perceptron)
            return RateAdjustedDecorator(
                self._rate_carrier("quantization_rate"),
                LIFCountStaircaseDecorator(
                    pipeline_config["simulation_steps"],
                    perceptron.activation_scale,
                    thresholding_mode=str(
                        pipeline_config.get("thresholding_mode", "<=")
                    ),
                ),
                NestedAdjustmentStrategy(
                    [RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]
                ))

        if sync_exact_qat_active(pipeline_config):
            # [MBH T6] The QAT endpoint is the deployed ceil kernel itself; the
            # floor + half-step proxy (and its mapping-time bias compensation)
            # is bypassed for models trained this way. The marker is per-MODEL
            # (staged hops beyond the frontier still end exact by rate 1.0).
            mark_sync_exact_qat(perceptron)
            rate_carrier = self._rate_carrier("quantization_rate")
            n_levels = getattr(self, "quantization_hop_levels", None)
            if n_levels:
                # [5v B1] the hop frontier: convert hops 0..k at FULL rate,
                # keep hops beyond it float (no decorator).
                k = hop_frontier(self._rate_value("quantization_rate"), n_levels)
                if int(getattr(perceptron, HOP_DEPTH_ATTR, 0)) >= k:
                    return None
                rate_carrier = 1.0
            return RateAdjustedDecorator(
                rate_carrier,
                TTFSCeilStaircaseDecorator(
                    pipeline_config["simulation_steps"], perceptron.activation_scale
                ),
                NestedAdjustmentStrategy(
                    [RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]
                ))

        use_ttfs = requires_ttfs_firing(pipeline_config.get("spiking_mode", "lif"))
        shift = calculate_activation_shift(
            pipeline_config["target_tq"], perceptron.activation_scale
        )
        if use_ttfs:
            shift_back_amount = -shift
        else:
            shift_back_amount = -shift * self.shift_rate

        return RateAdjustedDecorator(
            self._rate_carrier("quantization_rate"),
            NestedDecoration(
                [ShiftDecorator(shift_back_amount),
                QuantizeDecorator(torch.tensor(pipeline_config["target_tq"]), perceptron.activation_scale)]),
            NestedAdjustmentStrategy([RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]))
    