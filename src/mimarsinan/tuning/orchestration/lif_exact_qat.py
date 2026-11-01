"""[lif_exact_qat_program.md §6] LIF exact-QAT arm: predicate, markers, installers."""

from __future__ import annotations

from typing import Any

import torch

from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer

_LIF_EXACT_QAT_ATTR = "_mbh_lif_exact_qat"
_LIF_ENTRY_SNAP_ATTR = "_lif_entry_input_quantizer_installed"

# The rate ladders whose decorators the LIF spiking node subsumes (see
# ``AdaptationManager.update_activation``) — behaviorally inert in LIF mode,
# measured in MBH X1. Under the exact-QAT arm the QUANTIZATION ladder is
# un-subsumed: it hosts the staircase exact-QAT (the AQ stage's rungs train).
_MBH_LIF_SUBSUMED_RATE_ATTRS = frozenset({"clamp_rate", "quantization_rate"})


def lif_exact_qat_active(pipeline_config) -> bool:
    """The ``lif_exact_qat`` config knob is on AND the P-L5/R5 preconditions hold.

    Non-LIF modes read False (the knob is mode-scoped); a LIF config that arms
    the knob with a broken precondition fails LOUD — silent disarming would be
    a silent Goodhart hole (lif_exact_qat_program.md §4.5, §6.5).
    """
    if not bool(pipeline_config.get("lif_exact_qat", False)):
        return False
    # Lazy: chip_simulation has a fragile import cycle with tuning at init time.
    from mimarsinan.chip_simulation.spiking_semantics import is_lif

    if not is_lif(str(pipeline_config.get("spiking_mode", "lif"))):
        return False
    firing_mode = str(pipeline_config.get("firing_mode", "Default"))
    if firing_mode != "Default":
        raise ValueError(
            "lif_exact_qat requires firing_mode='Default' (P-L5): Novena's "
            "zero reset breaks the Theorem-0 charge identity, so the count "
            "staircase is the wrong QAT anchor."
        )
    if not bool(pipeline_config.get("cycle_accurate_lif_forward", True)):
        raise ValueError(
            "lif_exact_qat requires the cycle-accurate LIF forward (P-L5): the "
            "value-domain proxy is not the deployed composition."
        )
    if not bool(pipeline_config.get("lif_per_hop_retiming", False)):
        raise ValueError(
            "lif_exact_qat requires lif_per_hop_retiming: staircase-QAT weights "
            "deployed on the raw single-window cascade are the measured Goodhart "
            "hole (-2.5 pp, lif_exact_qat_program.md §5); config derivation "
            "auto-pairs the knobs."
        )
    return True


def lif_exact_qat_kd_active(pipeline_config) -> bool:
    """The ``lif_exact_qat_kd`` knob is on AND the exact-QAT arm is active.

    [lif_exact_qat_program §8] Composes the exact-QAT predicate, so a broken
    exact precondition fails LOUD through here too; the KD lever is meaningful
    only at the exact endpoint (with the exact arm off it is a silent no-op).
    """
    if not bool(pipeline_config.get("lif_exact_qat_kd", False)):
        return False
    return lif_exact_qat_active(pipeline_config)


def lif_subsumed_ladder_steps(pipeline_config, rate_attr, steps):
    """Fast-ladder training steps, with the LIF-subsumed ladders dropped (X3 default).

    In lif mode the spiking node subsumes ``rate_attr``'s decorator, so its
    ladder is behaviorally inert (measured, X1/X2) and trains 0 steps — EXCEPT
    the quantization ladder under the exact-QAT arm, which hosts the staircase
    QAT and keeps its budget. Otherwise ``steps`` unchanged.
    """
    if rate_attr not in _MBH_LIF_SUBSUMED_RATE_ATTRS:
        return int(steps)
    # Lazy: chip_simulation has a fragile import cycle with tuning at init time.
    from mimarsinan.chip_simulation.spiking_semantics import is_lif

    if not is_lif(pipeline_config.get("spiking_mode", "lif")):
        return int(steps)
    if rate_attr == "quantization_rate" and lif_exact_qat_active(pipeline_config):
        return int(steps)
    return 0


def mark_lif_exact_qat(perceptron) -> None:
    """Persistently mark ``perceptron`` as trained through the exact LIF count staircase."""
    setattr(perceptron, _LIF_EXACT_QAT_ATTR, True)


def model_trained_lif_exact(model) -> bool:
    """Whether the model's AQ endpoint was the exact LIF count staircase (all-or-none).

    Mixed marking fails loud: the half-step fold ownership and the re-timing
    pairing are per-model decisions, so a partially marked model has no sound
    deployment convention."""
    flags = [
        bool(getattr(p, _LIF_EXACT_QAT_ATTR, False)) for p in model.get_perceptrons()
    ]
    if not flags or not any(flags):
        return False
    assert all(flags), (
        "MBH lif-exact QAT marker is inconsistent: "
        f"{sum(flags)}/{len(flags)} perceptrons are marked; the exact-staircase "
        "endpoint must cover every perceptron or none."
    )
    return True


def deployed_lif_gauge_forward(clone, pipeline_config):
    """[PR18, spiking_deployment_calculus §14.4] The deployed-composition gauge
    for an exact-QAT candidate: install fresh LIF activations at each
    perceptron's own theta on ``clone`` and return the (retimed) chip-aligned
    genuine forward — the A2 square's deploy side, so the QAT is measured and
    gated on what actually ships."""
    # Lazy: chip_simulation/spiking pull import cycles at init (house pattern).
    from mimarsinan.chip_simulation.spiking_semantics import (
        lif_execution_synchronized,
        lif_membrane_init,
        lif_per_hop_retiming_enabled,
        spike_phase_dither_enabled,
    )
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
    from mimarsinan.spiking.wire_gauge_seam import require_established_wire_gauge

    # [U-SC S-1] the walk is a wire-currency twin: refuse a clone no stage established, re-derive a stamped one.
    require_established_wire_gauge(
        clone, input_data_scale=ResolvedWorkloadProfile.from_config(pipeline_config).input_data_scale,
        reader="the exact-QAT D-hat gauge (deployed_lif_gauge_forward)",
    )
    T = int(pipeline_config["simulation_steps"])
    thresholding = str(pipeline_config.get("thresholding_mode", "<="))
    firing = str(pipeline_config.get("firing_mode", "Default"))
    device = next(clone.parameters()).device
    for perceptron in clone.get_perceptrons():
        perceptron.activation = LIFActivation(
            T=T,
            activation_scale=perceptron.activation_scale,
            thresholding_mode=thresholding,
            firing_mode=firing,
            membrane_init=lif_membrane_init(pipeline_config),
        ).to(device)
    retime = lif_per_hop_retiming_enabled(pipeline_config)
    dither = spike_phase_dither_enabled(pipeline_config)
    sync = lif_execution_synchronized(pipeline_config)
    soma_law = SomaLaw.resolve(pipeline_config)

    def _deployed_forward(x):
        return chip_aligned_segment_forward(
            clone, x, T, retime=retime, phase_dither=dither, synchronized=sync,
            soma_law=soma_law,
        )

    return _deployed_forward


def install_lif_input_quantizer(perceptron, simulation_steps: int) -> bool:
    """Idempotently append the LIF entry ``ChipInputQuantizer``; True when this call installed it."""
    if getattr(perceptron, _LIF_ENTRY_SNAP_ATTR, False):
        return False
    perceptron.append_input_wire_op(
        ChipInputQuantizer(
            T=int(simulation_steps),
            activation_scale=perceptron.input_activation_scale,
        )
    )
    setattr(perceptron, _LIF_ENTRY_SNAP_ATTR, True)
    return True


def deployed_negative_boundary_seam(model, trainer: Any, pipeline_config):
    """[U-SC2 A-2] The negative-value boundary policy the gauge seam applies, bound to the calibration batch Soft Core
    Mapping reads (two validation batches) and the run's switches (``negative_value_shift``, the input data scale, the
    resolved soma point Soft Core Mapping's LIF calibration walk carries).

    ``trainer`` is a trainer or a zero-argument factory of one, called only when the graph has a boundary the policy
    could lift. None (nothing read, nothing changed) without a trainer or such a boundary, or on a signed boundary
    contract: the policy guards the unsigned [0, 1] encode clamp, as Soft Core Mapping decides. The calibration read
    preserves the run's random state and the trainer's validation cursor."""
    # Lazy: pipelining (the plan) and the seam module import back into tuning at init (house pattern).
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
    from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.tuning.orchestration.negative_boundary_seam import (
        NegativeBoundarySeam,
        negative_boundary_candidates,
    )

    if trainer is None or getattr(model, "get_mapper_repr", lambda: None)() is None:
        return None
    if packaging_contract_for(DeploymentPlan.resolve(pipeline_config)).boundary.signed:
        return None
    if not negative_boundary_candidates(model):
        return None
    with torch.random.fork_rng(devices=[]):
        source: Any = trainer() if callable(trainer) and not hasattr(trainer, "iter_validation_batches") else trainer
        cursor = getattr(source, "_gpu_val_cursor", None)
        batches = [x for x, _ in source.iter_validation_batches(2)]
        if cursor is not None:  # the read leaves the trainer's validation cursor where it was
            source._gpu_val_cursor = cursor
    if not batches:
        return None
    device = next(model.parameters()).device
    return NegativeBoundarySeam(
        calibration_x=torch.cat(batches, dim=0).to(device),
        simulation_steps=int(pipeline_config["simulation_steps"]),
        input_data_scale=float(ResolvedWorkloadProfile.from_config(pipeline_config).input_data_scale),
        soma_law=SomaLaw.resolve(pipeline_config),
        shift_enabled=bool(pipeline_config.get("negative_value_shift", True)),
    )


def install_lif_entry_input_quantizers(model, pipeline_config, *, trainer=None) -> int:
    """Install the deployed entry round at the AQ install seam so the exact-QAT
    ladder trains through it (§4.3). Placement decides the carriers: under
    ``subsume`` the encoders themselves (they deploy host-side with the
    quantizer riding along; their consumers read grid-aligned staircase
    outputs, so the boundary round is idempotent there); under ``offload`` the
    encoding marks are cleared and every SEGMENT-ENTRY core reads a host float
    through the wire-encode round — unmodeled, it read 36 pts under NF on the
    offloaded ViT. Idempotent; returns installs (0 when the arm is off)."""
    if not lif_exact_qat_active(pipeline_config):
        return 0
    # Lazy: torch_mapping/spiking pull chip_simulation (import cycle at init).
    from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
    from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
    from mimarsinan.torch_mapping.encoding_layers import (
        encoder_deploys_as_staircase_hop,
    )

    placement = str(pipeline_config.get("encoding_layer_placement", "subsume"))
    if encoder_deploys_as_staircase_hop(placement):
        targets = [
            p for p in model.get_perceptrons()
            if getattr(p, "is_encoding_layer", False)
        ]
    else:
        from mimarsinan.torch_mapping.encoding_layers import (
            segment_entry_perceptrons,
        )

        targets = segment_entry_perceptrons(model.get_mapper_repr())
    installed = 0
    for perceptron in targets:
        if install_lif_input_quantizer(
            perceptron, int(pipeline_config["simulation_steps"])
        ):
            installed += 1
    # [U-SC S-1] the first stage that walks the deployed composition (the D-hat
    # gauge reads it from the first rung): establish the gauge on EVERY
    # wire-currency graph, whatever the placement, so the exact-QAT trains and
    # measures the SAME armed composition every later seam (WQ / SCM) re-derives.
    # The scales the quantizers read are written in place, so they follow.
    if getattr(model, "get_mapper_repr", lambda: None)() is not None:
        # [U-SC2 A-2] with the negative-value boundary policy, so the D-hat reads the boundary SCM deploys; decided
        # after the entry carriers above exist (review M1): an entry carrying the trained quantizer is the QAT's own
        # clamp, outside the policy's scope as at Soft Core Mapping, and its lift under offload is the opt-in's.
        establish_deployed_wire_gauge(
            model,
            input_data_scale=ResolvedWorkloadProfile.from_config(
                pipeline_config
            ).input_data_scale,
            negative_boundary=deployed_negative_boundary_seam(model, trainer, pipeline_config),
        )
    return installed
