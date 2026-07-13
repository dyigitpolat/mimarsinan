"""[lif_exact_qat_program.md §6] LIF exact-QAT arm: predicate, markers, installers."""

from __future__ import annotations

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


def install_lif_entry_input_quantizers(model, pipeline_config) -> int:
    """Install the deployed entry round on encoding perceptrons at the AQ install
    seam so the exact-QAT ladder trains through it (§4.3 — the entry seam is
    already exact; this moves its install before the QAT). Idempotent; returns
    installs (0 when the arm is off)."""
    if not lif_exact_qat_active(pipeline_config):
        return 0
    installed = 0
    for perceptron in model.get_perceptrons():
        if not getattr(perceptron, "is_encoding_layer", False):
            continue
        if install_lif_input_quantizer(
            perceptron, int(pipeline_config["simulation_steps"])
        ):
            installed += 1
    return installed
