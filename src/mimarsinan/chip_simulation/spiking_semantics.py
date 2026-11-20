"""Spiking-mode taxonomy and per-backend capability matrix, queried by intent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import FrozenSet

from mimarsinan.chip_simulation.window_firing_axis import FIRE_WHILE_INTEGRATING, INTEGRATE_THEN_EMIT

ANALYTICAL_TTFS_MODES: FrozenSet[str] = frozenset({"ttfs", "ttfs_quantized"})
TTFS_MODES: FrozenSet[str] = ANALYTICAL_TTFS_MODES
TTFS_FAMILY_MODES: FrozenSet[str] = frozenset(
    {"ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)
CYCLE_BASED_MODES: FrozenSet[str] = frozenset({"lif", "ttfs_cycle_based"})
LIF_MODES: FrozenSet[str] = frozenset({"lif"})
ALL_SPIKING_MODES: FrozenSet[str] = frozenset(
    {"lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)

_REMOVED_SPIKING_MODES: dict = {"rate": "lif"}

_ACT_QUANT_MODES: FrozenSet[str] = frozenset({"ttfs_quantized", "ttfs_cycle_based"})


def _norm(spiking_mode: str) -> str:
    return str(spiking_mode or "lif")


def require_known_spiking_mode(spiking_mode: str) -> str:
    """Return the normalized mode, or raise ``ValueError`` if it is not deployable."""
    mode = _norm(spiking_mode)
    if mode in ALL_SPIKING_MODES:
        return mode
    if mode in _REMOVED_SPIKING_MODES:
        raise ValueError(
            f"spiking_mode {mode!r} was removed from the deploy taxonomy; "
            f"use {_REMOVED_SPIKING_MODES[mode]!r} instead."
        )
    raise ValueError(
        f"unknown spiking_mode {mode!r}; valid modes: {sorted(ALL_SPIKING_MODES)}."
    )


def requires_ttfs_firing(spiking_mode: str) -> bool:
    """Mode needs ``firing_mode='TTFS'`` / ``spike_generation_mode='TTFS'``."""
    return _norm(spiking_mode) in TTFS_FAMILY_MODES


def is_analytical_ttfs(spiking_mode: str) -> bool:
    """Mode uses the closed-form real-valued TTFS forward (not cycle-based)."""
    return _norm(spiking_mode) in ANALYTICAL_TTFS_MODES


def is_ttfs_cycle_based(spiking_mode: str) -> bool:
    """Mode is the genuine binary-spike, cycle-based TTFS mode."""
    return _norm(spiking_mode) == "ttfs_cycle_based"


def is_cycle_based(spiking_mode: str) -> bool:
    """Mode performs genuine per-cycle binary integrate-and-fire (LIF or TTFS-cycle)."""
    return _norm(spiking_mode) in CYCLE_BASED_MODES


def is_lif(spiking_mode: str) -> bool:
    """Mode is the plain rate-coded LIF deployment family."""
    return _norm(spiking_mode) in LIF_MODES


def lif_membrane_init(config) -> float:
    """[calculus sec.15.11] window-start membrane guard (normalized units;
    negative recenters the signed-charge rectifier against transient overfire)."""
    return float(config.get("lif_membrane_init", 0.0))


def spike_phase_dither_enabled(config) -> bool:
    """[calculus sec.15.11] count-exact per-channel comb rotation at uniform
    encodes, decorrelating cross-channel arrival (decode-invariant)."""
    return bool(config.get("spike_phase_dither", False))


def lif_per_hop_retiming_enabled(config) -> bool:
    """[C3/R5] the per-hop count-exact re-timing knob is armed for a LIF plan.

    BOTH twins consult this: the mapping splits neural segments per hop and the
    chip-aligned NF re-times its emitted trains (``LifSegmentPolicy(retime)``).
    """
    return bool(config.get("lif_per_hop_retiming", False)) and is_lif(
        str(config.get("spiking_mode", "lif"))
    )


def lif_config(config) -> bool:
    """[W15] The config runs a LIF plan (the mode the dither and the membrane guard act on)."""
    return is_lif(str(config.get("spiking_mode", "lif")))


def cycle_accurate_lif_trains(config) -> bool:
    """[W15] Host ops leave cycle-accurate trains a stage reads verbatim: a LIF plan with the cycle-accurate forward
    (``SpikingHybridCoreFlow._use_cycle_accurate_trains``; off unless declared, as the plan resolves it)."""
    return lif_config(config) and bool(config.get("cycle_accurate_lif_forward", False))


def ttfs_half_step_folded(config, *, floor_ceil_convention: bool) -> bool:
    """[W15] The half-step bias compensation is folded into the biases (``SoftCoreMappingStep``: a plan of the
    floor-ceil convention, which the caller decides from its resolved semantics, with activation quantization), so
    Proposition 2's P3 and P4 bind."""
    return bool(floor_ceil_convention) and bool(config.get("activation_quantization", False))


def forces_activation_quantization(spiking_mode: str) -> bool:
    """Mode requires the activation-quantization chain (S-level activations)."""
    return _norm(spiking_mode) in _ACT_QUANT_MODES


def is_bit_parity_lossless_conversion(spiking_mode: str) -> bool:
    """Mode whose trained forward is per-neuron bit-equivalent to its deployed executor.

    Exactly the analytic BIT_PARITY family (5u): analytical ``ttfs``. The
    near-lossless modes (lif, ttfs_quantized, synchronized) keep a real endpoint
    gap and join only after that gap is verified sub-SE; cascaded is lossy.
    """
    return is_analytical_ttfs(spiking_mode) and not forces_activation_quantization(
        spiking_mode
    )


DEFAULT_TTFS_CYCLE_SCHEDULE = "cascaded"
TTFS_CYCLE_SCHEDULES: FrozenSet[str] = frozenset({"cascaded", "synchronized"})


def ttfs_cycle_schedule(schedule) -> str:
    """Normalize the ttfs_cycle schedule value, defaulting to ``cascaded``."""
    s = str(schedule or DEFAULT_TTFS_CYCLE_SCHEDULE)
    return s if s in TTFS_CYCLE_SCHEDULES else DEFAULT_TTFS_CYCLE_SCHEDULE


def is_explicit_ttfs_cycle_schedule(schedule) -> bool:
    """``schedule`` explicitly names a known ttfs_cycle schedule (no default applied)."""
    return schedule in TTFS_CYCLE_SCHEDULES


def is_cascaded_ttfs(spiking_mode: str, schedule) -> bool:
    """ttfs_cycle_based running the greedy pipelined (cascaded) schedule."""
    return is_ttfs_cycle_based(spiking_mode) and ttfs_cycle_schedule(schedule) == "cascaded"


def is_synchronized_ttfs(spiking_mode: str, schedule) -> bool:
    """ttfs_cycle_based running the exact sequential (synchronized) schedule."""
    return is_ttfs_cycle_based(spiking_mode) and ttfs_cycle_schedule(schedule) == "synchronized"


def uses_ttfs_floor_ceil_convention(spiking_mode: str, schedule=None) -> bool:
    """Modes that train the floor-staircase + half-step-bias convention yet deploy the ceil TTFS kernel.

    ``ttfs_quantized`` and the synchronized floor-collapse; both take the half-step bias
    compensation and are excluded from the bit-exact per-neuron parity gate.
    """
    return _norm(spiking_mode) == "ttfs_quantized" or is_synchronized_ttfs(
        spiking_mode, schedule
    )


DEFAULT_FIRING_MODE = "Default"
NOVENA_FIRING_MODE = "Novena"

# THE legal-value sets of the mode-dependent spiking knobs. `legal_*` answers
# "what may this key hold given spiking_mode"; `derived_*` answers "what does it
# hold when the document is silent". The enforcers (FiringStrategy, the rate
# encoder) are pinned against these in tests/unit/config_schema/test_legality.py.
LIF_FIRING_MODES: FrozenSet[str] = frozenset({DEFAULT_FIRING_MODE, NOVENA_FIRING_MODE})
TTFS_FIRING_MODE = "TTFS"
RATE_SPIKE_GENERATION_MODES = ("Uniform", "Deterministic", "Stochastic")
THRESHOLDING_MODES = ("<", "<=")
DEFAULT_THRESHOLDING_MODE = "<="


def _ttfs_or(rate_modes: tuple, spiking_mode: str) -> tuple:
    """TTFS-family modes admit only the TTFS value; an unknown mode rules
    nothing out (its own error is the single truth)."""
    if _norm(spiking_mode) not in ALL_SPIKING_MODES:
        return rate_modes + (TTFS_FIRING_MODE,)
    if requires_ttfs_firing(spiking_mode):
        return (TTFS_FIRING_MODE,)
    return rate_modes


def legal_firing_modes(spiking_mode: str) -> tuple:
    """Firing modes this spiking mode may declare."""
    return _ttfs_or((DEFAULT_FIRING_MODE, NOVENA_FIRING_MODE), spiking_mode)


def legal_spike_generation_modes(spiking_mode: str) -> tuple:
    """Input spike-train encodings this spiking mode may declare (the rate
    encoder refuses TTFS; the TTFS path refuses every rate encoding)."""
    return _ttfs_or(RATE_SPIKE_GENERATION_MODES, spiking_mode)


def legal_thresholding_modes(spiking_mode: str) -> tuple:
    """Membrane-threshold comparisons (mode-independent, both always legal)."""
    return THRESHOLDING_MODES


def derived_firing_mode(spiking_mode: str) -> str:
    return TTFS_FIRING_MODE if requires_ttfs_firing(spiking_mode) else DEFAULT_FIRING_MODE


def derived_spike_generation_mode(spiking_mode: str) -> str:
    if requires_ttfs_firing(spiking_mode):
        return TTFS_FIRING_MODE
    return RATE_SPIKE_GENERATION_MODES[0]


def is_default_firing_mode(firing_mode) -> bool:
    """Firing mode with the Default (subtractive-reset) LIF semantics."""
    return firing_mode == DEFAULT_FIRING_MODE


def is_novena_firing_mode(firing_mode) -> bool:
    """Firing mode with the Novena (zero-reset) LIF semantics."""
    return firing_mode == NOVENA_FIRING_MODE


@dataclass(frozen=True)
class BackendSpikingCapabilities:
    """What one backend can execute: the four mode columns, the soma-law axes, then [U-LIF2 W3] the window firings of a
    synchronized LIF hop it executes. The columns default to nothing, so every backend refuses what no row declares
    (an undeclared name declares no window firing and refuses both values by name)."""

    lif: bool
    ttfs: bool
    ttfs_quantized: bool
    ttfs_cycle_based: bool
    per_event_firing: bool = False
    saturating_membrane: bool = False
    signed_membrane: bool = False
    window_firings: FrozenSet[str] = frozenset()


#: [U-LIF2 W3] Every row fires while integrating; [W9, W10] the hybrid flows and the training walk also emit gated.
_PER_CYCLE, _GATED = frozenset({FIRE_WHILE_INTEGRATING}), frozenset({FIRE_WHILE_INTEGRATING, INTEGRATE_THEN_EMIT})
#: Every mode and every soma-law column; the ODIN fabrics' row: LIF only (no TTFS executor on the crossbar).
_EVERY_COLUMN = BackendSpikingCapabilities(True, True, True, True, True, True, True, window_firings=_PER_CYCLE)
_ODIN_ROW = BackendSpikingCapabilities(True, False, False, False, True, True, True, window_firings=_PER_CYCLE)
_LIF_ONLY = BackendSpikingCapabilities(True, False, False, False, window_firings=_PER_CYCLE)

_BACKEND_CAPS: dict[str, BackendSpikingCapabilities] = {
    # [ODIN P2/P3] the torch executors (models/spiking/serial) and nevresim's integration policies implement the
    # event-serial fold on the saturating unsigned membrane (nevresim where its tree carries them, which its driver
    # probes: nevresim/integration_policy_read); every other backend refuses the point by name.
    "hcm": replace(_EVERY_COLUMN, window_firings=_GATED),
    "nevresim": _EVERY_COLUMN,
    "unified": replace(_EVERY_COLUMN, window_firings=_GATED),
    "hybrid": replace(_EVERY_COLUMN, window_firings=_GATED),
    # [ODIN P6, P7a, P8] the RTL cosimulation instrument (not a BACKEND_REGISTRY backend: the stock per-event core and
    # every generated variant, the sync-fire core's two's-complement membrane included), the physical backend, and
    # the HACC export, which freezes a bundle FOR that device and so can never admit a law the device refuses.
    "odin_rtl": _ODIN_ROW,
    "odin_fpga": _ODIN_ROW,
    "odin_hacc": _ODIN_ROW,
    "sanafe": BackendSpikingCapabilities(True, True, True, True, window_firings=_PER_CYCLE),
    "lava": _LIF_ONLY,
    "loihi": _LIF_ONLY,
    # The NF twin executes the SAME fold as a parity instrument under eval, but no gradient is defined for a count
    # fold, so training refuses the point rather than claim an executor it lacks. [U-LIF2 W10] The walk's count branch
    # (``lif_window_count`` of each hop's window charge) is its integrate-then-emit executor.
    "training": replace(_LIF_ONLY, window_firings=_GATED),
    # The soft-core mapping's torch twin of the deployed program (the parity contracts' reference beside the HCM).
    "scm": replace(_LIF_ONLY, window_firings=_GATED),
}


def backend_capabilities(backend: str) -> BackendSpikingCapabilities:
    key = str(backend or "").lower()
    return _BACKEND_CAPS.get(key, replace(_LIF_ONLY, window_firings=frozenset()))


def supports_spiking_mode(backend: str, spiking_mode: str) -> bool:
    spiking = _norm(spiking_mode)
    caps = backend_capabilities(backend)
    if spiking == "ttfs":
        return caps.ttfs
    if spiking == "ttfs_quantized":
        return caps.ttfs_quantized
    if spiking == "ttfs_cycle_based":
        return caps.ttfs_cycle_based
    return caps.lif


def require_spiking_mode_supported(
    spiking_mode: str,
    *,
    backend: str,
    context: str,
) -> None:
    if not supports_spiking_mode(backend, spiking_mode):
        raise ValueError(
            f"{context}: backend {backend!r} does not support "
            f"spiking_mode={spiking_mode!r}"
        )
