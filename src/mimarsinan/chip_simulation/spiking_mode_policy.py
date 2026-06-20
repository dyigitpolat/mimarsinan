"""Behavior-carrying policy per (firing × sync) spiking mode (Vector V2).

One polymorphic ``SpikingModePolicy`` per (firing × sync) collapses the
``cascaded``/``synchronized`` predicate-then-branch that callers re-derive:

- ``LifModePolicy`` — LIF rate family (``lif`` / ``rate``).
- ``TtfsAnalyticalModePolicy`` — closed-form analytical TTFS (``ttfs`` /
  ``ttfs_quantized``).
- ``TtfsSyncCycleModePolicy`` — genuine single-spike, synchronized schedule.
- ``TtfsCascadeModePolicy`` — genuine fire-once-latch, cascaded schedule.

Each carries the behavior callers previously re-derived: ``training_forward_kind``
(the NF algorithm fine-tuners train through), ``calibration_forward`` (the
negative-shift NF forward), ``soma_hw_name`` / ``soma_model_attributes`` (the
SANA-FE soma chain), ``log_potential``, ``decode_mode``, and ``valid_backends``.

Selection (``policy_for_spiking_mode`` / ``SpikingModePolicy.from_contract``)
is the SSOT for ``(spiking_mode, schedule) → policy``; mirrors
``spiking/segment_policies.py``.
"""

from __future__ import annotations

from typing import Any, Optional

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_analytical_ttfs,
    is_cascaded_ttfs,
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    requires_ttfs_firing,
    supports_spiking_mode,
)

__all__ = [
    "SpikingModePolicy",
    "LifModePolicy",
    "TtfsAnalyticalModePolicy",
    "TtfsSyncCycleModePolicy",
    "TtfsCascadeModePolicy",
    "policy_for_spiking_mode",
]


class SpikingModePolicy:
    """Behavior carried per (firing × sync) deployment mode.

    Concrete subclasses override the firing-/schedule-derived behavior. The
    base resolves only what is shared (capability gate). ``spiking_mode``
    captures the sub-variation within a (firing × sync) family (``ttfs`` vs
    ``ttfs_quantized``, ``lif`` vs ``rate``); ``schedule`` is the normalized
    ``ttfs_cycle_schedule`` for the TTFS-cycle family (``None`` otherwise).
    """

    def __init__(self, spiking_mode: str, schedule: Optional[str] = None) -> None:
        self.spiking_mode = spiking_mode
        self.schedule = schedule

    # ── selection (SSOT) ────────────────────────────────────────────────────
    @classmethod
    def from_contract(cls, contract: Any) -> "SpikingModePolicy":
        return policy_for_spiking_mode(
            contract.spiking_mode, contract.ttfs_cycle_schedule
        )

    # ── training ────────────────────────────────────────────────────────────
    def training_forward_kind(self) -> str:
        """NF algorithm the fine-tuners must train through for this deployment."""
        raise NotImplementedError

    # ── calibration (negative-value shift NF forward) ───────────────────────
    def calibration_forward(self):
        """NF forward that produces this mode's boundary values for calibration."""
        raise NotImplementedError

    # ── decode ──────────────────────────────────────────────────────────────
    def decode_mode(self) -> str:
        """How a segment's output is decoded: ``count`` (rate) vs ``timing`` (TTFS)."""
        raise NotImplementedError

    # ── SANA-FE soma model ──────────────────────────────────────────────────
    def soma_hw_name(self) -> str:
        """Name of the SANA-FE soma plugin for this mode."""
        raise NotImplementedError

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        """The ``model_attributes`` dict for one SANA-FE neuron of this mode."""
        raise NotImplementedError

    @property
    def log_potential(self) -> bool:
        """Whether SANA-FE must log the membrane potential trace (TTFS V-decode)."""
        return False

    @property
    def requires_ttfs_firing(self) -> bool:
        return requires_ttfs_firing(self.spiking_mode)

    # ── backend capability ──────────────────────────────────────────────────
    def valid_backends(self, candidates) -> tuple[str, ...]:
        """Subset of ``candidates`` whose capabilities support this mode."""
        return tuple(
            b for b in candidates if supports_spiking_mode(b, self.spiking_mode)
        )


class LifModePolicy(SpikingModePolicy):
    """LIF rate family: per-cycle integrate-and-fire, count decode."""

    def training_forward_kind(self) -> str:
        return "rate" if self.spiking_mode == "rate" else "lif_cycle"

    def calibration_forward(self):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        return chip_aligned_segment_forward

    def decode_mode(self) -> str:
        return "count"

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_LIF_NAME

        return SOMA_LIF_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import lif_model_attributes

        return lif_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
            firing_mode=firing_mode,
        )


class TtfsAnalyticalModePolicy(SpikingModePolicy):
    """Closed-form analytical TTFS (``ttfs`` continuous / ``ttfs_quantized``)."""

    def training_forward_kind(self) -> str:
        return "analytical_staircase"

    def calibration_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _analytical_segment_calibration_forward,
        )

        return _analytical_segment_calibration_forward

    def decode_mode(self) -> str:
        return "timing"

    @property
    def _is_quantized(self) -> bool:
        return forces_activation_quantization(self.spiking_mode)

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import (
            SOMA_TTFS_CONTINUOUS_NAME,
            SOMA_TTFS_QUANTIZED_NAME,
        )

        return (
            SOMA_TTFS_QUANTIZED_NAME if self._is_quantized else SOMA_TTFS_CONTINUOUS_NAME
        )

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_continuous_model_attributes,
            ttfs_quantized_model_attributes,
        )

        builder = (
            ttfs_quantized_model_attributes
            if self._is_quantized
            else ttfs_continuous_model_attributes
        )
        return builder(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        return True


class _TtfsCycleModePolicy(SpikingModePolicy):
    """Shared base for the genuine single-spike (ttfs_cycle_based) schedules."""

    def calibration_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _ttfs_segment_calibration_forward,
        )

        return _ttfs_segment_calibration_forward


class TtfsSyncCycleModePolicy(_TtfsCycleModePolicy):
    """Genuine single-spike, synchronized schedule (V-reconstruction soma)."""

    def training_forward_kind(self) -> str:
        return "analytical_staircase"

    def decode_mode(self) -> str:
        return "timing"

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_TTFS_CYCLE_NAME

        return SOMA_TTFS_CYCLE_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cycle_model_attributes,
        )

        return ttfs_cycle_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        return True


class TtfsCascadeModePolicy(_TtfsCycleModePolicy):
    """Genuine fire-once-latch, cascaded schedule (count-based decode, LIF-like)."""

    def training_forward_kind(self) -> str:
        return "segment_spike"

    def decode_mode(self) -> str:
        return "count"

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_TTFS_CASCADE_NAME

        return SOMA_TTFS_CASCADE_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cascade_model_attributes,
        )

        return ttfs_cascade_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        # Cascade decodes from spike counts, not the membrane potential trace.
        return False


def policy_for_spiking_mode(
    spiking_mode: str, schedule=None
) -> SpikingModePolicy:
    """Resolve the ``(firing × sync)`` policy for ``(spiking_mode, schedule)``.

    The single source of truth for the dispatch; downstream callers take the
    resolved policy instead of re-branching on ``cascaded``/``synchronized``.
    """
    mode = str(spiking_mode or "lif")
    if is_ttfs_cycle_based(mode):
        if is_synchronized_ttfs(mode, schedule):
            from mimarsinan.chip_simulation.spiking_semantics import ttfs_cycle_schedule

            return TtfsSyncCycleModePolicy(mode, ttfs_cycle_schedule(schedule))
        from mimarsinan.chip_simulation.spiking_semantics import ttfs_cycle_schedule

        return TtfsCascadeModePolicy(mode, ttfs_cycle_schedule(schedule))
    if is_analytical_ttfs(mode):
        return TtfsAnalyticalModePolicy(mode)
    return LifModePolicy(mode)
