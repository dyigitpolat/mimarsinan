"""Optimization driver: controller vs fast-ladder, orthogonal to the ramp strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationDriver:
    """The resolved optimization driver for a rate-tuner run."""

    fast_ladder: bool
    fast_ladder_rates: list
    fast_ladder_steps_per_rate: int
    fast_ladder_eta_min_factor: float

    @property
    def controller(self) -> bool:
        """Whether the full SmoothAdaptation controller drives the ramp."""
        return not self.fast_ladder

    @classmethod
    def for_family(
        cls,
        *,
        fast: bool,
        rates: list,
        steps_per_rate: int,
        eta_min_factor: float = 0.0,
    ) -> "OptimizationDriver":
        """The generic ``controller | fast`` axis for the single-switch families.

        One switch (``fast``) selects the driver; the ladder is the family's uniform
        rate list, carried even when ``fast=False`` so the tuner can configure
        ``_setup_fast_ladder`` idempotently."""
        return cls(
            fast_ladder=bool(fast),
            fast_ladder_rates=[float(r) for r in (rates or [])] or [1.0],
            fast_ladder_steps_per_rate=int(steps_per_rate),
            fast_ladder_eta_min_factor=max(0.0, float(eta_min_factor)),
        )

    @classmethod
    def resolve(
        cls,
        *,
        genuine_blend_fast: bool,
        proxy_fast: bool,
        blend_fast_rates: list,
        blend_fast_steps_per_rate: int,
        blend_fast_lr_eta_min: float,
    ) -> "OptimizationDriver":
        fast = genuine_blend_fast or proxy_fast
        eta = blend_fast_lr_eta_min if proxy_fast else 0.0
        return cls(
            fast_ladder=fast,
            fast_ladder_rates=blend_fast_rates,
            fast_ladder_steps_per_rate=blend_fast_steps_per_rate,
            fast_ladder_eta_min_factor=eta,
        )
