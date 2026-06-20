"""Optimization driver: controller vs fast-ladder, orthogonal to the ramp strategy.

*How* the rate is driven 0→1 (which optimization machinery) is a concern orthogonal
to *what* the model ramps toward (the ramp strategy) and to the conversion-health
calibration. It used to be a cluster of booleans (``ttfs_blend_fast`` /
``ttfs_genuine_blend_fast`` / ``ttfs_staircase_ste_fast``) whose fast-ladder rung was
hand-derived inline.

Two drivers:

* **controller** — the full SmoothAdaptation machinery (adaptive rate scheduler +
  bisect, recover-to-target, rollback clone/restore, stabilization, per-cycle LR find,
  catastrophic gate, target adjuster). Robust/high-quality, the default.
* **fast-ladder** — the schedule-not-search ``fixed_ladder`` policy: walk an explicit
  rate ladder with ONE shared optimizer + spanning warmup/cosine LR, no per-cycle
  rollback/recovery/LR-find/stabilization. The STE-fast variant owns a single rung at
  rate 1.0 (its dedicated split-LR + progressive-depth loop runs the steps); the
  blend/proxy variants walk a multi-rung ladder, with the proxy flooring the endpoint
  LR (its value-domain endpoint needs real recovery; the genuine-blend's genuine-CE
  carries it).

``OptimizationDriver.resolve`` is the SINGLE place that maps the (already
precedence-resolved) fast-path selectors to the concrete ``_setup_fast_ladder``
arguments the tuner consumes, so the rung derivation is read in one tested place.

E2 (Fix A) makes the driver a pipeline-wide axis, not a KD-blend island: every
rate tuner inherits the fast ladder (``FastLadderMixin``) and reads ITS driver
through the same ``OptimizationDriver``. ``for_family`` is the generic resolver
for the families with one fast switch + a uniform ladder (LIF, and the analytical
clamp/shift/activation-quant/manager-rate chain): a single declarative axis,
``fast=False`` ⇒ controller ⇒ byte-identical. The TTFS family keeps the richer
``resolve`` because its fast path forks three ways (STE-fast / blend / proxy).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationDriver:
    """The resolved optimization driver for a rate-tuner run."""

    # ── which driver (mutually exclusive; controller is the default) ──
    fast_ladder: bool
    # ── resolved fast-ladder rung (what _setup_fast_ladder consumes) ──
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

        One declarative switch (``fast``) selects the driver; the ladder is the
        family's uniform value-domain rate list. ``fast=False`` ⇒ controller (the
        ladder is still carried so the tuner can configure ``_setup_fast_ladder``
        idempotently — ``enabled=False`` disables the ``fixed_ladder`` policy)."""
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
        staircase_ste_fast: bool,
        genuine_blend_fast: bool,
        proxy_fast: bool,
        ste_steps: int,
        blend_fast_rates: list,
        blend_fast_steps_per_rate: int,
        blend_fast_lr_eta_min: float,
    ) -> "OptimizationDriver":
        if staircase_ste_fast:
            # One rung at full rate; the dedicated STE loop owns its split-LR optimizer
            # + progressive depth + cosine over ste_steps.
            return cls(
                fast_ladder=True,
                fast_ladder_rates=[1.0],
                fast_ladder_steps_per_rate=ste_steps,
                fast_ladder_eta_min_factor=0.0,
            )
        fast = genuine_blend_fast or proxy_fast
        # eta floors the endpoint LR for the proxy (value-domain endpoint needs real
        # recovery); the genuine blend lets its genuine-CE carry it.
        eta = blend_fast_lr_eta_min if proxy_fast else 0.0
        return cls(
            fast_ladder=fast,
            fast_ladder_rates=blend_fast_rates,
            fast_ladder_steps_per_rate=blend_fast_steps_per_rate,
            fast_ladder_eta_min_factor=eta,
        )
