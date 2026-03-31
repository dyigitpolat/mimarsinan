"""Data-driven initial tolerance for SmartSmoothAdaptation step search.

See tuning/ARCHITECTURE.md and root ARCHITECTURE.md (SmartSmoothAdaptation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

# Learning rate for the one-epoch probe: a constant or a zero-arg callable
# (e.g. ``tuner._find_lr``) evaluated once when calibration runs, after
# ``before_cycle`` and baseline validation, so it matches adaptation timing.
LrProbeSpec = Union[float, Callable[[], float]]


@dataclass(frozen=True)
class ToleranceCalibrationConfig:
    """Controls probe ladder and how instant-drop tolerance is derived."""

    delta_t_schedule: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    residual_threshold: float = 1e-3
    tolerance_min: float = 0.01
    tolerance_max: float = 0.15
    baseline_epsilon: float = 1e-9


def estimate_tolerable_instant_drop(
    baseline: float,
    config: ToleranceCalibrationConfig,
    probe_at_delta: Callable[[float], Tuple[float, float]],
) -> float:
    """
    Choose an allowed *instant* metric drop fraction for step search.

    For each ``delta_t`` in ``config.delta_t_schedule`` (largest first), runs
    ``probe_at_delta(delta_t)`` which must return ``(instant_acc, acc_after_one_epoch)``.
    The first probe whose residual drop vs baseline is within
    ``residual_threshold`` wins; the returned tolerance is the normalized
    instant drop at that probe, clamped to ``[tolerance_min, tolerance_max]``.

    If no probe passes, returns ``tolerance_min``.
    """
    denom = max(float(baseline), config.baseline_epsilon)

    for delta_t in config.delta_t_schedule:
        if delta_t <= 0.0:
            continue
        d = min(delta_t, 1.0)
        instant, recovered = probe_at_delta(d)
        residual = (float(baseline) - float(recovered)) / denom
        if residual <= config.residual_threshold:
            raw = max(0.0, (float(baseline) - float(instant)) / denom)
            return max(
                config.tolerance_min,
                min(raw, config.tolerance_max),
            )

    return config.tolerance_min


def tolerance_config_from_pipeline_config(
    pipeline_config: dict,
) -> ToleranceCalibrationConfig:
    """Build :class:`ToleranceCalibrationConfig` from deployment-style config keys."""
    schedule = pipeline_config.get("tuner_smooth_tolerance_delta_schedule")
    if schedule is None:
        delta_t_schedule: Tuple[float, ...] = ToleranceCalibrationConfig().delta_t_schedule
    else:
        delta_t_schedule = tuple(float(x) for x in schedule)

    return ToleranceCalibrationConfig(
        delta_t_schedule=delta_t_schedule,
        residual_threshold=float(
            pipeline_config.get("tuner_smooth_tolerance_residual_threshold", 1e-3)
        ),
        tolerance_min=float(pipeline_config.get("tuner_smooth_tolerance_min", 0.01)),
        tolerance_max=float(pipeline_config.get("tuner_smooth_tolerance_max", 0.15)),
        baseline_epsilon=float(
            pipeline_config.get("tuner_smooth_tolerance_baseline_epsilon", 1e-9)
        ),
    )


def effective_probe_lr(pipeline_config: dict, lr_probe: LrProbeSpec) -> float:
    """
    Resolve LR used for each one-epoch calibration probe.

    If ``tuner_smooth_tolerance_lr`` is set in ``pipeline_config`` (not ``None``),
    it wins. Otherwise ``lr_probe`` is used: call it if callable, else treat as
    float. The result is multiplied by ``tuner_smooth_tolerance_lr_scale``
    (default 1.0).
    """
    override = pipeline_config.get("tuner_smooth_tolerance_lr")
    if override is not None:
        lr = float(override)
    else:
        lr = float(lr_probe()) if callable(lr_probe) else float(lr_probe)

    scale = float(pipeline_config.get("tuner_smooth_tolerance_lr_scale", 1.0))
    return lr * scale


def make_smooth_tolerance_calibration_fn(
    pipeline_config: dict,
    *,
    clone_state: Callable[[], object],
    restore_state: Callable[[object], None],
    evaluate_at_rate: Callable[[float], float],
    validate_fn: Callable[[], float],
    train_validation_epochs: Callable[[float, int, int], float],
    lr_probe: LrProbeSpec,
    before_cycle: Optional[Callable[[], None]] = None,
) -> Callable[[], float]:
    """
    Factory for ``SmartSmoothAdaptation``'s ``initial_tolerance_fn``.

    Runs one calibration pass: optional ``before_cycle``, baseline validation,
    then resolves probe LR via :func:`effective_probe_lr`, then for each
    ``delta_t`` clones state, evaluates at ``rate=delta_t``, trains exactly
    one epoch at that LR, validates, restores. Returns
    :func:`estimate_tolerable_instant_drop` over those probes.
    """

    def _initial_tolerance_fn() -> float:
        if before_cycle is not None:
            before_cycle()
        baseline = float(validate_fn())
        cfg = tolerance_config_from_pipeline_config(pipeline_config)
        lr_epoch = effective_probe_lr(pipeline_config, lr_probe)

        def probe_at_delta(delta_t: float) -> Tuple[float, float]:
            state = clone_state()
            try:
                instant = float(evaluate_at_rate(delta_t))
                train_validation_epochs(lr_epoch, 1, 0)
                recovered = float(validate_fn())
                return instant, recovered
            finally:
                restore_state(state)

        return estimate_tolerable_instant_drop(baseline, cfg, probe_at_delta)

    return _initial_tolerance_fn


def initial_tolerance_fn_for_pipeline_if_enabled(
    pipeline_config: dict,
    *,
    clone_state: Callable[[], object],
    restore_state: Callable[[object], None],
    evaluate_at_rate: Callable[[float], float],
    validate_fn: Callable[[], float],
    train_validation_epochs: Callable[[float, int, int], float],
    lr_probe: LrProbeSpec,
    before_cycle: Optional[Callable[[], None]] = None,
) -> Optional[Callable[[], float]]:
    """
    Returns ``None`` when ``tuner_calibrate_smooth_tolerance`` is false
    (callers keep default / tuner-specific static tolerance).
    """
    if not bool(pipeline_config.get("tuner_calibrate_smooth_tolerance", False)):
        return None
    return make_smooth_tolerance_calibration_fn(
        pipeline_config,
        clone_state=clone_state,
        restore_state=restore_state,
        evaluate_at_rate=evaluate_at_rate,
        validate_fn=validate_fn,
        train_validation_epochs=train_validation_epochs,
        lr_probe=lr_probe,
        before_cycle=before_cycle,
    )
