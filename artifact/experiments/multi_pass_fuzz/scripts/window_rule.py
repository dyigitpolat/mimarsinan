""
from __future__ import annotations

import contextlib
from typing import Any, Mapping, Optional


DOCUMENT_KEYS = ("spiking_family", "spiking_variant", "lif_window_firing", "lif_per_hop_retiming", "core_semantics")


def _is_document(run: Mapping[str, Any]) -> bool:
    return any(key in run for key in DOCUMENT_KEYS)


def point_of(run: Mapping[str, Any]):
    ""
    if _is_document(run):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = DeploymentPlan.resolve(dict(run))
        return None if plan.is_mvm else plan.require_spiking_point()
    from mimarsinan.chip_simulation.spiking_point import HISTORICAL_WINDOW_FIRING, stated_point
    from mimarsinan.chip_simulation.window_firing import INTEGRATE_THEN_EMIT

    record = {"spiking_mode": run["spiking_mode"], "ttfs_cycle_schedule": run.get("ttfs_cycle_schedule"),
              "spike_phase_dither": bool(run.get("spike_phase_dither", False)),
              "lif_membrane_init": float(run.get("lif_membrane_init", 0.0))}
    firing = INTEGRATE_THEN_EMIT if run.get("lif_execution_synchronized", False) else HISTORICAL_WINDOW_FIRING
    return stated_point(record, window_firing=firing)


def cascade_policy(policy) -> Optional[bool]:
    ""
    from mimarsinan.chip_simulation.spiking_mode_policy import SpikingModePolicy

    return bool(policy.runs_per_cycle_cascade) if isinstance(policy, SpikingModePolicy) else None


def cascade_of(run: Mapping[str, Any]) -> bool:
    ""
    if _is_document(run):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        return bool(DeploymentPlan.resolve(dict(run)).mode_policy().runs_per_cycle_cascade)
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

    return bool(policy_for_spiking_mode(point_of(run)).runs_per_cycle_cascade)


def reads_window(run: Mapping[str, Any], *, per_hop: bool) -> bool:
    ""
    if per_hop or not cascade_of(run):
        return False
    point = point_of(run)
    return point is not None and not point.integrates_then_emits


def armed(run: Mapping[str, Any]) -> bool:
    ""
    from mimarsinan.mapping.latency.cycle_windows import RELAY_KEY

    point = point_of(run)
    if point is None:
        return False
    if _is_document(run):
        from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters

        config = dict(run)
        derive_deployment_parameters(config)
        return bool(config.get(RELAY_KEY, False))
    from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

    return bool(ConversionPolicy.derive(point).knobs.get(RELAY_KEY, False))


def relays_for(run: Mapping[str, Any]) -> bool:
    ""
    return cascade_of(run) and armed(run)


def through_window(ir, run: Mapping[str, Any], *, thresholding_mode: str = "<=", q_max: int = 127) -> int:
    ""
    from mimarsinan.mapping.latency.depth_balancing import balance_cycle_windows

    cascade = cascade_of(run)
    return balance_cycle_windows(ir, cascade=cascade, armed=cascade and armed(run), thresholding_mode=thresholding_mode,
                                 q_max=int(q_max))


@contextlib.contextmanager
def building(owner, name: str, run: Mapping[str, Any], *, thresholding_mode: str = "<=", q_max: int = 127):
    ""
    had = name in vars(owner)
    original = getattr(owner, name)

    def built(*args, **kwargs):
        ir = original(*args, **kwargs)
        through_window(ir, run, thresholding_mode=thresholding_mode, q_max=q_max)
        return ir

    setattr(owner, name, built)
    try:
        yield built
    finally:
        if had:
            setattr(owner, name, original)
        else:
            delattr(owner, name)
