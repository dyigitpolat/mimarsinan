"""Workload-profile contracts: the facts a dataset or model architecture registers with the framework."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, MutableMapping, Optional, Tuple


@dataclass(frozen=True)
class CalibrationSetPolicy:
    """Calibration-set extents by purpose; ``None`` = the consumer's frozen generic default."""

    distmatch_bias_iters: Optional[int] = None
    distmatch_cal_batches: Optional[int] = None
    gauge_batches: Optional[int] = None
    stat_batches: Optional[int] = None
    analysis_batches_max: Optional[int] = None
    analysis_batch_size_cap: Optional[int] = None

    def declared_fields(self) -> dict[str, int]:
        """The non-None fields, as a plain dict (the config-key payload)."""
        return {
            f.name: int(value)
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }


@dataclass(frozen=True)
class DataWorkloadProfile:
    """What a DATASET may lawfully tell the framework (data-loader registration
    contract). Every field optional; ``None`` = the framework derives the value
    or applies its workload-neutral default."""

    input_value_range: Optional[Tuple[float, float]] = None
    eval_subsample_target: Optional[int] = None
    tuning_step_cap_epochs: Optional[float] = None
    calibration: Optional[CalibrationSetPolicy] = None

    def config_updates(self) -> dict[str, Any]:
        """The flat config keys this registration declares (absent = no claim)."""
        updates: dict[str, Any] = {}
        if self.input_value_range is not None:
            updates["input_data_scale"] = float(self.input_value_range[1])
        if self.eval_subsample_target is not None:
            updates["eval_subsample_target"] = int(self.eval_subsample_target)
        if self.tuning_step_cap_epochs is not None:
            updates["tuning_step_cap_epochs"] = float(self.tuning_step_cap_epochs)
        if self.calibration is not None and (declared := self.calibration.declared_fields()):
            updates["calibration_set_policy"] = declared
        return updates


@dataclass(frozen=True)
class ModelWorkloadProfile:
    """What a MODEL ARCHITECTURE may lawfully tell the framework (model-builder
    registration contract). Every field optional, mapped 1:1 to a config key."""

    prefix_stage_lr: Optional[float] = None
    endpoint_floor_lr: Optional[float] = None
    pretrained_weight_source: Optional[str] = None
    proven_recovery_depth: Optional[int] = None
    clamp_cuda_assert_prone: Optional[bool] = None

    def config_updates(self) -> dict[str, Any]:
        """The flat config keys this registration declares (absent = no claim)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


def fold_workload_profiles(
    config: MutableMapping[str, Any],
    *,
    model_profile: Optional[ModelWorkloadProfile] = None,
    data_profile: Optional[DataWorkloadProfile] = None,
) -> None:
    """Inject registered workload facts beneath explicit config keys.

    Precedence, highest first: explicit config > model-builder registration >
    data-loader registration > framework workload-neutral default (applied at
    :meth:`ResolvedWorkloadProfile.from_config` / the consumer).
    """
    for profile in (model_profile, data_profile):
        if profile is None:
            continue
        for key, value in profile.config_updates().items():
            existing = config.get(key)
            if key == "calibration_set_policy" and isinstance(existing, Mapping):
                config[key] = {**value, **existing}
                continue
            config.setdefault(key, value)


@dataclass(frozen=True)
class ResolvedWorkloadProfile:
    """The once-resolved workload view carried by the deployment plan.

    ``None`` fields mean "no registration and no explicit value" — the consumer
    applies its frozen workload-neutral default. ``input_data_scale`` defaults
    to the unit-range identity."""

    input_data_scale: float = 1.0
    eval_subsample_target: Optional[int] = None
    tuning_step_cap_epochs: Optional[float] = None
    calibration: CalibrationSetPolicy = CalibrationSetPolicy()
    prefix_stage_lr: Optional[float] = None
    endpoint_floor_lr: Optional[float] = None
    pretrained_weight_source: Optional[str] = None
    proven_recovery_depth: Optional[int] = None
    clamp_cuda_assert_prone: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ResolvedWorkloadProfile":
        """Read the profile-injectable keys out of a resolved flat config."""

        def opt(key: str, cast):
            value = config.get(key)
            return None if value is None else cast(value)

        calibration_raw = config.get("calibration_set_policy") or {}
        known = {f.name for f in fields(CalibrationSetPolicy)}
        unknown = set(calibration_raw) - known
        if unknown:
            raise ValueError(
                f"calibration_set_policy: unknown fields {sorted(unknown)}; "
                f"known fields: {sorted(known)}"
            )
        return cls(
            input_data_scale=float(config.get("input_data_scale", 1.0)),
            eval_subsample_target=opt("eval_subsample_target", int),
            tuning_step_cap_epochs=opt("tuning_step_cap_epochs", float),
            calibration=CalibrationSetPolicy(
                **{key: int(value) for key, value in calibration_raw.items()}
            ),
            prefix_stage_lr=opt("prefix_stage_lr", float),
            endpoint_floor_lr=opt("endpoint_floor_lr", float),
            pretrained_weight_source=opt("pretrained_weight_source", str),
            proven_recovery_depth=opt("proven_recovery_depth", int),
            clamp_cuda_assert_prone=bool(config.get("clamp_cuda_assert_prone", False)),
        )
