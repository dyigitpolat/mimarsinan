"""Canonical hypervolume-axis encoding for configs, plans, and ledger rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_VALID_SYNC = ("cascaded", "synchronized")
_DEPTH_IN_RUN_ID = re.compile(r"_d(\d+)_")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pretrained"}
    return bool(value)


def quantization_axis(
    *, weight_quantization: Any = False, activation_quantization: Any = False,
) -> str:
    """Encode the four-value quantization axis from activation/weight booleans."""
    wq = _truthy(weight_quantization)
    aq = _truthy(activation_quantization)
    if wq and aq:
        return "wq_aq"
    if wq:
        return "wq"
    if aq:
        return "aq"
    return "none"


def pruning_axis(
    *,
    prune_sparsity: Any = 0.0,
    pruning: Any = False,
    pruning_fraction: Any = 0.0,
) -> str:
    """Encode the deployment pruning axis (structured ``prune_sparsity`` or the legacy ``pruning`` pair)."""
    if float(prune_sparsity or 0.0) > 0.0:
        return "pruned"
    if _truthy(pruning) and float(pruning_fraction or 0.0) > 0.0:
        return "pruned"
    return "dense"


def regime_axis(*, weight_source: Any = None, preload_weights: Any = False) -> str:
    """Encode the training regime axis from resolved source or preload intent."""
    if weight_source:
        return "pretrained"
    if _truthy(preload_weights):
        return "pretrained"
    return "from_scratch"


def normalize_dataset(value: Any) -> str:
    """Ledger/provider dataset names to the compact hypervolume coordinate."""
    if not value:
        return "unknown"
    text = str(value).strip()
    text = text.replace("_DataProvider", "").replace("DataProvider", "")
    text = text.lower()
    aliases = {
        "fashionmnist": "fmnist",
        "fashion_mnist": "fmnist",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
    }
    return aliases.get(text, text)


def schedule_sync(schedule: Any) -> str:
    """Normalize one schedule token to a sync-axis value."""
    return str(schedule) if schedule in _VALID_SYNC else "none"


def syncs_from_row(row: Mapping[str, Any]) -> list[str]:
    """The sync-axis value(s) covered by a ledger row."""
    explicit = schedule_sync(row.get("schedule"))
    if explicit in _VALID_SYNC:
        return [explicit]
    reported = [
        sync
        for sync, key in (
            ("cascaded", "cascaded_deployed_mean"),
            ("synchronized", "synchronized_deployed_mean"),
        )
        if row.get(key) is not None
    ]
    return reported or ["none"]


def _depth_from_row(row: Mapping[str, Any], *, axis_wildcard: str) -> str:
    explicit = row.get("depth")
    if explicit is not None and str(explicit) != "":
        return str(explicit)
    for field_name in ("run_id", "run_ids", "cascaded_run_ids", "synchronized_run_ids"):
        value = row.get(field_name)
        candidates: Sequence[Any] = value if isinstance(value, (list, tuple)) else (value,)
        for candidate in candidates:
            if not candidate:
                continue
            match = _DEPTH_IN_RUN_ID.search(str(candidate))
            if match:
                return match.group(1)
    return axis_wildcard


def _depth_from_config(config: Mapping[str, Any], *, axis_wildcard: str) -> str:
    explicit = config.get("depth")
    if explicit is not None and str(explicit) != "":
        return str(explicit)
    model_config = config.get("model_config")
    if isinstance(model_config, Mapping):
        depth = model_config.get("depth")
        if depth is not None and str(depth) != "":
            return str(depth)
    return axis_wildcard


@dataclass(frozen=True)
class AxisCoordinates:
    """All non-collapsed hypervolume cell coordinates, before cell construction."""

    firing: str
    sync: str
    backend: str
    vehicle: str
    dataset: str
    regime: str
    quantization: str
    pruning: str
    mapping_strategy: str
    s: str
    depth: str

    def as_cell_kwargs(self) -> dict[str, str]:
        return {
            "firing": self.firing,
            "sync": self.sync,
            "backend": self.backend,
            "vehicle": self.vehicle,
            "dataset": self.dataset,
            "regime": self.regime,
            "quantization": self.quantization,
            "pruning": self.pruning,
            "mapping_strategy": self.mapping_strategy,
            "s": self.s,
            "depth": self.depth,
        }

    @classmethod
    def from_plan(
        cls,
        plan: Any,
        config: Mapping[str, Any] | None = None,
        *,
        backend: str = "sanafe",
        mapping_strategy: str = "packed",
        axis_wildcard: str = "any",
    ) -> "AxisCoordinates":
        """Project a resolved ``DeploymentPlan`` into hypervolume coordinates."""
        cfg = config or getattr(plan, "config", {}) or {}
        sync = "none"
        if bool(getattr(plan, "is_ttfs_cycle_based", False)):
            sync = str(getattr(plan, "ttfs_cycle_schedule", "none"))
        dataset = normalize_dataset(cfg.get("dataset") or cfg.get("data_provider_name"))
        s_value = cfg.get("S", cfg.get("simulation_steps", axis_wildcard))
        return cls(
            firing=str(getattr(plan, "spiking_mode", cfg.get("spiking_mode", "lif"))),
            sync=sync,
            backend=str(backend),
            vehicle=str(getattr(plan, "model_type", cfg.get("model_type", ""))),
            dataset=dataset,
            regime=regime_axis(
                weight_source=getattr(plan, "weight_source", cfg.get("weight_source")),
                preload_weights=cfg.get("preload_weights", False),
            ),
            quantization=quantization_axis(
                weight_quantization=getattr(
                    plan, "weight_quantization", cfg.get("weight_quantization", False)
                ),
                activation_quantization=getattr(
                    plan,
                    "activation_quantization",
                    cfg.get("activation_quantization", False),
                ),
            ),
            pruning=pruning_axis(
                prune_sparsity=getattr(plan, "prune_sparsity", cfg.get("prune_sparsity", 0.0)),
                pruning=getattr(plan, "pruning", cfg.get("pruning", False)),
                pruning_fraction=getattr(
                    plan, "pruning_fraction", cfg.get("pruning_fraction", 0.0)
                ),
            ),
            mapping_strategy=str(mapping_strategy),
            s=str(s_value),
            depth=_depth_from_config(cfg, axis_wildcard=axis_wildcard),
        )


def cell_coordinates_from_row(
    row: Mapping[str, Any], *, sync: str, axis_wildcard: str = "any",
) -> AxisCoordinates:
    """Project one ledger row + one sync arm into hypervolume coordinates."""
    s_value = row.get("S", row.get("simulation_steps", axis_wildcard))
    quantization = row.get("quantization")
    if quantization is None:
        quantization = quantization_axis(
            weight_quantization=row.get("weight_quantization", False),
            activation_quantization=row.get("activation_quantization", False),
        )
    pruning = row.get("pruning")
    if pruning is None or isinstance(pruning, bool):
        pruning = pruning_axis(
            prune_sparsity=row.get("prune_sparsity", 0.0),
            pruning=pruning if pruning is not None else row.get("pruning_enabled", False),
            pruning_fraction=row.get("pruning_fraction", 0.0),
        )
    regime = row.get("regime")
    if regime is None:
        regime = regime_axis(
            weight_source=row.get("weight_source"),
            preload_weights=row.get("preload_weights", False),
        )
    return AxisCoordinates(
        firing=str(row.get("spiking_mode") or row.get("mode") or "lif"),
        sync=sync,
        backend=str(row.get("backend") or "sanafe"),
        vehicle=str(row.get("model") or row.get("model_type") or ""),
        dataset=normalize_dataset(row.get("dataset")),
        regime=str(regime),
        quantization=str(quantization),
        pruning=str(pruning),
        mapping_strategy=str(row.get("mapping_strategy") or "packed"),
        s=str(s_value),
        depth=_depth_from_row(row, axis_wildcard=axis_wildcard),
    )
