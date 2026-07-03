"""Bidirectional codec between compilagent Plans and mimarsinan configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.search.search_space_description import SearchSpaceDescription


ARCH_KIND = "arch"
HW_CORE_KIND = "hw.core"

HW_DIM_NAMES = ("max_axons", "max_neurons", "count")


class PlanCodecError(ValueError):
    """Raised when an intervention cannot be decoded into a config update."""


@dataclass(frozen=True)
class CodecDefaults:
    """Defaults the decoder layers under each plan."""

    model_config: Mapping[str, Any]
    platform_constraints: Mapping[str, Any]

    @classmethod
    def from_description(
        cls,
        description: SearchSpaceDescription,
        *,
        fixed_model_config: Mapping[str, Any] | None = None,
        fixed_platform_constraints: Mapping[str, Any] | None = None,
    ) -> "CodecDefaults":
        """Derive defaults from description plus any non-searched values."""

        model_defaults: Dict[str, Any] = dict(fixed_model_config or {})
        if description.searches_model:
            for key, values in description.arch_options:
                if not values:
                    continue
                model_defaults.setdefault(key, values[len(values) // 2])

        if fixed_platform_constraints is not None and not description.searches_hw:
            platform_defaults: Dict[str, Any] = {
                k: v for k, v in fixed_platform_constraints.items()
            }
            platform_defaults.setdefault("cores", list(fixed_platform_constraints.get("cores", [])))
        else:
            example_pcfg = description.to_agent_evolve_example().get(
                "platform_constraints", {}
            )
            platform_defaults = {
                "cores": [dict(c) for c in example_pcfg.get("cores", [])],
                "target_tq": int(description.target_tq),
                "weight_bits": int(description.weight_bits),
            }
            if fixed_platform_constraints:
                for k, v in fixed_platform_constraints.items():
                    if k not in {"cores", "target_tq", "weight_bits"}:
                        platform_defaults.setdefault(k, v)

        return cls(
            model_config=model_defaults,
            platform_constraints=platform_defaults,
        )


def decode_plan(plan: Any, defaults: CodecDefaults) -> Dict[str, Any]:
    """Translate a compilagent Plan into a mimarsinan configuration dict."""

    model_config: Dict[str, Any] = dict(defaults.model_config)
    platform_constraints: Dict[str, Any] = _deep_copy_platform(defaults.platform_constraints)

    interventions = getattr(plan, "interventions", ())
    for iv in interventions:
        target = getattr(iv, "target", None)
        if target is None:
            raise PlanCodecError("intervention is missing `target`")
        kind = getattr(target, "kind", None)
        selector = getattr(target, "selector", "") or ""
        payload = getattr(iv, "payload", None)

        if kind == ARCH_KIND:
            if not selector:
                raise PlanCodecError("arch intervention requires non-empty selector")
            model_config[str(selector)] = payload
        elif kind == HW_CORE_KIND:
            _apply_hw_core(platform_constraints, str(selector), payload)
        else:
            raise PlanCodecError(
                f"unknown intervention kind `{kind}` (expected one of "
                f"{ARCH_KIND!r}, {HW_CORE_KIND!r})"
            )

    return {
        "model_config": model_config,
        "platform_constraints": platform_constraints,
    }


def _deep_copy_platform(pcfg: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pcfg.items():
        if k == "cores":
            out[k] = [dict(c) for c in v]
        else:
            out[k] = v
    return out


def _apply_hw_core(
    platform_constraints: Dict[str, Any],
    selector: str,
    payload: Any,
) -> None:
    parts = selector.split(".")
    if len(parts) != 2:
        raise PlanCodecError(
            f"hw.core selector must be `<core_index>.<dim>`, got {selector!r}"
        )
    try:
        idx = int(parts[0])
    except ValueError as exc:
        raise PlanCodecError(
            f"hw.core selector core index must be integer, got {parts[0]!r}"
        ) from exc
    dim = parts[1]
    if dim not in HW_DIM_NAMES:
        raise PlanCodecError(
            f"hw.core dim must be one of {HW_DIM_NAMES}, got {dim!r}"
        )
    cores: List[Dict[str, Any]] = list(platform_constraints.get("cores", []))
    while len(cores) <= idx:
        template = cores[0] if cores else {"max_axons": 64, "max_neurons": 64, "count": 50}
        cores.append(dict(template))
    try:
        value = int(payload)
    except (TypeError, ValueError) as exc:
        raise PlanCodecError(
            f"hw.core payload for {selector} must be an integer, got {payload!r}"
        ) from exc
    cores[idx][dim] = value
    platform_constraints["cores"] = cores


def encode_plan(
    configuration: Mapping[str, Any],
    defaults: CodecDefaults,
    *,
    description: SearchSpaceDescription | None = None,
) -> Tuple[Tuple[str, str, Any], ...]:
    """Translate a mimarsinan configuration into an ordered intervention sequence."""

    triples: List[Tuple[str, str, Any]] = []
    model_cfg: Mapping[str, Any] = configuration.get("model_config", {}) or {}
    pcfg: Mapping[str, Any] = configuration.get("platform_constraints", {}) or {}

    arch_keys = (
        {key for key, _ in description.arch_options}
        if description is not None and description.searches_model
        else set(model_cfg.keys()) | set(defaults.model_config.keys())
    )
    for key in sorted(arch_keys):
        value = model_cfg.get(key, defaults.model_config.get(key))
        default = defaults.model_config.get(key)
        if value is None or value == default:
            continue
        triples.append((ARCH_KIND, key, value))

    if description is None or description.searches_hw:
        cores: Sequence[Mapping[str, Any]] = pcfg.get("cores", []) or []
        default_cores: Sequence[Mapping[str, Any]] = defaults.platform_constraints.get(
            "cores", []
        ) or []
        for idx, core in enumerate(cores):
            default_core = default_cores[idx] if idx < len(default_cores) else {}
            for dim in HW_DIM_NAMES:
                if dim not in core:
                    continue
                value = int(core[dim])
                default_value = (
                    int(default_core[dim]) if dim in default_core else None
                )
                if value == default_value:
                    continue
                triples.append((HW_CORE_KIND, f"{idx}.{dim}", value))

    return tuple(triples)


__all__ = [
    "ARCH_KIND",
    "HW_CORE_KIND",
    "HW_DIM_NAMES",
    "CodecDefaults",
    "PlanCodecError",
    "decode_plan",
    "encode_plan",
]
