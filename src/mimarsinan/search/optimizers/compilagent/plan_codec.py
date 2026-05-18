"""Bidirectional codec between compilagent ``Plan``s and mimarsinan configs.

A mimarsinan candidate configuration has two top-level keys::

    {
        "model_config":          {<arch key>: <value>, ...},
        "platform_constraints":  {"cores": [{"max_axons", "max_neurons", "count"}, ...],
                                  "target_tq": int, "weight_bits": int, ...},
    }

A compilagent ``Plan`` is an ordered tuple of ``Intervention``s; each
intervention names a ``Target(kind, selector)`` and carries an opaque
``payload``. We use exactly two ``target.kind`` values:

* ``"arch"`` — selector is the model-config key, payload is the chosen
  arch-option value.
* ``"hw.core"`` — selector is ``"<core_index>.<dim>"`` where ``dim`` is
  one of ``max_axons``, ``max_neurons``, ``count``; payload is the chosen
  integer.

Encoding is the inverse: every non-default value in the configuration is
emitted as one ordered intervention. The codec is lossless on every
configuration the search step can reach (``encode_plan(decode_plan(p))``
yields the same ordered intervention set up to ordering).

The defaults the decoder layers under each plan come from the
``SearchSpaceDescription`` and ``JointArchHwProblem`` the optimizer holds,
so a plan that omits a variable inherits its default value rather than
crashing — this matches compilagent's "baseline = empty plan" expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.search.search_space_description import SearchSpaceDescription


# Stable target-kind strings used in `Intervention.target.kind`.
ARCH_KIND = "arch"
HW_CORE_KIND = "hw.core"

# Allowed dim names for ``hw.core`` interventions; matches the order
# ``_decode_hw`` in ``JointArchHwProblem`` consumes.
HW_DIM_NAMES = ("max_axons", "max_neurons", "count")


class PlanCodecError(ValueError):
    """Raised when an intervention cannot be decoded into a config update."""


@dataclass(frozen=True)
class CodecDefaults:
    """Defaults the decoder layers under each plan.

    ``model_config`` maps every searchable arch key to its default value;
    ``platform_constraints`` carries the canonical core list (one entry
    per core type with ``max_axons``/``max_neurons``/``count``) plus the
    fixed ``target_tq`` / ``weight_bits``.

    The decoder treats the defaults as a deep snapshot — incoming
    interventions overlay on top, never mutate it.
    """

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
        """Derive defaults from ``description`` plus any non-searched values.

        For arch keys not currently searched we fall back to the
        ``fixed_model_config`` provided by the search step (mirrors
        ``JointArchHwProblem.fixed_model_config``). The HW defaults come
        from the description's ``to_agent_evolve_example()`` rendering so
        they always satisfy the bounds.
        """

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
                # Carry over non-searched flags (allow_scheduling, has_bias, ...).
                for k, v in fixed_platform_constraints.items():
                    if k not in {"cores", "target_tq", "weight_bits"}:
                        platform_defaults.setdefault(k, v)

        return cls(
            model_config=model_defaults,
            platform_constraints=platform_defaults,
        )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_plan(plan: Any, defaults: CodecDefaults) -> Dict[str, Any]:
    """Translate a compilagent ``Plan`` into a mimarsinan configuration dict.

    ``plan.interventions`` is iterated in order; later interventions on
    the same target win (the same semantics ``Backend.apply_intervention``
    uses by default). Unknown ``target.kind`` values raise
    ``PlanCodecError`` so ``Backend.validate_intervention`` can surface a
    clear retry message to the agent.
    """

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
        # Synthesise a placeholder core so partial plans (the agent has
        # only set one of the dims so far) do not crash. The default
        # values come from the existing core 0 if present, otherwise
        # 64-axon / 64-neuron / 50-count.
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


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_plan(
    configuration: Mapping[str, Any],
    defaults: CodecDefaults,
    *,
    description: SearchSpaceDescription | None = None,
) -> Tuple[Tuple[str, str, Any], ...]:
    """Translate a mimarsinan configuration into an ordered intervention sequence.

    Returns a tuple of ``(target_kind, target_selector, payload)`` triples
    that the optimizer wraps into ``compilagent.Intervention`` objects
    (this avoids a hard import of compilagent at codec module-load time).

    Only values that *differ from defaults* are emitted, mirroring the
    "minimal diff" semantics of an LLM-proposed plan. When ``description``
    is provided we restrict encoding to its searchable variables; without
    it every diff is emitted.
    """

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
