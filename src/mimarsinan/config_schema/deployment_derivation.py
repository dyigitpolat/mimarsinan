"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional, Set, Tuple

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_cycle_based,
)
from mimarsinan.common.env import (
    UNSAFE_QUANT_OVERRIDES_VAR,
    unsafe_quant_overrides_enabled,
)
from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe
from mimarsinan.config_schema.registry import REGISTRY

_AQ_RULE = (
    "activation_quantization is derived from the deployment mode "
    "(SSOT: config_schema/deployment_derivation.py): ON for spiking_mode in "
    "{lif, ttfs_quantized, ttfs_cycle_based}; OFF for analytical ttfs and for "
    "float-weight (vanilla) deployments."
)


def _contract_error(detail: str, rule: str) -> ValueError:
    return ValueError(
        f"quantization contract violation: {detail} {rule} "
        f"Remove the explicit key to accept the derivation, or set "
        f"{UNSAFE_QUANT_OVERRIDES_VAR}=1 to honor the explicit value "
        f"(unsafe research override)."
    )


def _unsafe_override_log(detail: str) -> None:
    print(f"[UNSAFE-OVERRIDE] {detail} ({UNSAFE_QUANT_OVERRIDES_VAR}=1)")


def _fold_mirror_training_recipe(dp: MutableMapping[str, Any], explicit: Set[str]) -> None:
    """mirror_training_recipe=true reflects the (effective) training recipe
    as-is into the tuning recipe; a document-declared tuning_recipe conflicts."""
    if not bool(dp.get("mirror_training_recipe")):
        return
    if "tuning_recipe" in explicit:
        raise ValueError(
            "mirror_training_recipe=true owns tuning_recipe (the tuning recipe "
            "reflects the training recipe). Drop the explicit tuning_recipe "
            "declaration or disable the mirror."
        )
    recipe = dp.get("training_recipe")
    dp["tuning_recipe"] = dict(recipe) if isinstance(recipe, dict) else recipe


def _resolve_activation_quantization(
    explicit_aq: Optional[Any], derived_aq: bool, *, spiking_mode: str, float_weights: bool
) -> bool:
    """Derived AQ unless a contradicting explicit value raises or is force-honored."""
    if explicit_aq is None or bool(explicit_aq) == derived_aq:
        return derived_aq
    regime = ("float-weight (vanilla) deployment" if float_weights
              else f"spiking_mode={spiking_mode!r}")
    detail = (
        f"explicit activation_quantization={bool(explicit_aq)} contradicts the "
        f"derived value {derived_aq} for {regime}."
    )
    if unsafe_quant_overrides_enabled():
        _unsafe_override_log(detail + " Explicit value honored")
        return bool(explicit_aq)
    raise _contract_error(detail, _AQ_RULE)


def derive_deployment_parameters(
    dp: MutableMapping[str, Any],
    explicit_keys: Optional[Iterable[str]] = None,
) -> None:
    """Derive AQ/WQ/pipeline_mode in-place — the ONLY derivation implementation
    (the wizard consumes it via ``/api/config/resolve``; no JS copy exists).
    ``explicit_keys`` names the keys the source DOCUMENT declared (so merged
    defaults don't masquerade as declarations); ``None`` = every present key."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    pipeline_mode = str(dp.get("pipeline_mode", ""))
    explicit_aq = dp.get("activation_quantization")
    float_weights = pipeline_mode == "vanilla" or not bool(dp.get("weight_quantization", True))

    fold_conversion_recipe(dp, spiking_mode, explicit_keys)
    _fold_mirror_training_recipe(
        dp, set(dp) if explicit_keys is None else set(explicit_keys)
    )

    if float_weights:
        dp["pipeline_mode"] = "vanilla"
        dp["weight_quantization"] = False
        dp["activation_quantization"] = _resolve_activation_quantization(
            explicit_aq, False, spiking_mode=spiking_mode, float_weights=True
        )
        return

    derived_aq = forces_activation_quantization(spiking_mode) or is_cycle_based(spiking_mode)
    act_quant = _resolve_activation_quantization(
        explicit_aq, derived_aq, spiking_mode=spiking_mode, float_weights=False
    )
    wt_quant = bool(dp.get("weight_quantization", True))
    dp["activation_quantization"] = act_quant
    dp["weight_quantization"] = wt_quant

    if act_quant or wt_quant:
        dp.setdefault("pipeline_mode", "phased")
    else:
        dp.setdefault("pipeline_mode", "vanilla")


def enforce_quantization_assembly_contract(
    deployment_parameters: Mapping[str, Any],
    platform_constraints: Mapping[str, Any],
    *,
    pipeline_mode: Optional[str],
) -> None:
    """Reject WQ declarations that contradict the assembly (raw config values only).

    Weight quantization is bits-driven: ``weight_bits`` declares a quantized
    artifact, so a float-weight deployment must be declared via the vanilla
    mechanism (``pipeline_mode='vanilla'``, or ``weight_quantization=false``
    without ``weight_bits``) instead of contradicting the bits.
    """
    explicit_wq = deployment_parameters.get("weight_quantization")
    bits_provided = "weight_bits" in platform_constraints

    if pipeline_mode == "vanilla" and explicit_wq is True:
        detail = (
            "explicit weight_quantization=true contradicts pipeline_mode='vanilla' "
            "(vanilla is the float-weight assembly)."
        )
        rule = "Drop weight_quantization or use a phased pipeline_mode."
        if unsafe_quant_overrides_enabled():
            _unsafe_override_log(detail + " Legacy float collapse honored")
            return
        raise _contract_error(detail, rule)

    if explicit_wq is False and bits_provided and pipeline_mode != "vanilla":
        detail = (
            "weight quantization is bits-driven: platform weight_bits declares a "
            "quantized artifact while weight_quantization=false declares float "
            f"weights, and pipeline_mode={pipeline_mode!r} does not arbitrate."
        )
        rule = (
            "Declare float-weight deployment via pipeline_mode='vanilla' "
            "(the fp mechanism), or drop weight_bits."
        )
        if unsafe_quant_overrides_enabled():
            _unsafe_override_log(detail + " Legacy float collapse honored")
            return
        raise _contract_error(detail, rule)


def derive_platform_constraints(
    pc: MutableMapping[str, Any], *, cores_declared: bool = True
) -> None:
    """Derive the scalar per-core maxima from the core grid (wizard parity).

    ``max_axons``/``max_neurons`` are derivable from ``cores`` (the mapping
    itself always re-derives them via ``resolve_platform_mapping_params``);
    an absent scalar is filled, a consistent explicit one accepted, and a
    contradicting one rejected — a scalar the mapping would ignore must not
    masquerade as a constraint. When the document declares only scalars (the
    legacy / hardware-search shape), ``cores_declared=False`` skips the pass:
    the scalars are the only constraint information there.
    """
    if not cores_declared:
        return
    cores = pc.get("cores")
    if not isinstance(cores, list) or not cores:
        return
    for dim in ("max_axons", "max_neurons"):
        values = [
            int(core[dim]) for core in cores
            if isinstance(core, dict)
            and isinstance(core.get(dim), (int, float))
            and not isinstance(core.get(dim), bool)
        ]
        if len(values) != len(cores):
            continue  # incomplete grid mid-edit; shape validation reports it
        derived = max(values)
        explicit = pc.get(dim)
        if explicit is None:
            pc[dim] = derived
        elif int(explicit) != derived:
            raise ValueError(
                f"{dim}={explicit} contradicts the cores-derived value {derived} "
                f"(the largest per-core value across the declared core types; "
                f"the mapping uses the derived value). Drop {dim} to accept "
                f"the derivation, or fix the core grid."
            )


def legal_value_error(flat_key: str, value: Any, legal: Iterable[Any]) -> ValueError:
    """THE canonical illegal-value message (the wizard renders the same text)."""
    options = ", ".join(repr(option) for option in legal)
    return ValueError(
        f"{flat_key}={value!r} is not legal here: the current config admits "
        f"{{{options}}}. Remove {flat_key} to accept the derived value."
    )


def legal_values_for(flat_key: str, cfg: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """The registry's legal value set for ``flat_key`` under this config state.
    ``None`` = legality does not apply here (the rule was not consulted): neither
    locked nor judged. An EMPTY tuple = consulted and admits nothing."""
    result = REGISTRY[flat_key].legal_values(cfg)  # type: ignore[misc]
    return None if result is None else tuple(result)


def legality_bearing_keys() -> Tuple[str, ...]:
    """Keys whose legality depends on other config (registry-declared)."""
    return tuple(k for k, e in REGISTRY.items() if e.legal_values is not None)


def derive_pipeline_runtime_parameters(dp: MutableMapping[str, Any]) -> None:
    """Fill runtime spiking fields that minimal persisted configs may omit.

    Generic: the registry's ``derived_default`` supplies the mode-aware value of
    every legality-bearing key, and its ``legal_values`` set judges an explicit
    one — no per-mode ladder lives here. A legality-bearing key with a SCHEMA
    default (s_allocation) is never filled here; the validators judge it.
    """
    for flat_key in legality_bearing_keys():
        derived = REGISTRY[flat_key].derived_default
        if derived is None:
            continue
        if dp.get(flat_key) is None:
            if (value := derived(dp)) is not None:
                dp[flat_key] = value
        elif (legal := legal_values_for(flat_key, dp)) is not None and (
                dp[flat_key] not in legal):
            raise legal_value_error(flat_key, dp[flat_key], legal)
    # Recipe-owned correctness mechanism (LIF trains the deployed forward);
    # inert for TTFS modes but always resolved, never a knob.
    dp.setdefault("cycle_accurate_lif_forward", True)
    # Boundary-lossless requirement with TWO sound positions (round-5): ON =
    # calibrated shift + bias pre-correction; OFF = the mapper's subsume-forward.
    dp.setdefault("negative_value_shift", True)
