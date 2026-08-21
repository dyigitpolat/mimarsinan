"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional, Set

from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
from mimarsinan.chip_simulation.core_semantics import (
    is_mvm_core_semantics,
    resolve_core_semantics,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_cycle_based,
)
from mimarsinan.common.env import (
    UNSAFE_QUANT_OVERRIDES_VAR,
    unsafe_quant_overrides_enabled,
)
from mimarsinan.config_schema.derivation.legality import (
    legal_value_error, legal_values_for, legality_bearing_keys)
from mimarsinan.config_schema.derivation.platform import (
    derive_platform_constraints as derive_platform_constraints)
from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe, fold_mvm_recipe
from mimarsinan.config_schema.registry import REGISTRY

_AQ_RULE = (
    "activation_quantization is derived from the deployment mode "
    "(SSOT: config_schema/deployment_derivation.py): ON for spiking_mode in "
    "{lif, ttfs_quantized, ttfs_cycle_based}; OFF for analytical ttfs and for "
    "float-weight (vanilla) deployments."
)

_MVM_AQ_RULE = (
    "the value-domain (core_semantics='mvm') family derives "
    "activation_quantization from the platform: True iff activation_bits is "
    "declared (boundary value-grid quantization); on-chip activations do not "
    "exist, so the event AQ ladder never applies either way."
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
    explicit_aq: Optional[Any], derived_aq: bool, *, regime: str, rule: str = _AQ_RULE
) -> bool:
    """Derived AQ unless a contradicting explicit value raises or is force-honored."""
    if explicit_aq is None or bool(explicit_aq) == derived_aq:
        return derived_aq
    detail = (
        f"explicit activation_quantization={bool(explicit_aq)} contradicts the "
        f"derived value {derived_aq} for {regime}."
    )
    if unsafe_quant_overrides_enabled():
        _unsafe_override_log(detail + " Explicit value honored")
        return bool(explicit_aq)
    raise _contract_error(detail, rule)


def derive_deployment_parameters(
    dp: MutableMapping[str, Any],
    explicit_keys: Optional[Iterable[str]] = None,
) -> None:
    """Derive AQ/WQ/pipeline_mode in-place — the ONLY derivation implementation
    (the wizard consumes it via ``/api/config/resolve``; no JS copy exists).
    ``explicit_keys`` names the keys the source DOCUMENT declared (so merged
    defaults don't masquerade as declarations); ``None`` = every present key."""
    mvm = is_mvm_core_semantics(resolve_core_semantics(dp))
    if not mvm:
        # Fold the authored (family, variant) axes into their legacy twins
        # BEFORE anything reads spiking_mode/ttfs_cycle_schedule.
        fold_spiking_axes(dp)
    mvm_aq = mvm and bool(dp.get("activation_bits"))
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    pipeline_mode = str(dp.get("pipeline_mode", ""))
    explicit_aq = dp.get("activation_quantization")
    float_weights = pipeline_mode == "vanilla" or not bool(dp.get("weight_quantization", True))
    aq_regime = ("the mvm (value-domain) family" if mvm
                 else f"spiking_mode={spiking_mode!r}")
    aq_rule = _MVM_AQ_RULE if mvm else _AQ_RULE

    if mvm:
        fold_mvm_recipe(dp, explicit_keys)
    else:
        fold_conversion_recipe(dp, spiking_mode, explicit_keys)
    _fold_mirror_training_recipe(
        dp, set(dp) if explicit_keys is None else set(explicit_keys)
    )

    if float_weights:
        dp["pipeline_mode"] = "vanilla"
        dp["weight_quantization"] = False
        dp["activation_quantization"] = _resolve_activation_quantization(
            explicit_aq, mvm_aq if mvm else False,
            regime=(aq_regime if mvm else "float-weight (vanilla) deployment"),
            rule=aq_rule,
        )
        return

    derived_aq = mvm_aq if mvm else (
        forces_activation_quantization(spiking_mode) or is_cycle_based(spiking_mode)
    )
    act_quant = _resolve_activation_quantization(
        explicit_aq, derived_aq, regime=aq_regime, rule=aq_rule
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


def derive_pipeline_runtime_parameters(dp: MutableMapping[str, Any]) -> None:
    """Fill runtime spiking fields that minimal persisted configs may omit.

    Generic: the registry's ``derived_default`` supplies the mode-aware value of
    every legality-bearing key, and its ``legal_values`` set judges an explicit
    one — no per-mode ladder lives here. A legality-bearing key with a SCHEMA
    default (s_allocation) is never filled here; the validators judge it.
    """
    if not is_mvm_core_semantics(resolve_core_semantics(dp)):
        fold_spiking_axes(dp)
        # Recipe-owned mapping arm: the windowed-lif exact-QAT pairing writes
        # True before this runs; inert False for every other mode — always
        # resolved, never a knob (RETIRED as a document key).
        dp.setdefault("lif_per_hop_retiming", False)
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
