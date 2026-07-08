"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_cycle_based,
    requires_ttfs_firing,
)
from mimarsinan.common.env import (
    UNSAFE_QUANT_OVERRIDES_VAR,
    unsafe_quant_overrides_enabled,
)
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

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


def _fold_conversion_recipe(dp: MutableMapping[str, Any], spiking_mode: str) -> None:
    """Fold the ConversionPolicy SSOT recipe for ``spiking_mode`` into ``dp``.

    sim_enables, driver, and per-mode knobs are written authoritatively
    (Pure SSOT): any user value for them is overwritten by the recipe.
    """
    recipe = ConversionPolicy.derive(spiking_mode, dp.get("ttfs_cycle_schedule"))
    for key, value in recipe.sim_enables.items():
        dp[key] = value
    dp["optimization_driver"] = recipe.driver
    for key, value in recipe.knobs.items():
        dp[key] = value


def _resolve_activation_quantization(
    explicit_aq: Optional[Any], derived_aq: bool, *, spiking_mode: str, float_weights: bool
) -> bool:
    """Derived AQ unless a contradicting explicit value raises or is force-honored."""
    if explicit_aq is None or bool(explicit_aq) == derived_aq:
        return derived_aq
    regime = (
        "float-weight (vanilla) deployment"
        if float_weights
        else f"spiking_mode={spiking_mode!r}"
    )
    detail = (
        f"explicit activation_quantization={bool(explicit_aq)} contradicts the "
        f"derived value {derived_aq} for {regime}."
    )
    if unsafe_quant_overrides_enabled():
        _unsafe_override_log(detail + " Explicit value honored")
        return bool(explicit_aq)
    raise _contract_error(detail, _AQ_RULE)


def derive_deployment_parameters(dp: MutableMapping[str, Any]) -> None:
    """Derive AQ/WQ/pipeline_mode in-place — the ONLY derivation implementation
    (the wizard consumes it via ``/api/config/resolve``; no JS copy exists)."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    pipeline_mode = str(dp.get("pipeline_mode", ""))
    explicit_aq = dp.get("activation_quantization")
    float_weights = pipeline_mode == "vanilla" or not bool(dp.get("weight_quantization", True))

    _fold_conversion_recipe(dp, spiking_mode)

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


def derive_pipeline_runtime_parameters(dp: MutableMapping[str, Any]) -> None:
    """Fill runtime spiking fields that minimal persisted configs may omit."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    if requires_ttfs_firing(spiking_mode):
        dp.setdefault("firing_mode", "TTFS")
        dp.setdefault("spike_generation_mode", "TTFS")
        dp.setdefault("thresholding_mode", "<=")
        for key in ("firing_mode", "spike_generation_mode"):
            if dp[key] != "TTFS":
                raise ValueError(
                    f"spiking_mode='{spiking_mode}' requires {key}='TTFS', "
                    f"got '{dp[key]}'"
                )
    else:
        dp.setdefault("firing_mode", "Default")
        dp.setdefault("spike_generation_mode", "Uniform")
        dp.setdefault("thresholding_mode", "<=")
        dp.setdefault("cycle_accurate_lif_forward", True)
