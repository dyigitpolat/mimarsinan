"""Platform-constraints derivation: cores-derived maxima and cross-key tiling."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping


def _require_tq_divides_simulation_steps(pc: Mapping[str, Any]) -> None:
    """[registry: target_tq doc] The QAT quantization level Tq must tile the
    deployment spike window S — the deployment identity needs the trained grid
    to divide the simulator window. Enforced fail-loud over already-valid
    positive ints; a one-operand config (search sets Tq without S) or a
    per-field-invalid value (type/bounds — the registry surfaces its own keyed
    error) is not cross-checkable and passes through untouched."""
    tq = pc.get("target_tq")
    s = pc.get("simulation_steps")
    if not isinstance(tq, int) or not isinstance(s, int):
        return
    if isinstance(tq, bool) or isinstance(s, bool) or tq <= 0 or s <= 0:
        return
    if s % tq != 0:
        raise ValueError(
            f"target_tq={tq} must divide simulation_steps={s} (the QAT "
            f"activation-quantization level must tile the deployment spike "
            f"window; the tier configs set them equal). Fix target_tq or "
            f"simulation_steps so simulation_steps % target_tq == 0."
        )


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
    _require_tq_divides_simulation_steps(pc)
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
