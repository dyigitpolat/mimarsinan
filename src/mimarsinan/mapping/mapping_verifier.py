"""
Mapping verification: confirm a model (native or torch-converted) can be represented
as soft-core mappings and retrieve shape/requirement information.

Step 1 in the hardware planning workflow:
  - Run LayoutIRMapping over the model to collect LayoutSoftCoreSpec shapes.
  - Optionally compare against actual IRMapping to ensure 1-1 correspondence.
  - Return a structured result usable by the greedy HW config suggester.

``verify_hardware_config`` notes
---------------------------------
- Does **not** pre-check ``total_count >= len(softcores)``.  Multiple softcores
  can be packed into a single hardware core (bin-packing), so fewer hardware
  cores than softcores is perfectly valid.  ``pack_layout`` is the sole arbiter
  of feasibility.
- The ``"total_count"`` key in ``field_errors`` only appears when packing
  genuinely fails — it is never raised as a speculative pre-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec, LayoutHardCoreType
from mimarsinan.mapping.layout.layout_packer import pack_layout


@dataclass
class MappingVerificationResult:
    """Result of a soft-core mapping verification pass."""

    feasible: bool
    softcores: List[LayoutSoftCoreSpec]

    # Summary statistics
    num_neural_cores: int
    max_input_size: int
    max_output_size: int
    total_area: int

    # Optional error message when feasible=False
    error: Optional[str] = None


def verify_soft_core_mapping(
    model_repr,
    max_axons: int,
    max_neurons: int,
    *,
    threshold_groups: int = 1,
    pruning_fraction: float = 0.0,
    threshold_seed: int = 0,
) -> MappingVerificationResult:
    """Verify that a mapper-graph model representation can be laid out as soft cores.

    Parameters
    ----------
    model_repr:
        A ``ModelRepresentation`` (or any object exposing ``map_to_ir(mapping)``)
        for a native mimarsinan model, OR the ``mapper_repr`` obtained from a
        torch-converted ``Supermodel`` via ``supermodel.get_mapper_repr()``.
    max_axons:
        Maximum axon count per hardware core.
    max_neurons:
        Maximum neuron count per hardware core.
    threshold_groups:
        Number of threshold groups to simulate (random assignment).
    pruning_fraction:
        Expected pruning fraction (0–1). The estimator applies 80% of this
        as a random row/column reduction on each softcore.
    threshold_seed:
        RNG seed for deterministic threshold-group assignment.

    Returns
    -------
    MappingVerificationResult
    """
    try:
        layout = LayoutIRMapping(
            max_axons=max_axons,
            max_neurons=max_neurons,
            threshold_groups=threshold_groups,
            threshold_seed=threshold_seed,
            pruning_fraction=pruning_fraction,
        )
        softcores = layout.collect_layout_softcores(model_repr)
    except Exception as exc:
        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            error=str(exc),
        )

    if not softcores:
        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            error="No neural cores produced by mapping — model may have no perceptron layers.",
        )

    max_in = max(sc.input_count for sc in softcores)
    max_out = max(sc.output_count for sc in softcores)
    total_area = sum(sc.area for sc in softcores)

    return MappingVerificationResult(
        feasible=True,
        softcores=softcores,
        num_neural_cores=len(softcores),
        max_input_size=max_in,
        max_output_size=max_out,
        total_area=total_area,
    )


def verify_hardware_config(
    softcores: List[LayoutSoftCoreSpec],
    core_types: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether a hardware core configuration is sufficient for the given softcores.

    Parameters
    ----------
    softcores:
        List of ``LayoutSoftCoreSpec`` from ``verify_soft_core_mapping``.
    core_types:
        List of dicts with keys ``max_axons``, ``max_neurons``, ``count``.

    Returns
    -------
    dict with keys:
        - ``feasible`` (bool): True if all softcores can be packed.
        - ``errors`` (list[str]): Human-readable error messages (empty on success).
        - ``field_errors`` (dict): Per-field error hints for UI display.
          Keys: ``"max_axons"``, ``"max_neurons"``, ``"count"`` per core-type index.
        - ``packing_result``: ``LayoutPackingResult`` (or None on fatal error).
    """
    errors: List[str] = []
    field_errors: Dict[str, str] = {}

    if not softcores:
        return {
            "feasible": False,
            "errors": ["No softcores to verify against."],
            "field_errors": {},
            "packing_result": None,
        }

    if not core_types:
        return {
            "feasible": False,
            "errors": ["No core types defined. Add at least one core type."],
            "field_errors": {"core_types": "Add at least one core type."},
            "packing_result": None,
        }

    max_req_axons = max(sc.input_count for sc in softcores)
    max_req_neurons = max(sc.output_count for sc in softcores)

    # Build LayoutHardCoreType list and check per-type feasibility
    hw_types: List[LayoutHardCoreType] = []
    for i, ct in enumerate(core_types):
        ax = int(ct.get("max_axons", 0))
        neu = int(ct.get("max_neurons", 0))
        cnt = int(ct.get("count", 0))
        hw_types.append(LayoutHardCoreType(max_axons=ax, max_neurons=neu, count=cnt))

        if ax < max_req_axons:
            field_errors[f"core_type_{i}_max_axons"] = (
                f"Core type {i+1}: max_axons={ax} is too small. "
                f"Need at least {max_req_axons} axons to accommodate the largest soft core."
            )
        if neu < max_req_neurons:
            field_errors[f"core_type_{i}_max_neurons"] = (
                f"Core type {i+1}: max_neurons={neu} is too small. "
                f"Need at least {max_req_neurons} neurons to accommodate the largest soft core."
            )

    # Attempt greedy packing — this is the precise feasibility check.
    # (We do NOT pre-check total_count < len(softcores): multiple softcores can
    # share a single hardware core, so far fewer than len(softcores) cores may suffice.)
    result = pack_layout(softcores=softcores, core_types=hw_types)

    if not result.feasible:
        err_msg = result.error or "Hardware configuration cannot fit all soft cores."
        errors.append(err_msg)
        total_core_count = sum(int(ct.get("count", 0)) for ct in core_types)
        field_errors["total_count"] = (
            f"Packing failed ({total_core_count} cores for {len(softcores)} soft cores): "
            f"{err_msg}. Increase core counts or core dimensions."
        )

    errors = list(field_errors.values()) if field_errors else errors

    return {
        "feasible": result.feasible,
        "errors": errors,
        "field_errors": field_errors,
        "packing_result": result,
    }
