"""Hashable request signature for the wizard / NAS layout-mapping path.

``LayoutMappingRequest`` collapses a wizard request body (or equivalent
internal call site) into a frozen, hash-stable dataclass.  Used as the key
for :class:`mimarsinan.mapping.verification.layout_mapping_service.LayoutMappingService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _freeze(obj: Any) -> Any:
    """Recursively freeze ``obj`` into a hashable canonical form.

    Dicts become sorted tuples of ``(key, _freeze(value))``; lists / tuples
    become tuples; scalars pass through.  Two semantically-equal request
    bodies always produce equal frozen forms regardless of dict-key order.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


@dataclass(frozen=True)
class LayoutMappingRequest:
    """Hashable request key for ``LayoutMappingService``.

    Two distinct slices of the key drive the two cache levels:

    - :meth:`model_identity_key` -- model_repr cache; tiling parameters
      don't affect the mapper graph, only the per-softcore split.
    - :meth:`verification_key` -- verification cache; everything matters.
    """

    model_type: str
    model_config_key: tuple
    input_shape: tuple[int, ...]
    num_classes: int
    target_tq: int
    max_axons: int
    max_neurons: int
    allow_coalescing: bool
    hardware_bias: bool

    @classmethod
    def from_wizard_body(
        cls,
        body: dict,
        *,
        tiling_max_axons: int | None = None,
        tiling_max_neurons: int | None = None,
    ) -> "LayoutMappingRequest":
        from mimarsinan.mapping.verification.wizard_layout_verify import (
            resolve_tiling_params_from_body,
        )

        max_ax_default = int(body.get("max_axons", 1024))
        max_neu_default = int(body.get("max_neurons", 1024))

        if (body.get("core_types") or body.get("cores")) is not None:
            eff_ax, eff_neu, hw_bias, coalescing = resolve_tiling_params_from_body(
                body,
                tiling_max_axons=tiling_max_axons,
                tiling_max_neurons=tiling_max_neurons,
            )
        else:
            eff_ax = int(
                tiling_max_axons
                if tiling_max_axons is not None
                else max_ax_default
            )
            eff_neu = int(
                tiling_max_neurons
                if tiling_max_neurons is not None
                else max_neu_default
            )
            hw_bias = bool(body.get("hardware_bias", False))
            coalescing = bool(body.get("allow_coalescing", False))

        return cls(
            model_type=str(body.get("model_type", "simple_mlp")),
            model_config_key=_freeze(body.get("model_config", {})),
            input_shape=tuple(int(d) for d in body.get("input_shape", [1, 28, 28])),
            num_classes=int(body.get("num_classes", 10)),
            target_tq=int(body.get("target_tq", 32)),
            max_axons=int(eff_ax),
            max_neurons=int(eff_neu),
            allow_coalescing=bool(coalescing),
            hardware_bias=bool(hw_bias),
        )

    def model_identity_key(self) -> tuple:
        """Sub-key for the model-repr cache slot.

        Tiling parameters (``max_axons``, ``max_neurons``, ``allow_coalescing``,
        ``hardware_bias``) do not change the mapper graph -- only the
        per-softcore split inside ``LayoutIRMapping`` -- so two requests
        differing only on these fields share their model_repr.
        """
        return (
            self.model_type,
            self.model_config_key,
            self.input_shape,
            self.num_classes,
            self.target_tq,
        )

    def verification_key(self) -> tuple:
        """Full key for the verification (softcore-list) cache slot."""
        return (
            self.model_identity_key(),
            self.max_axons,
            self.max_neurons,
            self.allow_coalescing,
            self.hardware_bias,
        )

    def to_body(self) -> dict:
        """Reconstruct a wizard-style body dict (for callers that hand off
        to existing model-repr builders)."""
        return {
            "model_type": self.model_type,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "model_config": _unfreeze(self.model_config_key),
            "target_tq": self.target_tq,
            "max_axons": self.max_axons,
            "max_neurons": self.max_neurons,
            "allow_coalescing": self.allow_coalescing,
            "hardware_bias": self.hardware_bias,
        }


def _unfreeze(obj: Any) -> Any:
    """Inverse of ``_freeze`` for the dict-shaped subtrees we serialise."""
    if isinstance(obj, tuple) and obj and all(
        isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)
        for x in obj
    ):
        return {k: _unfreeze(v) for k, v in obj}
    if isinstance(obj, tuple):
        return [_unfreeze(x) for x in obj]
    return obj
