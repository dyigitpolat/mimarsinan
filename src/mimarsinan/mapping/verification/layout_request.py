"""Hashable request signature (key for ``LayoutMappingService``) for the wizard / NAS layout-mapping path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.mapping.verification.wizard_layout_verify import (
    resolve_tiling_params_from_body,
)


def _freeze(obj: Any) -> Any:
    """Recursively freeze ``obj`` into a hashable canonical form.

    Dicts become sorted tuples of ``(key, _freeze(value))``; lists/tuples become
    tuples; scalars pass through, so key order never changes the frozen form.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


@dataclass(frozen=True)
class LayoutMappingRequest:
    """Hashable request key for ``LayoutMappingService``.

    ``model_identity_key`` drives the model_repr cache (tiling-independent);
    ``verification_key`` drives the verification cache (everything matters).
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
    encoding_layer_placement: str = "subsume"

    @classmethod
    def from_wizard_body(
        cls,
        body: dict,
        *,
        tiling_max_axons: int | None = None,
        tiling_max_neurons: int | None = None,
    ) -> "LayoutMappingRequest":
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
            encoding_layer_placement=str(
                body.get("encoding_layer_placement", "subsume")
            ),
        )

    def model_identity_key(self) -> tuple:
        """Sub-key for the model-repr cache slot; excludes tiling params, which don't change the mapper graph."""
        return (
            self.model_type,
            self.model_config_key,
            self.input_shape,
            self.num_classes,
            self.target_tq,
            self.encoding_layer_placement,
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
        """Reconstruct a wizard-style body dict for existing model-repr builders."""
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
            "encoding_layer_placement": self.encoding_layer_placement,
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
