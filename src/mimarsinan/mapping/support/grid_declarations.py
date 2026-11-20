"""Declared value grids (Wave 2 WI-5b to WI-5d): the vocabularies, their readers and the projection's refusals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from mimarsinan.common.refusal import ToolchainRefusal

#: ``weight_scale_rule``: the integer-lattice scale ``max(1, floor(q_max / max|w|))``, or ``q_max / max|w|``.
WEIGHT_SCALE_INTEGER, WEIGHT_SCALE_FRACTIONAL = "integer", "fractional"
WEIGHT_SCALE_RULES = (WEIGHT_SCALE_INTEGER, WEIGHT_SCALE_FRACTIONAL)
#: ``bias_resolution``: the bias on the weight grid (its own grid under two-scale), or held at the readout law's
#: resolution (the law's B; float on the float path).
BIAS_ON_WEIGHT_GRID, BIAS_AT_LAW = "weight_grid", "law"
BIAS_RESOLUTIONS = (BIAS_ON_WEIGHT_GRID, BIAS_AT_LAW)
#: ``readout_law.output_law``: output cores requantize onto the activation grid, or write their exact integers.
OUTPUT_LAW_REQUANTIZE, OUTPUT_LAW_IDENTITY = "requantize", "identity"
OUTPUT_LAWS = (OUTPUT_LAW_REQUANTIZE, OUTPUT_LAW_IDENTITY)
#: [U-PCH] ``weight_scale_granularity``: one weight scale per perceptron (absent), or one per output neuron with its
#: own threshold register (``mapping/support/neuron_thresholds``, read on a chip that declares them).
WEIGHT_SCALE_GRANULARITY_KEY = "weight_scale_granularity"
WEIGHT_SCALE_PER_PERCEPTRON, WEIGHT_SCALE_PER_NEURON = "per_perceptron", "per_neuron"
WEIGHT_SCALE_GRANULARITIES = (WEIGHT_SCALE_PER_PERCEPTRON, WEIGHT_SCALE_PER_NEURON)


class GridDeclarationError(ToolchainRefusal, ValueError):
    """A grid declaration the program cannot realize or that contradicts another declaration."""


@dataclass(frozen=True)
class ParameterGrids:
    """How the deployed weight and bias grids are chosen; the default is the integer lattice with a gridded bias.

    ``register_bias``: the two-scale projection's on-chip register route, a run fact rather than a declaration (the
    ``wq_two_scale_projection`` key on a platform whose cores carry an on-chip bias,
    ``resolve_wq_two_scale_projection``): the bias is coded on the weight lattice, so the export spans each bias
    register over its own codes (``±q_max r``) instead of saturating it at the weight register."""

    fractional_scale: bool = False
    law_bias: bool = False
    register_bias: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, register_bias: bool = False) -> "ParameterGrids":
        return cls(fractional_scale=config.get("weight_scale_rule") == WEIGHT_SCALE_FRACTIONAL,
                   law_bias=config.get("bias_resolution") == BIAS_AT_LAW, register_bias=bool(register_bias))

    @property
    def declared(self) -> bool:
        return self.fractional_scale or self.law_bias


def identity_output_law(readout_law: Optional[Mapping[str, Any]]) -> bool:
    """Whether ``readout_law`` declares the identity law on output cores."""
    return readout_law is not None and readout_law.get("output_law") == OUTPUT_LAW_IDENTITY


def projection_grids(config: Mapping[str, Any], *, two_scale: bool,
                     on_chip_bias: Callable[[], bool]) -> ParameterGrids:
    """The grids the Weight Quantization projection takes, refused by name where the declarations contradict
    (``on_chip_bias`` is read only under a declared law bias)."""
    grids = ParameterGrids.from_config(config)
    if grids.law_bias and two_scale:
        raise GridDeclarationError(
            "bias_resolution='law' holds the bias in the readout law, but wq_two_scale_projection puts it on its "
            "own integer grid: declare one of the two"
        )
    if grids.law_bias and not on_chip_bias():
        raise GridDeclarationError(
            "bias_resolution='law' needs an on-chip bias (has_bias true): a parameter-encoded bias rides the "
            "weight grid as an always-on axon row"
        )
    return grids
