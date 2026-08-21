"""``CoreSpec``: one declared core type times one resolved soma law, projected."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    PER_AXON_SIGN,
    PER_SYNAPSE_SIGN,
    physical_row_expansion,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import NOVENA_FIRING_MODE

#: The CORE_FIELDS names, so a spec is READ from a declared core type and can
#: never carry a geometry the platform never declared.
CORE_TYPE_FIELDS: Tuple[str, ...] = ("max_axons", "max_neurons", "count", "has_bias")

#: Geometry the v1 generator emits: powers of two, because the address widths,
#: the synapse-word packing and the sweep bounds are all derived from clog2 and
#: a ragged geometry buys nothing a padded power of two does not.
GENERATED_AXON_CHOICES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
GENERATED_NEURON_CHOICES: Tuple[int, ...] = GENERATED_AXON_CHOICES

#: Membrane register widths the v1 generator emits. Theta shares the width.
GENERATED_MEMBRANE_BITS: Tuple[int, ...] = (8, 16)

#: Synapse cell widths that tile a 32-bit programming word exactly.
GENERATED_WEIGHT_BITS: Tuple[int, ...] = (2, 4, 8, 16)

PROGRAM_WORD_BITS = 32


class CoreSpecError(ValueError):
    """A declared core type + soma law is not a core this generator emits."""


@dataclass(frozen=True)
class CoreSpec:
    """The generated core, stated once: geometry, weight width, and the law.

    A PROJECTION, never an independent declaration — ``project`` reads the
    ``CORE_FIELDS`` of one declared core type and the resolved ``SomaLaw``, so
    the RTL, the packer tables and the descriptor cannot drift from the platform
    the deployment was mapped against.
    """

    max_axons: int
    max_neurons: int
    count: int
    has_bias: bool
    soma_law: SomaLaw
    weight_bits: int
    weight_sign_granularity: str

    @classmethod
    def project(
        cls,
        core_type: Mapping[str, Any],
        *,
        soma_law: SomaLaw,
        weight_bits: int,
        weight_sign_granularity: str,
    ) -> "CoreSpec":
        """THE constructor: one core type of a declared platform, times the law."""
        missing = [f for f in ("max_axons", "max_neurons") if f not in core_type]
        if missing:
            raise CoreSpecError(
                f"a core type must declare {', '.join(missing)}; got "
                f"{dict(core_type)!r}. The spec is a projection of the declared "
                f"platform and has nothing to fall back on.")
        return cls(
            max_axons=int(core_type["max_axons"]),
            max_neurons=int(core_type["max_neurons"]),
            count=int(core_type.get("count", 1)),
            has_bias=bool(core_type.get("has_bias", True)),
            soma_law=soma_law,
            weight_bits=int(weight_bits),
            weight_sign_granularity=str(weight_sign_granularity),
        )

    def core_type(self) -> Dict[str, Any]:
        """The declared core type this spec projects — the inverse direction."""
        return {
            "max_axons": self.max_axons, "max_neurons": self.max_neurons,
            "count": self.count, "has_bias": self.has_bias,
        }

    @property
    def membrane_bits(self) -> int:
        return int(self.soma_law.membrane_bits)

    @property
    def membrane_signed(self) -> bool:
        return self.soma_law.is_signed_membrane

    @property
    def per_event(self) -> bool:
        return self.soma_law.is_per_event

    @property
    def reset_zero(self) -> bool:
        return self.soma_law.firing_mode == NOVENA_FIRING_MODE

    @property
    def compare_inclusive(self) -> bool:
        return self.soma_law.thresholding_mode == "<="

    @property
    def asserts_no_saturation(self) -> bool:
        return self.soma_law.asserts_no_saturation

    @property
    def membrane_low(self) -> int:
        bounds = self.soma_law.membrane_bounds
        if bounds is None:
            raise CoreSpecError(
                "the generated core IS a fixed-width register, so an unbounded "
                "membrane has no interval to emit. Declare membrane_bits.")
        return int(bounds[0])

    @property
    def membrane_high(self) -> int:
        bounds = self.soma_law.membrane_bounds
        if bounds is None:
            raise CoreSpecError(
                "the generated core IS a fixed-width register, so an unbounded "
                "membrane has no interval to emit. Declare membrane_bits.")
        return int(bounds[1])

    @property
    def physical_row_factor(self) -> int:
        return physical_row_expansion(self.weight_sign_granularity)

    @property
    def cells_per_word(self) -> int:
        return PROGRAM_WORD_BITS // self.weight_bits

    @property
    def synapse_words_per_row(self) -> int:
        return self.max_neurons // self.cells_per_word

    @property
    def synapse_depth(self) -> int:
        return self.max_axons * self.synapse_words_per_row

    @property
    def axon_address_bits(self) -> int:
        return _log2(self.max_axons)

    @property
    def neuron_address_bits(self) -> int:
        return _log2(self.max_neurons)

    @property
    def synapse_address_bits(self) -> int:
        return max(1, _bits_for(self.synapse_depth - 1))

    def spec_key(self) -> str:
        """A short stable name: the geometry, the register, and the law."""
        law = "pe" if self.per_event else "pc"
        register = f"{self.membrane_bits}{'s' if self.membrane_signed else 'u'}"
        return (f"a{self.max_axons}n{self.max_neurons}"
                f"w{self.weight_bits}m{register}_{law}")


def _log2(value: int) -> int:
    return int(value).bit_length() - 1


def _bits_for(value: int) -> int:
    return max(1, int(value).bit_length())


def require_generatable(spec: CoreSpec) -> CoreSpec:
    """Refuse a spec this generator cannot emit, naming the declaration at fault."""
    _require_choice("max_axons", spec.max_axons, GENERATED_AXON_CHOICES)
    _require_choice("max_neurons", spec.max_neurons, GENERATED_NEURON_CHOICES)
    _require_choice("membrane_bits", spec.membrane_bits, GENERATED_MEMBRANE_BITS)
    _require_choice("weight_bits", spec.weight_bits, GENERATED_WEIGHT_BITS)
    if spec.count < 1:
        raise CoreSpecError(
            f"a core type must declare a positive population, got count="
            f"{spec.count}")
    if spec.has_bias:
        raise CoreSpecError(
            "the generated core has no on-chip bias lane: the bias is delivered "
            "as an always-on axon ROW at the tail of the canonical slot order "
            "(SomaLaw.bias_slot), which is the param-encoded bias mode. Declare "
            "has_bias=false on the core grid, or deploy a target that has a lane.")
    if spec.weight_sign_granularity != PER_SYNAPSE_SIGN:
        raise CoreSpecError(
            f"the generated synapse cell holds a two's-complement weight, so it "
            f"signs itself: weight_sign_granularity="
            f"{spec.weight_sign_granularity!r} costs {spec.physical_row_factor} "
            f"physical rows per logical slot and is the STOCK crossbar's layout "
            f"({PER_AXON_SIGN!r}, signed once per row by SPI_SYN_SIGN). Deploy "
            f"the vendored stock core for that layout, or declare "
            f"{PER_SYNAPSE_SIGN!r}.")
    if not spec.soma_law.saturates:
        raise CoreSpecError(
            f"the generated core IS a fixed-width membrane register; the law "
            f"declares membrane_arithmetic="
            f"{spec.soma_law.membrane_arithmetic!r} with no register to emit. "
            f"Declare membrane_bits on the platform.")
    if spec.per_event and spec.membrane_signed:
        raise CoreSpecError(
            "the event-serial law is refused on a two's-complement register: "
            "the row-pair realization's zero-magnitude member is a no-op only "
            "against a register that floors at zero. Declare "
            "membrane_signed=false, or generate the per_cycle sync-fire core.")
    if spec.per_event and spec.soma_law.firing_mode != spec.soma_law.required_firing_mode:
        raise CoreSpecError(
            f"firing_granularity='per_event' requires firing_mode="
            f"{spec.soma_law.required_firing_mode!r} (the hard-zero reset); the "
            f"law declares {spec.soma_law.firing_mode!r}.")
    if spec.cells_per_word < 2:
        raise CoreSpecError(
            f"weight_bits={spec.weight_bits} leaves {spec.cells_per_word} cell(s) "
            f"per {PROGRAM_WORD_BITS}-bit programming word; the packer addresses "
            f"cells within a word and needs at least two.")
    if spec.max_neurons % spec.cells_per_word:
        raise CoreSpecError(
            f"max_neurons={spec.max_neurons} does not tile "
            f"{spec.cells_per_word} cells per programming word exactly.")
    if spec.weight_bits > spec.membrane_bits:
        raise CoreSpecError(
            f"weight_bits={spec.weight_bits} is wider than the "
            f"{spec.membrane_bits}-bit membrane it charges; a single event would "
            f"be unrepresentable in the register it lands in.")
    return spec


def _require_choice(name: str, value: int, choices: Tuple[int, ...]) -> None:
    if value not in choices:
        raise CoreSpecError(
            f"{name}={value} is not a geometry this generator emits; v1 emits "
            f"{', '.join(str(choice) for choice in choices)}. The address "
            f"widths, the synapse-word packing and the sweep bounds are all "
            f"derived from the declared value, so an unlisted one is a real "
            f"gap, not a rounding.")
