"""The named variant catalog: the specs the cosim gates prove and P8 measures.

ONE list, so the compile-limits study cannot report resources for a geometry no
cosimulation ever proved, and a gate cannot prove a geometry the study never
costs. Every entry is a projection of a declared core type times a resolved
``SomaLaw`` -- the specs are never written down as literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec

#: The generated substrate signs its own synapse cell, which is the OTHER value
#: of the weight-sign axis from the stock crossbar's per-row SPI_SYN_SIGN.
SIGN_GRANULARITY = "per_synapse"

WEIGHT_BITS = 4

_BASE: Mapping[str, Any] = {
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "thresholding_mode": "<=",
}


def per_event_law(membrane_bits: int) -> SomaLaw:
    """The event-serial law on an unsigned register of the declared width."""
    return SomaLaw.resolve({
        **_BASE, "firing_granularity": "per_event", "membrane_bits": membrane_bits,
    })


def sync_fire_law(membrane_bits: int) -> SomaLaw:
    """The sync-fire law: per-cycle compare on a two's-complement register."""
    return SomaLaw.resolve({
        **_BASE, "membrane_bits": membrane_bits, "membrane_signed": True,
    })


def unbounded_law() -> SomaLaw:
    """The contract the sync-fire law claims to reproduce exactly."""
    return SomaLaw.resolve(dict(_BASE))


def spec_for(law: SomaLaw, *, axons: int, neurons: int, count: int = 1) -> CoreSpec:
    """One declared core type times the law -- the projection, never a literal."""
    return CoreSpec.project(
        {"max_axons": axons, "max_neurons": neurons, "count": count,
         "has_bias": False},
        soma_law=law, weight_bits=WEIGHT_BITS,
        weight_sign_granularity=SIGN_GRANULARITY,
    )


@dataclass(frozen=True)
class NamedVariant:
    """One proven variant: its spec, the gate that proved it, and its memories."""

    name: str
    spec: CoreSpec
    proven_by: str

    def memory_shapes(self) -> Tuple[Dict[str, Any], ...]:
        """Every array the generated core declares, as words x width x bits.

        Read straight off the template's declarations (``syn_mem``, ``thr_arr``,
        ``vmem_arr``) through the spec that sized them, so the bit arithmetic in
        the compile-limits study is the RTL's own and not a second derivation.
        """
        spec = self.spec
        return tuple(
            {"array": array, "words": words, "width": width,
             "bits": words * width}
            for array, words, width in (
                ("syn_mem", spec.synapse_depth, 32),
                ("thr_arr", spec.max_neurons, spec.membrane_bits),
                ("vmem_arr", spec.max_neurons, spec.membrane_bits),
            )
        )


#: The three GENERATED variants plan §7 row 20 proves at zero difference.
PROVEN_VARIANTS: Tuple[NamedVariant, ...] = (
    NamedVariant(
        name="gen_a128n128_mb8_per_event",
        spec=spec_for(per_event_law(8), axons=128, neurons=128),
        proven_by="tests/integration/test_odin_gen_geometry.py::small_variant",
    ),
    NamedVariant(
        name="gen_a512n256_mb16_per_event",
        spec=spec_for(per_event_law(16), axons=512, neurons=256),
        proven_by="tests/integration/test_odin_gen_geometry.py::wide_variant",
    ),
    NamedVariant(
        name="gen_a256n256_mb16s_sync_fire",
        spec=spec_for(sync_fire_law(16), axons=256, neurons=256),
        proven_by="tests/integration/test_odin_gen_sync_fire.py",
    ),
)


def variant_named(name: str) -> NamedVariant:
    """The catalog entry of ``name``, refusing one the catalog does not carry."""
    for variant in PROVEN_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(
        f"{name!r} is not a proven variant; the catalog carries "
        f"{', '.join(v.name for v in PROVEN_VARIANTS)}")
