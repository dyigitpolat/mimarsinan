"""A profile's converter over resolution (E-5, W34): the converter constants at the run's ``readout_law.adc_bits``.

A physics profile states one converter at its own resolution in its flat constants (``e_adc_conversion``,
``t_adc_conversion``, ``area_per_adc``). It may also state ``converter``: a table of rows over resolution, each row a
``bits`` with any of the three constants (each with its evidence, as a profile constant), and for the energy a cited
scaling law anchored at one row (``energy_law``):

    E(b) = E(a) x (k1 b + k2 4^b) / (k1 a + k2 4^a),   a the anchor row's bits, b within the law's ``bits_range``,

the form of the ADC energy model houshmand2023benchmarking Eq. 8 reads (E_ADC per conversion = (k1 ADCres + k2
4^ADCres) V^2, k1 = 100 fJ, k2 = 1 aJ, after its reference [5]); anchored at a measured row the supply term cancels.

``resolve`` gives the physics a run prices at its declared adc_bits: the row's constants where the row states them, the
law's energy within its range, and for every other converter constant a refusal by name (the pricer refuses each
headline whose converter work exists, ``cost/absolute``). A profile without a converter, or a run that declares no
adc_bits, prices its flat constants unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.declaration import judge_beside
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration  # [U-E3] below
from mimarsinan.deployment_record.units import to_canonical, unit_for

#: The converter constants a row may state (the flat constants' keys).
CONVERTER_KEYS = ("e_adc_conversion", "t_adc_conversion", "area_per_adc")
LAW_FORMS = ("linear_plus_exponential",)


class ConverterDeclarationError(ToolchainRefusal, ValueError):
    """A converter declaration the loader refuses: an unknown constant, a resolution stated twice, a law without its
    anchor row's energy, a bit count that is no positive integer."""


def _energy_j(data: Mapping[str, Any], name: str) -> float:
    if not isinstance(data, Mapping) or "nominal" not in data or "unit" not in data:
        raise ConverterDeclarationError(f"converter energy_law {name} needs a nominal and an energy unit")
    if unit_for(str(data["unit"])).dimension != "energy":
        raise ConverterDeclarationError(f"converter energy_law {name}: {data['unit']!r} is no energy unit")
    value = to_canonical(float(data["nominal"]), str(data["unit"]))
    if not value > 0.0:
        raise ConverterDeclarationError(f"converter energy_law {name} must be positive")
    return value


@dataclass(frozen=True)
class EnergyLaw:
    """E(b) = E(anchor) x (k1 b + k2 4^b) / (k1 anchor + k2 4^anchor), b in ``bits_range`` (k1, k2 in joules)."""

    form: str
    anchor_bits: int
    bits_range: Tuple[int, int]
    k1_j: float
    k2_j: float
    citation: str
    derivation: str
    source: Mapping[str, Any] = field(default_factory=dict)

    def shape(self, bits: int) -> float:
        return self.k1_j * bits + self.k2_j * 4.0 ** bits

    def covers(self, bits: int) -> bool:
        return self.bits_range[0] <= bits <= self.bits_range[1]


@dataclass(frozen=True)
class ConverterTable:
    """Rows of converter constants over resolution and an optional energy law."""

    rows: Mapping[int, Mapping[str, Any]]
    energy_law: Optional[EnergyLaw] = None
    source: Mapping[str, Any] = field(default_factory=dict)

    @property
    def bits(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rows))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConverterTable) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:  # pragma: no cover - frozen dataclasses of mappings are not hashed in use
        return hash(tuple(self.bits))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConverterTable":
        if not isinstance(data, Mapping):
            raise ConverterDeclarationError("converter must be an object of rows and an optional energy_law")
        unknown = sorted(set(data) - {"rows", "energy_law", "note"})
        if unknown:
            raise ConverterDeclarationError(f"converter declares unknown fields {unknown}")
        rows: Dict[int, Dict[str, Any]] = {}
        for raw in data.get("rows") or ():
            bits = raw.get("bits")
            if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
                raise ConverterDeclarationError(f"converter row bits {bits!r} is no positive integer")
            if bits in rows:
                raise ConverterDeclarationError(f"converter states the resolution {bits} bits twice")
            stray = sorted(k for k in raw if k not in ("bits",) + CONVERTER_KEYS)
            if stray:
                raise ConverterDeclarationError(f"converter row at {bits} bits states {stray}, outside "
                                                f"{list(CONVERTER_KEYS)}")
            rows[bits] = {k: constant_from_declaration(k, raw[k]) for k in CONVERTER_KEYS if k in raw}
        if not rows:
            raise ConverterDeclarationError("converter states no row")
        law = None
        if data.get("energy_law") is not None:
            raw = data["energy_law"]
            form = raw.get("form")
            if form not in LAW_FORMS:
                raise ConverterDeclarationError(f"converter energy_law form {form!r} is not one of {LAW_FORMS}")
            anchor = raw.get("anchor_bits")
            if anchor not in rows or "e_adc_conversion" not in rows[anchor]:
                raise ConverterDeclarationError(f"converter energy_law anchor_bits {anchor!r} is no row that states "
                                                "e_adc_conversion")
            lo, hi = (int(v) for v in raw.get("bits_range") or (0, -1))
            if not 1 <= lo <= anchor <= hi:
                raise ConverterDeclarationError(f"converter energy_law bits_range {raw.get('bits_range')!r} must "
                                                "hold its anchor and start at 1 or above")
            if not raw.get("citation"):
                raise ConverterDeclarationError("converter energy_law needs its citation")
            law = EnergyLaw(form=str(form), anchor_bits=int(anchor), bits_range=(lo, hi),
                            k1_j=_energy_j(raw.get("k1"), "k1"), k2_j=_energy_j(raw.get("k2"), "k2"),
                            citation=str(raw["citation"]), derivation=str(raw.get("derivation", "")),
                            source=dict(raw))
        return cls(rows=rows, energy_law=law, source=dict(data))


def declared_adc_bits(platform: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The run's declared converter resolution, ``readout_law.adc_bits``, or None when the run declares none."""
    law = (platform or {}).get("readout_law")
    bits = law.get("adc_bits") if isinstance(law, Mapping) else None
    return None if bits is None or isinstance(bits, bool) else int(bits)


def with_converter(physics: Any, declaration: Optional[Mapping[str, Any]]) -> Any:
    """The physics with ``declaration`` as its converter (None removes it), judged as a profile's: [U-AREA N-6] a
    table row's area_per_adc beside an area rule block's converter kind is the converter area stated twice."""
    table = None if declaration is None else ConverterTable.from_dict(declaration)
    block = getattr(physics, "area_rule", None)
    if block is not None and table is not None:
        judge_beside(block, converter_table=any("area_per_adc" in row for row in table.rows.values()))
    return replace(physics, converter=table)


def resolve(physics: Any, bits: Optional[int]) -> Tuple[Any, Dict[str, str]]:
    """(the physics priced at ``bits``, {converter constant: why it is refused at ``bits``}): unchanged without a
    converter or without a declared resolution."""
    table = getattr(physics, "converter", None)
    if table is None or bits is None:
        return physics, {}
    bits = int(bits)
    row = table.rows.get(bits, {})
    law = table.energy_law
    constants = dict(physics.constants)
    refused: Dict[str, str] = {}
    stated = ", ".join(str(b) for b in table.bits)
    for key in CONVERTER_KEYS:
        if key in row:
            constants[key] = replace(row[key], key=key)
            continue
        constants.pop(key, None)
        if key == "e_adc_conversion" and law is not None and law.covers(bits):
            anchor = table.rows[law.anchor_bits]["e_adc_conversion"]
            ratio = law.shape(bits) / law.shape(law.anchor_bits)
            constants[key] = replace(
                anchor, key=key, low=anchor.low * ratio, nominal=anchor.nominal * ratio, high=anchor.high * ratio,
                evidence_kind="derived", citation=f"{anchor.citation}; {law.citation}",
                derivation=(f"the {law.anchor_bits}-bit row scaled to {bits} bits by (k1 b + k2 4^b) / (k1 a + k2 4^a) "
                            f"= {ratio:.6g} ({law.citation})" + (f"; {law.derivation}" if law.derivation else "")))
            continue
        span = "" if law is None else (f"; its energy law covers {law.bits_range[0]} to {law.bits_range[1]} bits"
                                       if key == "e_adc_conversion" else "; its energy law prices energy only")
        refused[key] = (f"{key} at readout_law.adc_bits {bits}: the converter table states ({stated}) bits{span}")
    return replace(physics, constants=constants), refused
