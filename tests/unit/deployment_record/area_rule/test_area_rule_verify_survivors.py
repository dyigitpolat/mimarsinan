"""[U-AREA] The ten tests the build's verify found missing (UB/verify.md section 5 and mi-2): each names the verify's
mutant that survived the unit's tests, is green on the build's tree and on the fixed tree, and is red on that mutant
(re-pointed where the fix round moved its line: UB2/scripts/mut_r2.py).

A03 and A04: a type rule's declared cell kind, on both term forms; A11: a bound converter kind asked at other bits;
A15: a pooled census under a type rule with a per-core converter ceiling; A24: complete terms beside a gamma outside,
refused at load; A25: a type rule without its node, refused at load; B05: an anchor's converters per core ceiled off a
column multiple; B12: a held-logic anchor whose own share is refused; D04: an anchor that declares no converter kind;
G09: a type-rule census without weight bits.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED, CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import anchor, declaration, pricing, registry, rule
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G

AREA = "chip_area_mm2"


def _census(types, weight_bits):
    """A census of core types ``(count, rows, cols)``."""
    values = {"cores_physical": sum(n for n, _r, _c in types), "axons_physical": sum(n * r for n, r, _c in types),
              "neurons_physical": sum(n * c for n, _r, c in types),
              "cells_physical": sum(n * r * c for n, r, c in types), "weight_bits": weight_bits,
              "tiles": sum(n for n, _r, _c in types)}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items() if v is not None})


def _price(physics, types, weight_bits, geometry="declared", readout_bits=8):
    geo = [CoreGeometry(r, c, n) for n, r, c in types] if geometry == "declared" else geometry
    p = price_absolute(_census(types, weight_bits), physics, geometry=geo, readout_bits=readout_bits)
    term = {t.name: t for t in p.terms}.get(AREA)
    refusal = {r.name: r.reason for r in p.refusals}.get(AREA)
    return term, refusal, {f.kind for f in p.flags if f.name == AREA}


def _type_rule(type_name, **changes):
    physics = get_platform_physics("generic_estimated_22nm")
    return replace(physics, area_rule=replace(physics.area_rule, type=type_name, reading_low_edge=None, **changes))


TYPES = ((1000, 256, 256),)


class TestTheVerifysSurvivors:
    def test_a03_a_bits_type_rule_prices_its_declared_cell_kind(self):
        """A03: a declared cell kind on 'bits' terms is priced, never the type default (k_L times the push-rule
        cell)."""
        declared, _r, _f = _price(_type_rule("T1", cell_kind="sram_6t_logic_rule"), TYPES, 4)
        default, _r2, _f2 = _price(_type_rule("T1"), TYPES, 4)
        k_l = registry.load_registry().value("cell_kinds:kinds.sram_6t_logic_rule.factor_of_sigma")
        assert G.close(declared.band.nominal / default.band.nominal, k_l)

    def test_a04_a_cells_type_rule_prices_its_declared_cell_kind(self, tmp_path, monkeypatch):
        """A04: a declared cell kind on 'cells' terms is taken before the type default: it prices as a registry whose
        type default is that kind."""
        declared, _r, _f = _price(_type_rule("T3", cell_kind="rram_crosspoint"), TYPES, 4)
        shipped, _r2, _f2 = _price(_type_rule("T3"), TYPES, 4)
        data = tmp_path / "data"
        shutil.copytree(registry.DATA, data)
        path = data / "types" / "t3.json"
        doc = json.loads(path.read_text())
        doc["default_cell_kind"] = "rram_crosspoint"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        reg = registry.Registry.load(data)
        monkeypatch.setattr(pricing, "load_registry", lambda: reg)
        as_default, _r3, _f3 = _price(_type_rule("T3"), TYPES, 4)
        assert G.close(declared.band.nominal, as_default.band.nominal)
        assert not G.close(declared.band.nominal, shipped.band.nominal, 1e-3)

    def test_a11_a_bound_kind_asked_at_other_bits_is_its_bound_named_refusal(self):
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.converter_area(registry.load_registry(), "sar_bitcell_reusing", 6)
        assert str(caught.value) == "converter kind sar_bitcell_reusing: its bound holds at 8 bits; 6 requested"
        physics = _type_rule("T2_analog", cell_kind="cim_10t1c_mom",
                             converter={"kind": "sar_bitcell_reusing", "per_core": 64})
        # the fourth delta round (MA-C2): a physics the loader accepts, its converter area stated once (the generic's
        # own area_per_adc dropped beside the block's converter kind)
        physics = replace(physics, constants={k: v for k, v in physics.constants.items() if k != "area_per_adc"})
        term, refusal, _f = _price(physics, TYPES, 4, readout_bits=6)
        assert term is None and refusal == "converter kind sar_bitcell_reusing: its bound holds at 8 bits; 6 requested"

    def test_a15_a_pooled_census_under_a_type_rule_with_a_per_core_ceiling_is_a_named_refusal(self):
        physics = _type_rule("T3", organization={"columns_per_converter": 4})
        term, refusal, _f = _price(physics, ((24, 256, 256), (24, 256, 128)), 4, geometry=UNJUDGED)
        assert term is None and refusal == ("converter count: the census pools core types and the description "
                                            "converts every 4 columns per converter; pass the declared core types")

    def test_a24_complete_terms_beside_a_gamma_outside_are_refused_at_load(self):
        block = replace(get_platform_physics("isaac_like").area_rule, outside="own_die")
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert str(caught.value) == "area_rule: complete terms take their declared parts (outside declared_parts)"

    def test_a25_a_type_rule_without_its_node_is_refused_at_load(self):
        block = replace(get_platform_physics("generic_estimated_22nm").area_rule, node=None)
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert str(caught.value) == "area_rule: a type rule or an anchor names its node (node.node_nm)"

    def test_b05_an_anchors_converters_per_core_are_ceiled_off_a_column_multiple(self):
        org = dict(get_platform_physics("neurram").area_rule.organization, columns_per_converter=4)
        ratio = anchor.anchor_ratio(org, anchor.CensusCore(rows=256, cols=130, weight_bits=4))
        assert ratio.counts["converters"] == math.ceil(130 / 4) and ratio.anchor_counts["converters"] == 256 / 4

    def test_b12_a_held_logic_anchor_whose_own_share_is_refused_refuses_by_name_off_its_geometry(self):
        with pytest.raises(AreaRuleRefusal) as caught:
            anchor.anchor_core_band(0.0936, 0.5, None, logic_held=True)
        assert str(caught.value) == "the anchor holds its logic per core and its own share is refused"
        physics = get_platform_physics("truenorth")
        physics = replace(physics, area_rule=replace(physics.area_rule, node={"node_nm": 4}))
        term, refusal, _f = _price(physics, ((4096, 128, 128),), 1)
        assert term is None and refusal == "the anchor holds its logic per core and its own share is refused"

    def test_d04_an_anchor_that_declares_no_converter_kind_takes_no_type_default(self):
        physics = get_platform_physics("neurram")
        physics = replace(physics, area_rule=replace(physics.area_rule, converter=None))
        term, refusal, flags = _price(physics, ((48, 256, 256),), 4, readout_bits=None)
        assert refusal is None and term is not None, refusal
        assert "anchor_identity" in flags

    def test_g09_a_type_rule_census_without_weight_bits_is_read_at_the_descriptions_cell_bits(self):
        physics = _type_rule("T2_digital")
        withheld, _r, flags = _price(physics, TYPES, None)
        stated, _r2, _f2 = _price(physics, TYPES, physics.validity.geometry.cell_bits)
        assert "weight_bits_not_stated" in flags
        assert G.close(withheld.band.nominal, stated.band.nominal)
