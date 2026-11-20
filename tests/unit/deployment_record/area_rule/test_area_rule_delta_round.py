"""[U-AREA] The delta round (UB2/reverify.md: MA-R1, mi-R1, mi-R2, mi-R3; the class 'a core-area term computed outside
the one core rule', swept), tests first: each test is red on UB2's tree (UB2/uarea.patch on R5, applied digest
1e91a861) and on its named mutant (UB3/scripts/mut_r3.py), green on the delta round's tree; a guard (green on UB2's
tree too) is named as such.

- MA-R1: an anchor's own memory share is the type rule's memory term of the anchor's own organization (the one core
  equation, ``core.core_terms``: the same count form and the same cell resolution), so a declared cell kind is honoured
  and a compute-in-memory anchor's share uses its own cell; an organization that cannot state its count, or counts on
  other terms than its type's, is a named refusal at load and at the pricer; every anchor basis with every type
  prices or is a named refusal through price_absolute, never another exception.
- The class swept: one count form (``core.scaled_counts``) for a type rule's census, an anchor's ratio (census and own
  core), its wider-word reading and its own memory share.
- mi-R3: a memory beside a core, a converter refusal's array part and a row's eta come from the one core equation.
- mi-R1: a memoized refusal is raised fresh each time: a refused price priced 1,000 times holds its memory.
- mi-R2: a cited cell of no positive area is a named refusal at load and at the pricer.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import tracemalloc
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import (
    anchor, declaration, pricing, registry, rule, typerule)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut
from mimarsinan.deployment_record.platform_physics.area_rule.published import parity
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G
from unit.deployment_record.area_rule._physics import priced_physics

AREA = "chip_area_mm2"
#: The one core equation's module (absent on UB2's tree: the structural tests are red there by its absence).
CORE = "mimarsinan.deployment_record.platform_physics.area_rule.core.equation"
ANCHORS = ("truenorth", "loihi", "odin", "hermes_64", "neurram", "sram_cim_macro")
#: The shipped anchors' own shares on UB2's tree (scripts/c02_shares.py, out/shares_bt2.json): the values must not move.
SHIPPED_SHARES = {"truenorth": 0.19843310017163815, "loihi": 0.2858860721746701, "odin": 0.6069718358191285,
                  "hermes_64": 0.65625, "neurram": 0.4557228752555465, "sram_cim_macro": 0.3246069455016809}
BASES = ("published_split", "own_memory_share", "published_split without its expression", "an unknown basis",
         "no own share")
READOUT_BITS = 8


def _types():
    """The technology types, read from the type tables' own 'type' fields (data, never a list in a test)."""
    return sorted(json.loads(p.read_text())["type"] for p in (registry.DATA / "types").glob("*.json"))


def _scaled_terms(type_name):
    reg = registry.load_registry()
    return reg.type_table(type_name)["scaled_terms"]


def _census(cores, rows, cols, weight_bits):
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": weight_bits, "tiles": cores}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _area(physics, cores, rows, cols, weight_bits):
    """(nominal or None, the named refusal or None, flag kinds) of chip_area_mm2 through price_absolute."""
    priced = price_absolute(_census(cores, rows, cols, weight_bits), physics,
                            geometry=[CoreGeometry(rows, cols, cores)], readout_bits=READOUT_BITS)
    term = {t.name: t for t in priced.terms}.get(AREA)
    refusal = {r.name: r.reason for r in priced.refusals}.get(AREA)
    kinds = {f.kind for f in priced.flags if f.name == AREA}
    return (None if term is None else term.band.nominal), refusal, kinds


def _own_memory(block):
    return replace(block, anchor=dict(block.anchor, own_share={"basis": "own_memory_share"}))


def _spec(basis, type_name):
    reg = registry.load_registry()
    if basis == "published_split":
        return {"basis": basis, "expression": f"r('{reg.type_file(type_name)}:s')"}
    if basis == "published_split without its expression":
        return {"basis": "published_split"}
    if basis == "an unknown basis":
        return {"basis": "a_basis_no_rule_states"}
    return None if basis == "no own share" else {"basis": basis}


# ---------------------------------------------------------------------------------------------------------- MA-R1
class TestOwnShareThroughTheOneCoreRuleMAR1:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_every_anchor_basis_with_every_type_through_the_pricer_is_a_price_or_a_named_refusal(self, name):
        """Red on UB2's tree: an own memory share over cells terms raises KeyError 'stored_bits_per_cell', a type
        without eta KeyError 'eta', a published split without its expression KeyError 'expression'."""
        physics = priced_physics(name)
        org = physics.area_rule.organization
        crashes, priced = [], 0
        for type_name in _types():
            for basis in BASES:
                anchor_ = {k: v for k, v in physics.area_rule.anchor.items() if k != "own_share"}
                if _spec(basis, type_name) is not None:
                    anchor_["own_share"] = _spec(basis, type_name)
                p = replace(physics, area_rule=replace(physics.area_rule, type=type_name, anchor=anchor_))
                for rows, cols in ((org["rows"], org["cols"]), (org["rows"] // 2, org["cols"])):
                    try:
                        nominal, refusal, _k = _area(p, 4, rows, cols, org["weight_bits_per_cell"])
                    except Exception as caught:  # noqa: BLE001 - the class test records any other exception
                        crashes.append(f"{type_name} / {basis} / {rows}x{cols}: {type(caught).__name__} {caught}")
                        continue
                    assert (nominal is None) != (refusal is None), (type_name, basis, nominal, refusal)
                    if refusal is not None:
                        assert refusal.strip(), (type_name, basis)
                    else:
                        assert math.isfinite(nominal) and nominal > 0.0, (type_name, basis, nominal)
                        priced += 1
        assert not crashes, crashes
        assert priced > 0

    def test_a_declared_cell_kind_moves_an_own_memory_share_anchors_share_by_its_factor(self):
        """Red on UB2's tree: the kind is accepted and ignored (the share does not move)."""
        reg = registry.load_registry()
        block = get_platform_physics("loihi").area_rule
        base = anchor.anchor_law(reg, block).own_share.value
        kinded = anchor.anchor_law(reg, replace(block, cell_kind="sram_6t_logic_rule")).own_share.value
        factor = (rule.cell_area(reg, "sram_6t_logic_rule", block.node_nm, declared=True)
                  / rule.cell_area(reg, "sram_6t_push_rule", block.node_nm))
        assert factor > 1.5
        assert G.close(kinded / base, factor), (kinded, base, factor)

    def test_a_compute_in_memory_anchors_own_share_uses_its_own_cell(self):
        """Red on UB2's tree (0.1722, the push-rule cell): the macro's own memory share is its bits on its own
        compute cell over eta, which is the T2 digital share its published split reads (calibrated on that macro)."""
        reg = registry.load_registry()
        block = get_platform_physics("sram_cim_macro").area_rule
        law = anchor.anchor_law(reg, _own_memory(block))
        org = block.organization
        bits = org["rows"] * org["cols"] * org["stored_bits_per_cell"] + org["cols"] * org["bits_per_neuron"]
        cell = rule.cell_area(reg, block.cell_kind, block.node_nm, declared=True)
        eta = reg.value(f"{reg.type_file(block.type)}:eta")
        assert G.close(law.own_share.value, bits * cell / eta / (law.core_mm2 * 1e6)), law.own_share
        assert G.close(law.own_share.value, anchor.anchor_law(reg, block).own_share.value)

    @pytest.mark.parametrize("name", ANCHORS)
    def test_an_anchors_own_share_is_the_type_rules_memory_term_of_its_own_organization(self, name):
        """Red on UB2's tree for TrueNorth, ODIN and the macro (another count form or cell) and for NeuRRAM and
        HERMES (KeyError); Loihi agrees there (stored bits per cell equal the weight bits per cell, push-rule cell)."""
        reg = registry.load_registry()
        block = get_platform_physics(name).area_rule
        law = anchor.anchor_law(reg, _own_memory(block))
        org = block.organization
        census = rule.CensusCore(org["rows"], org["cols"], org["weight_bits_per_cell"])
        declared = typerule.declared_of(reg, block)
        counts = typerule.census_counts(reg, block.type, declared, org, census, census.weight_bits)
        memory = typerule.core_terms(reg, block.type, declared, typerule.Counts(counts.array)).array_um2
        assert G.close(law.own_share.value, memory / (law.core_mm2 * 1e6)), (law.own_share, memory)

    @pytest.mark.parametrize("name", ANCHORS)
    def test_the_shipped_anchors_shares_do_not_move(self, name):
        """A guard (UB2's own values, to 1e-12): the round moves no shipped anchor's share."""
        law = anchor.anchor_law(registry.load_registry(), get_platform_physics(name).area_rule)
        assert G.close(law.own_share.value, SHIPPED_SHARES[name]), law.own_share

    def test_an_organization_that_cannot_state_its_count_is_a_named_refusal_at_load_and_at_the_pricer(self):
        """Red on UB2's tree: accepted at load, KeyError 'stored_bits_per_cell' out of price_absolute."""
        physics = get_platform_physics("loihi")
        org = {k: v for k, v in physics.area_rule.organization.items() if k != "stored_bits_per_cell"}
        block = replace(physics.area_rule, organization=org)
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert "stored_bits_per_cell" in str(caught.value)
        nominal, refusal, _k = _area(replace(physics, area_rule=block), 128, 64, 512, 8)
        assert nominal is None and "stored_bits_per_cell" in refusal, refusal

    def test_an_organization_on_other_terms_than_its_types_is_a_named_refusal_at_load_and_at_the_pricer(self):
        """Red on UB2's tree: accepted at load and priced (or a KeyError) on the organization's own terms."""
        physics = get_platform_physics("loihi")
        other = next(t for t in _types() if _scaled_terms(t) != physics.area_rule.organization["terms"])
        block = replace(physics.area_rule, type=other)
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert "scaled terms" in str(caught.value)
        nominal, refusal, _k = _area(replace(physics, area_rule=block), 128, 64, 512, 8)
        assert nominal is None and "scaled terms" in refusal, refusal

    @pytest.mark.parametrize("change", [{"term_weights": {"cells": 1.0}}, {"term_weights": {"bits": 0.0}},
                                        {"term_weights": {"bits": "1"}}, {"term_weights": {}},
                                        {"stored_bits_per_cell": 0, "bits_per_neuron": 0.0}])
    def test_an_organization_whose_term_weights_miss_its_types_terms_is_a_named_refusal(self, change):
        """Red on UB2's tree: a weight on a term the type does not count raised KeyError out of price_absolute, a zero
        sum or a zero own count ZeroDivisionError, a text weight TypeError, and every one was accepted at load."""
        physics = get_platform_physics("loihi")
        block = replace(physics.area_rule, organization=dict(physics.area_rule.organization, **change))
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert "term_weights" in str(caught.value)
        nominal, refusal, _k = _area(replace(physics, area_rule=block), 128, 64, 512, 8)
        assert nominal is None and "term_weights" in refusal, refusal

    @pytest.mark.parametrize("share", [None, {"basis": "a_basis_no_rule_states"}, {"basis": "published_split"}])
    def test_an_anchors_own_share_states_a_known_basis_at_load(self, share):
        """Red on UB2's tree: an anchor without its own share, with an unknown basis or with a published split without
        its expression was accepted at load (and refused or crashed only at the price)."""
        physics = get_platform_physics("loihi")
        anchor_ = {k: v for k, v in physics.area_rule.anchor.items() if k != "own_share"}
        if share is not None:
            anchor_["own_share"] = share
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(replace(physics.area_rule, anchor=anchor_), registry.load_registry())
        assert "own_share" in str(caught.value)

    def test_the_own_memory_share_is_priced_by_the_one_core_equation(self, monkeypatch):
        """Red on UB2's tree (no core module) and on a share computed beside it: the anchor's own memory share calls
        core.core_terms with the anchor's type, its declaration and its own memory count."""
        core = importlib.import_module(CORE)
        calls, real = [], core.core_terms
        monkeypatch.setattr(core, "core_terms", lambda reg, t, d, c, *a, **k: calls.append((t, d, c)) or real(
            reg, t, d, c, *a, **k))
        reg = registry.load_registry()
        block = get_platform_physics("odin").area_rule
        # the fourth delta round: an anchor's law is made once per frozen block (core/door.remembered); forgotten, it
        # is made again here
        importlib.import_module("mimarsinan.deployment_record.platform_physics.area_rule.core.door").forget(reg)
        law = anchor.anchor_law(reg, block)
        org = block.organization
        own = org["rows"] * org["cols"] * org["stored_bits_per_cell"] + org["cols"] * org["bits_per_neuron"]
        assert [(t, c.array, c.converters) for t, _d, c in calls] == [(block.type, own, 0.0)]
        assert calls[0][1] == typerule.declared_of(reg, block)
        assert G.close(law.own_share.value, SHIPPED_SHARES["odin"])


# ---------------------------------------------------------------------------------------------------------- the class
class TestOneCountFormTheClassSwept:
    CENSUSES = (rule.CensusCore(256, 256, 4), rule.CensusCore(128, 512, 8), rule.CensusCore(64, 64, 1),
                rule.CensusCore(256, 192, 12), rule.CensusCore(170.5, 96.25, 4, pooled=True))

    @pytest.mark.parametrize("name", ANCHORS)
    def test_the_type_rule_and_the_anchor_count_one_form(self, name):
        """Red on UB2's tree for the anchors on bits terms: the type rule counted a w-bit weight as w bits whatever
        cell the organization declares (TrueNorth's 1-bit cells, ODIN's 3-bit cells of 4 stored bits)."""
        reg = registry.load_registry()
        block = get_platform_physics(name).area_rule
        org = block.organization
        declared = typerule.declared_of(reg, block)
        for census in self.CENSUSES:
            ratio = anchor.anchor_ratio(org, census)
            counts = typerule.census_counts(reg, block.type, declared, org, census, census.weight_bits)
            term = "bits" if "bits" in ratio.counts else "cells"
            assert counts.array == ratio.counts[term], (census, counts, ratio.counts)
            if "converters" in ratio.counts and counts.converters:
                assert counts.converters == ratio.counts["converters"], (census, counts, ratio.counts)

    def test_a_type_rule_reads_a_declared_cell_organization_as_an_anchor_does(self):
        """Red on UB2's tree: a type rule that declares ODIN's cells (3 weight bits, 4 stored) counted 4 bits per
        4-bit weight; the one count form counts two cells of 4 stored bits."""
        reg = registry.load_registry()
        odin = get_platform_physics("odin").area_rule
        org = {k: odin.organization[k] for k in ("weight_bits_per_cell", "stored_bits_per_cell", "bits_per_neuron")}
        census = rule.CensusCore(256, 256, 4)
        declared = typerule.declared_of(reg, odin)
        counts = typerule.census_counts(reg, odin.type, declared, org, census, 4)
        assert counts.array == 256 * 256 * 4 * 2 + 256 * odin.organization["bits_per_neuron"]

    def test_every_count_reads_the_one_count_form(self, monkeypatch):
        """Red on UB2's tree (no core module) and on any count restated beside it: a type rule's census, an anchor's
        census and own core, its wider-word reading and its own memory share call core.scaled_counts."""
        core = importlib.import_module(CORE)
        calls, real = [], core.scaled_counts
        monkeypatch.setattr(core, "scaled_counts", lambda *a, **k: calls.append(a[2:5]) or real(*a, **k))
        reg = registry.load_registry()
        odin = get_platform_physics("odin").area_rule
        org, census = odin.organization, rule.CensusCore(128, 256, 4)
        own = (org["rows"], org["cols"], org["weight_bits_per_cell"])
        anchor.anchor_ratio(org, census)
        assert (128, 256, 4) in calls and own in calls
        calls.clear()
        assert anchor.wider_word_ratio(org, census) is not None
        assert (128, 256, 4) in calls and own in calls
        calls.clear()
        typerule.census_counts(reg, odin.type, typerule.declared_of(reg, odin), org, census, 4)
        assert calls == [(128, 256, 4)]
        calls.clear()
        # the fourth delta round: an anchor's law is made once per frozen block (core/door.remembered); forgotten, it
        # is made again here
        importlib.import_module("mimarsinan.deployment_record.platform_physics.area_rule.core.door").forget(reg)
        anchor.anchor_law(reg, odin)
        assert own in calls


# ---------------------------------------------------------------------------------------------------------- mi-R3
class TestInlineTermsThroughTheOneRuleMiR3:
    def test_a_memory_beside_a_core_is_the_named_types_core_equation(self, monkeypatch):
        """Red on UB2's tree (no core module) and on the inline memory term: the memory a point states beside its
        core is the array part of its named type's own core equation."""
        core = importlib.import_module(CORE)
        reg = registry.Registry.load()
        points = [p for p in reg.published_points() if (p.get("organization") or {}).get("memory_beside")]
        assert points
        calls, real = [], core.core_terms
        monkeypatch.setattr(core, "core_terms", lambda r, t, d, c, *a, **k: calls.append((t, c)) or real(
            r, t, d, c, *a, **k))
        parity.estimate(reg, "in_sample")
        for point in points:
            beside = point["organization"]["memory_beside"]
            bits = reg.evaluate(beside["bits"])
            assert any(t == beside["type"] and c.array == bits and not c.converters and c.beside is None
                       for t, c in calls), (point["name"], beside)

    def test_a_converter_refusals_array_part_is_the_one_core_equations(self, monkeypatch):
        """Red on UB2's tree (no core module) and on the inline array part: a row refused by converter kind carries
        the array part of the one core equation (scaled with it)."""
        core = importlib.import_module(CORE)
        before = [r for r in parity.estimate(registry.Registry.load(), "held_out")
                  if r.refused_by == "converter" and r.array_um2 is not None]
        assert before
        real = core.core_area
        monkeypatch.setattr(core, "core_area", lambda *a, **k: (lambda t: t._replace(array_um2=3.0 * t.array_um2))(
            real(*a, **k)))
        after = {(r.chip, r.scope): r for r in parity.estimate(registry.Registry.load(), "held_out")}
        for row in before:
            assert G.close(after[(row.chip, row.scope)].array_um2, 3.0 * row.array_um2), row.chip

    def test_a_rows_eta_is_the_one_core_equations_eta(self, monkeypatch):
        """Red on UB2's tree (no core module) and on the eta restated in the estimate: a row's eta is the eta its
        core equation divides by (the decomposition reads it)."""
        core = importlib.import_module(CORE)
        before = {(r.chip, r.scope): r for r in parity.estimate(registry.Registry.load(), "in_sample")}
        real = core.type_eta
        monkeypatch.setattr(core, "type_eta", lambda reg, type_name: 2.0 * real(reg, type_name))
        after = parity.estimate(registry.Registry.load(), "in_sample")
        checked = 0
        for row in after:
            was = before[(row.chip, row.scope)]
            if row.refused is None and was.refused is None:
                assert G.close(row.eta, 2.0 * was.eta), row.chip
                assert G.close(row.array_um2, was.array_um2 / 2.0), row.chip
                checked += 1
        assert checked


# ---------------------------------------------------------------------------------------------------------- mi-R1
class TestMemoizedRefusalRaisedFreshMiR1:
    def test_a_refused_price_priced_a_thousand_times_holds_its_memory(self):
        """Red on UB2's tree: the memo re-raised one stored exception, whose traceback grew by every call (about
        7 KiB per refused call)."""
        physics = get_platform_physics("generic_estimated_22nm")
        physics = replace(physics, area_rule=replace(physics.area_rule, node={"node_nm": 3}))
        quantities, geometry = _census(1000, 256, 256, 4), [CoreGeometry(256, 256, 1000)]
        first = pricing.rule_area(physics, quantities, geometry, READOUT_BITS)
        assert first.refusal and "sigma(3 nm)" in first.refusal
        for _ in range(100):
            pricing.rule_area(physics, quantities, geometry, READOUT_BITS)
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for _ in range(1000):
                assert pricing.rule_area(physics, quantities, geometry, READOUT_BITS).refusal == first.refusal
            gc.collect()
            grown = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()
        assert grown < 256 * 1024, grown

    def test_a_memoized_refusal_is_a_fresh_exception_of_its_type_and_text(self):
        """Red on UB2's tree: the same exception object came back on every call."""
        reg = registry.load_registry()
        raised = []
        for _ in range(3):
            with pytest.raises(AreaRuleRefusal) as caught:
                reg.sigma(3.0)
            raised.append(caught.value)
        assert len({id(e) for e in raised}) == 3 and len({str(e) for e in raised}) == 1
        held = reg.held_out(("T1-TN",))
        again = []
        for _ in range(2):
            with pytest.raises(HeldOut) as caught:
                held.evaluate("v('T1-TN-01')")
            again.append(caught.value)
        assert again[0] is not again[1] and again[0].args == again[1].args


# ---------------------------------------------------------------------------------------------------------- mi-R2
class TestCitedCellPositiveMiR2:
    @pytest.mark.parametrize("stated", ["0 * v('S-SRAM-10')", "-1 * v('S-SRAM-10')"])
    @pytest.mark.parametrize("name", ["generic_estimated_22nm", "loihi"])
    def test_a_cited_cell_of_no_positive_area_is_a_named_refusal_at_load_and_at_the_pricer(self, name, stated):
        """A guard on UB2's tree (its check stands untested there); red on its mutant N_C5 (the check skipped)."""
        physics = get_platform_physics(name)
        block = replace(physics.area_rule, node={"node_nm": physics.area_rule.node_nm, "bitcell_um2": stated})
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert "not a cell's area" in str(caught.value)
        org = (physics.area_rule.organization or {"rows": 256, "cols": 256})
        nominal, refusal, _k = _area(replace(physics, area_rule=block), 4, org["rows"], org["cols"], 4)
        assert nominal is None and "not a cell's area" in refusal, refusal
