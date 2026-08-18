"""Every number the wizard renders carries the bounds its schema declares.

A served field spec states the range a run accepts — ``evaluation_budget``
floors at 1, ``pop_size`` at 2, a builder's ``depth`` runs 4..32 — and the
rendered control is where that range either reaches the user or is silently
dropped. Dropped, the form can offer a value the run refuses BY NAME, so a
typed 0 travels through the whole draft and dies at the step. The derivation is
ONE pure function in ``wizard/structured.js``, executed here under node (the
browser's own rule, never a python re-implementation), and every numeric widget
in that file renders through it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pytest
from js_module import JS_ROOT, call_js

from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
from mimarsinan.gui.wizard.schema_api import config_schema_payload
from mimarsinan.models.builders.wizard_schema import get_all_model_type_schemas
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    resolve_evaluation_budget,
)

MODULE = JS_ROOT / "wizard" / "structured.js"


def attrs(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The input attributes the browser derives for *spec*."""
    return call_js(MODULE, "numericAttrs", spec)


def bounded(fields: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Only the specs that declare a range — the ones with something to drop."""
    return [(name, spec) for name, spec in fields if "min" in spec or "max" in spec]


def nas_numeric_fields() -> List[Tuple[str, Dict[str, Any]]]:
    nas = get_wizard_nas_schema()
    return bounded([
        (f"{group}.{name}", spec)
        for group in ("common_fields", "agent_evolve_fields", "compilagent_fields")
        for name, spec in nas[group].items()
        if spec.get("type") in ("int", "float")
    ])


def builder_numeric_fields() -> List[Tuple[str, Dict[str, Any]]]:
    return bounded([
        (f"{schema['id']}.{spec['key']}", spec)
        for schema in get_all_model_type_schemas()
        for spec in (schema.get("config_schema") or [])
        if isinstance(spec, dict) and spec.get("type") == "number"
    ])


def declared_bounds(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"min": spec.get("min"), "max": spec.get("max")}


def rendered_bounds(spec: Dict[str, Any]) -> Dict[str, Any]:
    rendered = attrs(spec)
    return {"min": rendered["min"], "max": rendered["max"]}


def _ids(fields: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    return [name for name, _ in fields]


class TestTheControlOffersWhatTheSchemaDeclares:
    @pytest.mark.parametrize(
        "name,spec", nas_numeric_fields(), ids=_ids(nas_numeric_fields()),
    )
    def test_every_search_field_renders_its_declared_range(self, name, spec):
        assert rendered_bounds(spec) == declared_bounds(spec), (
            f"the wizard drops {name}'s declared range"
        )

    @pytest.mark.parametrize(
        "name,spec", builder_numeric_fields(), ids=_ids(builder_numeric_fields()),
    )
    def test_every_model_config_field_renders_its_declared_range(self, name, spec):
        assert rendered_bounds(spec) == declared_bounds(spec), (
            f"the wizard drops {name}'s declared range"
        )

    def test_the_preprocessing_floor_is_declared_by_the_schema(self):
        # The widget used to hardcode this floor; the served schema is the
        # configurability SSOT, so the floor is declared where every other
        # field's is and rendered by the same rule.
        served = config_schema_payload()["preprocessing_fields"]
        numeric = {
            name: spec for name, spec in served.items() if spec.get("type") == "int"
        }

        assert numeric, "the preprocessing sub-schema serves no numeric field"
        for name, spec in numeric.items():
            assert attrs(spec)["min"] == 1, f"{name} renders without its floor"

    def test_a_spec_that_declares_no_range_invents_none(self):
        assert attrs({"type": "int"}) == {"step": "1", "min": None, "max": None}


class TestTheBudgetFloorIsTheOneTheRunEnforces:
    """[TS1] The claim the wizard's budget field rests on: what the form can
    offer, the step accepts. Pinned across the seam, not inside the schema."""

    def spec(self) -> Dict[str, Any]:
        return get_wizard_nas_schema()["common_fields"]["evaluation_budget"]

    def test_the_rendered_floor_is_accepted_by_the_step(self):
        floor = attrs(self.spec())["min"]

        assert floor is not None, "the budget field renders without a floor"
        budget = resolve_evaluation_budget({"evaluation_budget": floor})
        assert budget is not None and budget.limit == floor

    def test_the_run_refuses_everything_below_the_rendered_floor(self):
        floor = attrs(self.spec())["min"]

        with pytest.raises(ValueError, match="evaluation_budget"):
            resolve_evaluation_budget({"evaluation_budget": floor - 1})


class TestTheStepFollowsTheDeclaredType:
    @pytest.mark.parametrize("declared,step", [
        ({"type": "int"}, "1"),
        ({"type": "float"}, "any"),
        ({"type": "number"}, "any"),
        ({"type": "int", "step": 16}, "16"),
    ])
    def test_the_rendered_step(self, declared, step):
        assert attrs(declared)["step"] == step


class TestEveryNumberFieldGoesThroughTheOneRule:
    """Executing the widgets themselves would need a DOM; what a test can still
    say without one is that no widget derives numeric attributes beside the
    rule, and that the rule's output reaches the element."""

    def source(self) -> str:
        return MODULE.read_text(encoding="utf-8")

    def numeric_calls(self, source: str) -> List[str]:
        """Each ``numeric(...)`` call's own text, its definition excluded."""
        calls = []
        for match in re.finditer(r"(?<![\w$])numeric\(", source):
            if source[:match.start()].rstrip().endswith("function"):
                continue
            depth, index = 0, match.end() - 1
            while index < len(source):
                depth += {"(": 1, ")": -1}.get(source[index], 0)
                if depth == 0:
                    break
                index += 1
            calls.append(source[match.start():index + 1])
        return calls

    def body_of(self, source: str, signature: str) -> str:
        body = source[source.index(signature):]
        return body[: body.index("\n}")]

    def test_every_numeric_control_derives_its_attributes_from_the_rule(self):
        calls = self.numeric_calls(self.source())

        assert calls, "structured.js renders no numeric control"
        for call in calls:
            assert "numericAttrs(" in call, (
                "a numeric control whose attributes do not come from "
                f"numericAttrs is a control this test cannot speak for: {call}"
            )

    def test_no_widget_keeps_a_second_copy_of_the_rule(self):
        source = self.source()
        rule = self.body_of(source, "export function numericAttrs(")
        elsewhere = source.replace(rule, "")

        for key in ("spec.min", "spec.max", "spec.step"):
            assert key not in elsewhere, (
                f"{key} is read outside numericAttrs — the range a field "
                "declares must be derived in exactly one place"
            )

    def test_both_bounds_reach_the_input_element(self):
        body = self.body_of(self.source(), "function numeric(")

        assert "input.min = String(min)" in body
        assert "input.max = String(max)" in body
