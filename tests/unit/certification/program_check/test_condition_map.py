"""[W15] The condition map is complete and every rule it checks statically has a seeded fault it catches: every
hypothesis of the theorem box has a row of its kind; every named rule of ``RULES`` has a mutant (``MUTANTS``) or an
entry of ``UNMUTATED`` with its reason; every rule a predicate can raise is in ``RULES``; every condition left to the
sampled run names why; every executor code clause the verdict assumes names its code site; the check reads the program
read-only."""

from __future__ import annotations

import ast
import importlib
import json
import pickle
import pkgutil
from pathlib import Path

from mimarsinan.certification import program_check
from mimarsinan.certification.program_check import ASSUMED, CONDITION_MAP, RULES, UNMUTATED

from unit.certification import program_check as tests_package
from unit.certification.program_check import programs as P
from unit.certification.program_check import test_mutants

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "theorem_box" / "hypothesis_table.json"
SRC = Path(program_check.__file__).resolve().parent


def _test_names() -> set:
    names = set()
    for info in pkgutil.iter_modules(tests_package.__path__):
        if info.name.startswith("test_"):
            module = importlib.import_module(f"{tests_package.__name__}.{info.name}")
            names |= {n for n in dir(module) if n.startswith("test_")}
    return names


def test_every_hypothesis_of_the_theorem_box_has_a_row_of_its_kind():
    rows = {s.condition: s for s in CONDITION_MAP}
    for hypothesis in json.loads(FIXTURE.read_text())["hypotheses"]:
        assert hypothesis["id"] in rows, hypothesis["id"]
        assert rows[hypothesis["id"]].kind == hypothesis["kind"], hypothesis["id"]
    assert {s.theorem for s in CONDITION_MAP} == {"theorem1", "proposition2"}


def test_every_named_rule_has_a_mutant_or_its_reason():
    names = _test_names()
    claimed = {(c, r) for c, rules in RULES.items() for r in rules}
    assert set(RULES) <= {s.condition for s in CONDITION_MAP}
    for key in sorted(claimed):
        mutants = test_mutants.MUTANTS.get(key, ())
        if not mutants:
            assert UNMUTATED.get(key), f"rule {key} has no mutant and no reason"
            continue
        assert set(mutants) <= names, (key, set(mutants) - names)
    assert set(test_mutants.MUTANTS) <= claimed, set(test_mutants.MUTANTS) - claimed
    assert set(UNMUTATED) <= claimed and not set(UNMUTATED) & set(test_mutants.MUTANTS)


def _raised_rules() -> set:
    """``(condition, rule)`` of every ``Violation`` the check's modules build with literal names."""
    out = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Violation" and len(node.args) >= 2:
                cond, rule = node.args[0], node.args[1]
                if isinstance(cond, ast.Constant) and isinstance(rule, ast.Constant):
                    out.add((cond.value, rule.value))
    return out


def test_every_rule_a_predicate_raises_is_named_in_the_map():
    claimed = {(c, r) for c, rules in RULES.items() for r in rules}
    raised = {key for key in _raised_rules() if key[0] != "forms"}
    assert raised <= claimed, raised - claimed


def test_every_condition_left_to_the_sampled_run_names_why_and_every_assumption_its_site():
    open_ = {s.condition for s in CONDITION_MAP if s.not_static_reason}
    assert open_ == {"H2", "H3", "H4", "half-step", "seam-currency"}, open_
    for assumption in ASSUMED:
        assert assumption.site and assumption.statement and assumption.status in ("assumed_executor",
                                                                                  "assumed_reference")
        assert assumption.theorems and set(assumption.theorems) <= {"theorem1", "proposition2"}


def test_the_check_leaves_the_program_byte_identical():
    for program, ir in (P.streamed_multipass(), P.coalesced(), P.residency_planned(), P.value_program(),
                        P.resident_chain()):
        before = pickle.dumps(program)
        semantics = "value_integer" if any(getattr(c, "readout_law", None) is not None
                                           for s in P.neural(program) for c in s.hard_core_mapping.cores) \
            else "streamed_lif"
        P.run(program, ir, P.decl(semantics))
        assert pickle.dumps(program) == before
