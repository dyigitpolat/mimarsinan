"""Optimizers depend on the introspection types — never on ``mapping`` internals.

The compilagent backend used to reach into ``mapping`` for the softcore type,
the capability class and the layout verifier, which is how it ended up computing
a layout answer with three of the platform's capability bits. The registry is now
the only door: this is the same AST direction test the record's own boundary uses,
pointed at the optimizer that consumes it.
"""

from __future__ import annotations

import ast
from pathlib import Path

COMPILAGENT = (
    Path(__file__).resolve().parents[5]
    / "src" / "mimarsinan" / "search" / "optimizers" / "compilagent"
)
FORBIDDEN = ("mimarsinan.mapping",)
INTROSPECTION = "mimarsinan.deployment_record.introspection"


def _imported_modules(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _all_modules():
    return sorted(COMPILAGENT.rglob("*.py"))


class TestImportDirection:
    def test_the_module_tree_is_non_empty(self):
        assert _all_modules()

    def test_no_optimizer_module_reaches_into_mapping(self):
        offenders = []
        for path in _all_modules():
            for module in _imported_modules(path.read_text(encoding="utf-8")):
                if any(
                    module == banned or module.startswith(banned + ".")
                    for banned in FORBIDDEN
                ):
                    offenders.append(f"{path.name}: {module}")
        assert not offenders, (
            "optimizers consume deployment_record.introspection types only: "
            f"{offenders}"
        )

    def test_the_layout_payload_is_collected_through_the_registry(self):
        source = (COMPILAGENT / "backend" / "backend_layout.py").read_text(
            encoding="utf-8"
        )
        imported = set(_imported_modules(source))
        assert any(
            module == INTROSPECTION or module.startswith(INTROSPECTION + ".")
            for module in imported
        ), sorted(imported)
