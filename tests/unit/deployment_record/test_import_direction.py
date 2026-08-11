"""The record is a near-leaf: ``search``/``gui``/``pipelining`` import it, never the reverse.

Stage 6 makes the direction load-bearing — ``search.results`` projects the
objectives registry and the wizard schema serves it — so a back-import would
close a cycle between the artifact and its consumers.
"""

import ast
from pathlib import Path

MODULE_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "mimarsinan" / "deployment_record"
)
FORBIDDEN = ("mimarsinan.search", "mimarsinan.gui", "mimarsinan.pipelining")


def _imported_modules(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


class TestImportDirection:
    def test_the_module_tree_is_non_empty(self):
        assert list(MODULE_ROOT.rglob("*.py"))

    def test_no_file_imports_a_consumer_module(self):
        offenders = []
        for path in sorted(MODULE_ROOT.rglob("*.py")):
            for module in _imported_modules(path.read_text(encoding="utf-8")):
                if any(
                    module == banned or module.startswith(banned + ".")
                    for banned in FORBIDDEN
                ):
                    offenders.append(f"{path.name}: {module}")
        assert not offenders, f"deployment_record must not import its consumers: {offenders}"
