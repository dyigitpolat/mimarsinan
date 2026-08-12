"""Ratchet: ``is_encoding_layer`` has ONE placement writer, and no builder writes it.

The reported defect was a second writer: ``SimpleMLPBuilder`` set
``is_encoding_layer = True`` at build, unconditionally, so the configured
``encoding_layer_placement`` could never take effect for the one ``native``
model. Two writers with different inputs is the shape of the bug — this guard
keeps the placement decision in ``torch_mapping/encoding_layers.py``.

``mapping/support/negative_boundary.py`` is the second SANCTIONED writer and a
different decision: with ``negative_value_shift=off`` a perceptron consuming a
signed host boundary is subsumed forward to the host regardless of placement.
It runs after placement and only ever ADDS host placements.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "mimarsinan"
FIELD = "is_encoding_layer"

# module -> why it may write the field.
SANCTIONED_WRITERS = {
    "torch_mapping/encoding_layers.py": "the placement decision itself",
    "mapping/support/negative_boundary.py": "negative-boundary subsume-forward",
    "models/perceptron_mixer/perceptron.py": "the unset constructor default",
}

BUILDERS = SRC / "models" / "builders"


def _is_setattr_write(node: ast.AST) -> bool:
    """``setattr(perceptron, "is_encoding_layer", ...)`` — a write in disguise.

    An attribute-assignment-only walk waves this through, and it is the FIRST
    shape a second writer takes when someone works around the ratchet.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if name != "setattr" or len(node.args) < 2:
        return False
    attr = node.args[1]
    return isinstance(attr, ast.Constant) and attr.value == FIELD


def _writes_field(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if _is_setattr_write(node):
            return True
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr == FIELD:
                return True
    return False


def test_only_sanctioned_modules_write_the_encoding_mark():
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if str(path.relative_to(SRC)) not in SANCTIONED_WRITERS and _writes_field(path)
    )
    assert not offenders, (
        f"{FIELD} is written outside the sanctioned owners {sorted(SANCTIONED_WRITERS)}: "
        f"{offenders}. Placement is resolved once, by mark_encoding_layers, from "
        "encoding_layer_placement; a second writer silently overrides the config."
    )


def test_the_detector_sees_both_write_shapes(tmp_path):
    """The ratchet is only as good as its detector — pin both shapes it must catch."""
    direct = tmp_path / "direct.py"
    direct.write_text(f"def f(p):\n    p.{FIELD} = True\n", encoding="utf-8")
    indirect = tmp_path / "indirect.py"
    indirect.write_text(f'def f(p):\n    setattr(p, "{FIELD}", True)\n', encoding="utf-8")
    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        f'def f(p):\n    return getattr(p, "{FIELD}", False)\n', encoding="utf-8"
    )

    assert _writes_field(direct)
    assert _writes_field(indirect), "a setattr write must not slip past the ratchet"
    assert not _writes_field(innocent)


def test_no_builder_decides_encoding_placement():
    """Builders build; the flow-birth site resolves placement from the config."""
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in BUILDERS.rglob("*.py")
        if _writes_field(path)
    )
    assert not offenders, (
        f"builders must not write {FIELD}: {offenders}. A baked mark makes "
        "encoding_layer_placement a no-op for that model type."
    )
