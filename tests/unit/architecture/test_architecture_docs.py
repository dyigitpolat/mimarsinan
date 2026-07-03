"""ARCHITECTURE.md policy: one root + one per top-level module, drift-guarded."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "mimarsinan"

_TABLE_REF = re.compile(r"^\|\s*`([^`]+)`", re.MULTILINE)


def _top_level_modules():
    return sorted(
        d for d in SRC.iterdir()
        if d.is_dir() and d.name != "__pycache__" and (d / "__init__.py").exists()
    )


def _direct_children(module_dir):
    names = set()
    for child in module_dir.iterdir():
        if child.name in ("__init__.py", "__pycache__", "ARCHITECTURE.md"):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name + "/")
        elif child.suffix == ".py":
            names.add(child.name)
    return names


def test_root_architecture_doc_exists_and_is_lean():
    doc = REPO / "ARCHITECTURE.md"
    assert doc.exists()
    assert len(doc.read_text().splitlines()) <= 400, (
        "the root ARCHITECTURE.md is an overview, not a manual"
    )


def test_every_top_level_module_has_a_doc():
    missing = [d.name for d in _top_level_modules() if not (d / "ARCHITECTURE.md").exists()]
    assert not missing, f"top-level modules without ARCHITECTURE.md: {missing}"


def test_no_leaf_architecture_docs():
    allowed = {SRC / d.name / "ARCHITECTURE.md" for d in _top_level_modules()}
    extras = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("ARCHITECTURE.md")
        if p not in allowed
    ]
    assert not extras, f"ARCHITECTURE.md files live only at module roots: {extras}"


def test_docs_reference_only_existing_children():
    stale = []
    for module_dir in _top_level_modules():
        doc = module_dir / "ARCHITECTURE.md"
        if not doc.exists():
            continue
        existing = _direct_children(module_dir)
        for ref in _TABLE_REF.findall(doc.read_text()):
            if ref not in existing:
                stale.append(f"{module_dir.name}/ARCHITECTURE.md -> {ref}")
    assert not stale, f"docs reference children that do not exist: {stale}"


def test_docs_cover_every_direct_child():
    undocumented = []
    for module_dir in _top_level_modules():
        doc = module_dir / "ARCHITECTURE.md"
        if not doc.exists():
            continue
        referenced = set(_TABLE_REF.findall(doc.read_text()))
        for child in _direct_children(module_dir):
            if child not in referenced:
                undocumented.append(f"{module_dir.name}/{child}")
    assert not undocumented, f"direct children missing from Key-files tables: {undocumented}"
