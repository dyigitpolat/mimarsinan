"""Template expansion with an explicit substitution table and no survivors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Tuple

#: ``@NAME@``, deliberately NOT ``{{NAME}}``: the templates are Verilog, whose
#: concatenation syntax already spells ``{{`` and would collide.
PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")

_REPO_ROOT = Path(__file__).resolve().parents[5]
HW_GEN_ROOT = _REPO_ROOT / "hw" / "gen"
HW_VENDOR_ROOT = _REPO_ROOT / "hw" / "vendor" / "odin"

TEMPLATE_SUFFIX = ".tmpl"


class TemplateError(ValueError):
    """A template and its substitution table disagree."""


def template_path(name: str) -> Path:
    """The template source of ``name`` (e.g. ``odin_gen_core.v``), refusing absence."""
    path = HW_GEN_ROOT / f"{name}{TEMPLATE_SUFFIX}"
    if not path.is_file():
        raise TemplateError(
            f"no template at {path}: the generator emits RTL from "
            f"{HW_GEN_ROOT}, and a missing template is a missing capability, "
            f"not a reason to synthesize text here.")
    return path


def placeholders_in(text: str) -> Tuple[str, ...]:
    """Every distinct placeholder name a template uses, in first-seen order."""
    seen: Dict[str, None] = {}
    for match in PLACEHOLDER.finditer(text):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


def expand(text: str, values: Mapping[str, object]) -> str:
    """Substitute EXACTLY the declared placeholders; both directions are checked.

    An unfilled placeholder would ship a template into a simulator, and an
    unused value means the table names something the template no longer has —
    which is how a renamed parameter silently keeps its old default.
    """
    names = set(placeholders_in(text))
    provided = set(values)
    missing = sorted(names - provided)
    if missing:
        raise TemplateError(
            f"the substitution table has no value for: {', '.join(missing)}")
    unused = sorted(provided - names)
    if unused:
        raise TemplateError(
            f"the substitution table names placeholders the template does not "
            f"have: {', '.join(unused)}")
    rendered = PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)
    assert_no_placeholders(rendered)
    return rendered


def assert_no_placeholders(text: str) -> None:
    """The emitted-artifact gate: nothing that looks like a placeholder survives."""
    leftovers = placeholders_in(text)
    if leftovers:
        raise TemplateError(
            f"the rendered artifact still carries placeholder(s): "
            f"{', '.join(leftovers)}")


def render(name: str, values: Mapping[str, object]) -> str:
    """Expand the named template from ``hw/gen`` with ``values``."""
    return expand(template_path(name).read_text(encoding="utf-8"), values)
