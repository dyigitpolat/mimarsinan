"""Offline ratchet: the GUI serves entirely from static/ with zero CDN reach-outs."""

import re
from pathlib import Path

import pytest

_STATIC_DIR = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static"
)
_VENDOR_DIR = _STATIC_DIR / "vendor"

_EXTERNAL_REF_PATTERNS = (
    re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE),
    re.compile(r"""@import\s+(?:url\()?["']?https?://""", re.IGNORECASE),
    re.compile(r"""url\(\s*["']?https?://""", re.IGNORECASE),
    # ES-module imports (`import … from 'https://…'`, `import 'https://…'`,
    # dynamic `import('https://…')`): a CDN module import fails the whole
    # dependent module graph on an air-gapped monitor.
    re.compile(r"""(?:from|import)\s+["']https?://""", re.IGNORECASE),
    re.compile(r"""import\(\s*["']https?://""", re.IGNORECASE),
)


def _static_files(*suffixes: str) -> list[Path]:
    files = [
        p for p in sorted(_STATIC_DIR.rglob("*"))
        if p.is_file() and p.suffix in suffixes and _VENDOR_DIR not in p.parents
    ]
    assert files, f"no {suffixes} files under {_STATIC_DIR}"
    return files


@pytest.mark.parametrize("path", _static_files(".html", ".css", ".js"), ids=lambda p: p.name)
def test_static_asset_has_no_external_references(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    hits = [
        line.strip()
        for line in text.splitlines()
        if any(pattern.search(line) for pattern in _EXTERNAL_REF_PATTERNS)
    ]
    assert not hits, (
        f"{path.name} references external hosts (breaks offline/air-gapped "
        f"monitors); vendor the asset under static/vendor/ instead: {hits}"
    )


def test_plotly_is_vendored() -> None:
    plotly = _VENDOR_DIR / "plotly-2.27.0.min.js"
    assert plotly.is_file(), "Plotly must be vendored (no CDN)"
    assert plotly.stat().st_size > 1_000_000, "vendored Plotly looks truncated"


def test_fonts_are_vendored() -> None:
    fonts_css = _VENDOR_DIR / "fonts.css"
    assert fonts_css.is_file(), "self-hosted @font-face sheet must exist"
    css = fonts_css.read_text(encoding="utf-8")
    for family in ("Inter", "JetBrains Mono", "Outfit"):
        assert family in css, f"fonts.css must declare {family!r}"
    woff2_refs = re.findall(r"url\(([^)]+\.woff2)\)", css)
    assert woff2_refs, "fonts.css must reference local .woff2 files"
    for ref in woff2_refs:
        rel = ref.strip("'\"").removeprefix("/static/vendor/")
        assert (_VENDOR_DIR / rel).is_file(), f"missing vendored font file {ref}"


@pytest.mark.parametrize("page", ["index.html", "welcome.html"])
def test_plotly_pages_load_the_vendored_script(page: str) -> None:
    html = (_STATIC_DIR / page).read_text(encoding="utf-8")
    assert "/static/vendor/plotly-2.27.0.min.js" in html


@pytest.mark.parametrize("page", ["index.html", "welcome.html", "wizard.html"])
def test_pages_load_the_vendored_fonts(page: str) -> None:
    html = (_STATIC_DIR / page).read_text(encoding="utf-8")
    assert "/static/vendor/fonts.css" in html
