""
from __future__ import annotations

import functools
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = HERE.parents[2]
SCIVIZ_ROOT = REPO / "sciviz"
if str(SCIVIZ_ROOT) not in sys.path:
    sys.path.insert(0, str(SCIVIZ_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib import scale as _mscale
from matplotlib.ticker import FuncFormatter as _FuncFormatter


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

DATA = HERE / "data"
OUT = HERE / "out"
MOCK = OUT / "mock"
for _d in (DATA, OUT):
    _d.mkdir(exist_ok=True)


COLUMN_PT = 252 * 72 / 72.27
TEXT_PT = 516 * 72 / 72.27
WIDTHS_PT = {"column": COLUMN_PT, "text": TEXT_PT}
MAX_H_PT = {"column": 240.0, "text": 250.0}


def width_pt(width) -> float:
    return WIDTHS_PT[width] if isinstance(width, str) else float(width)


def width_class(w_pt: float) -> str:
    return "column" if w_pt <= COLUMN_PT + 1.0 else "text"


FONT_FAMILIES = ["Noto Sans", "Noto Sans Math", "Noto Sans Symbols", "Noto Sans Symbols2", "DejaVu Sans"]
FONT_STACK = "'Noto Sans', 'Noto Sans Math', 'Noto Sans Symbols', 'Noto Sans Symbols2', 'DejaVu Sans', sans-serif"
MIN_FONT_PT = 6.5
FS = {
    "title": 7.5,
    "focal": 8.5,
    "label": 7.5,
    "tick": 7.0,
    "small": 6.5,
}
LW = {"axis": 0.6, "grid": 0.4, "line": 1.3, "thin": 0.8, "heavy": 1.9, "edge": 0.7}
MS = 4.2


for _f in sorted(Path(os.environ.get("FIG_FONT_DIR", "/usr/share/fonts/truetype/noto")).glob("NotoSans*-*.ttf")):
    try:
        _fm.fontManager.addfont(str(_f))
    except Exception:
        pass


C = {
    "blue":   "#2b6cb0",
    "teal":   "#1f9d8a",
    "orange": "#e07a2f",
    "violet": "#7a5aa6",
    "amber":  "#d49a12",
    "green":  "#2f9e58",
    "red":    "#c2413a",
    "ink":    "#1b1f24",
    "ink2":   "#4a5058",
    "gray":   "#858b93",
    "gray2":  "#b9bec5",
    "gray3":  "#e3e6ea",
    "white":  "#ffffff",
}
ROLE = {
    "ours": C["blue"], "cores": C["blue"], "certified": C["blue"], "multipass": C["blue"],
    "operand": C["teal"], "series2": C["teal"],
    "writes": C["orange"], "write_dominated": C["orange"],
    "changing": C["violet"], "admission": C["amber"], "cost_only": C["blue"],
    "exact": C["green"], "defect": C["red"],
    "host": C["gray"], "prior": C["gray"], "refused": C["gray2"], "reference": C["gray"],
    "pending": C["gray2"], "band": C["gray3"], "grid": C["gray3"],
    "ink": C["ink"], "ink2": C["ink2"],
}
SERIES = [C["blue"], C["teal"], C["orange"], C["violet"], C["gray"]]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def tint(color: str, amount: float = 0.85) -> str:
    ""
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v + (255 - v) * amount):02x}" for v in rgb)


def setup():
    ""
    plt.rcParams.update({
        "font.family": FONT_FAMILIES,
        "font.size": FS["label"],
        "mathtext.fontset": "custom", "mathtext.rm": "Noto Sans", "mathtext.it": "Noto Sans:italic",
        "mathtext.bf": "Noto Sans:bold", "mathtext.fallback": "stixsans", "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": C["ink"], "axes.labelcolor": C["ink"], "axes.edgecolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "xtick.labelcolor": C["ink"], "ytick.labelcolor": C["ink"],
        "axes.labelsize": FS["label"], "axes.titlesize": FS["title"], "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 4.0, "axes.labelpad": 2.5,
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["tick"],
        "axes.linewidth": LW["axis"], "xtick.major.width": LW["axis"], "ytick.major.width": LW["axis"],
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": C["gray3"], "grid.linewidth": LW["grid"],
        "axes.axisbelow": True,
        "lines.linewidth": LW["line"], "lines.markersize": MS, "lines.markeredgewidth": 0.7,
        "patch.linewidth": LW["edge"], "hatch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 2.0 / 72, "figure.constrained_layout.w_pad": 2.0 / 72,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
    })


def figsize(width="column", height_pt: float = 170.0):
    ""
    return (width_pt(width) / 72.0, float(height_pt) / 72.0)


def panel_title(ax, tag: str, title: str = "", **kw):
    ""
    text = f"({tag}) {title}".strip() if tag else title
    return ax.set_title(text, loc="left", fontsize=FS["title"], fontweight="bold", color=C["ink"], **kw)


def grid(ax, axis: str = "y"):
    ax.grid(True, axis=axis, color=C["gray3"], linewidth=LW["grid"])


def focal(ax, text: str, xy, xytext=None, *, color=None, coords="data", textcoords=None, arrow=True, **kw):
    ""
    color = color or C["blue"]
    ap = dict(arrowstyle="-", color=color, linewidth=0.7, shrinkA=1.5, shrinkB=2.0) if (arrow and xytext is not None) else None
    return ax.annotate(text, xy=xy, xytext=xytext if xytext is not None else xy, xycoords=coords,
                       textcoords=textcoords or coords, fontsize=FS["focal"], fontweight="bold", color=color,
                       arrowprops=ap, **kw)


def direct_label(ax, x, y, text, *, color=None, dx=3.0, dy=0.0, ha="left", va="center", size=None, **kw):
    ""
    return ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=size or FS["label"], color=color or C["ink"], **kw)


def top_legend(fig, handles, labels, *, ncol=None, **kw):
    ""
    return fig.legend(handles, labels, loc="outside upper center", ncol=ncol or len(labels), frameon=False,
                      fontsize=FS["tick"], handlelength=1.4, columnspacing=1.2, **kw)


def zero_mark(ax, x, y, *, size=None, **kw):
    ""
    ax.plot([x], [y], marker="o", markersize=size or 7.0, color=C["green"], markeredgewidth=0, zorder=5, **kw)
    ax.plot([x], [y], marker="$\\checkmark$", markersize=(size or 7.0) * 0.6, color="white", markeredgewidth=0,
            zorder=6)


def text_width_pt(fig, text, size, weight="normal"):
    ""
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi
    t.remove()
    return w


def thousands(v) -> str:
    ""
    return f"{int(v):,}"


class _LogCatch(logging.Handler):
    ""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _log_tick_label(v, pos=None):
    ""
    if v <= 0:
        return ""
    e = math.log10(v)
    return rf"$10^{{{round(e)}}}$" if abs(e - round(e)) < 1e-9 else ""


class _PlainLogScale(_mscale.LogScale):
    ""

    def set_default_locators_and_formatters(self, axis):
        super().set_default_locators_and_formatters(axis)
        axis.set_major_formatter(_FuncFormatter(_log_tick_label))


_mscale.register_scale(_PlainLogScale)


def _plain_log_ticks(fig):
    ""
    from matplotlib.ticker import LogFormatterSciNotation

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log" and type(axis.get_major_formatter()) is LogFormatterSciNotation:
                axis.set_major_formatter(_FuncFormatter(_log_tick_label))


def _drawn_texts(fig):
    ""
    from matplotlib.text import Text
    drawn, orig = [], Text.draw

    def rec(self, renderer, *a, **k):
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return orig(self, renderer, *a, **k)

    Text.draw = rec
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    seen, out = set(), []
    for t in drawn:
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def check_fig(fig, name: str) -> list:
    ""
    from matplotlib.text import Text
    texts = _drawn_texts(fig)
    r = fig.canvas.get_renderer()
    fails = []
    boxes = []
    W, H = fig.bbox.width, fig.bbox.height
    px = fig.dpi / 72.0
    for t in texts:
        try:
            bb = Text.get_window_extent(t, r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        if t.get_fontsize() < MIN_FONT_PT - 0.01:
            fails.append(f"FAIL {name}: text {t.get_text()!r} at {t.get_fontsize():.1f} pt, under the {MIN_FONT_PT} pt floor")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            fails.append(f"FAIL {name}: text {t.get_text()!r} runs off the canvas")
        boxes.append((t, bb))
    for i in range(len(boxes)):
        ti, a = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, b = boxes[j]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.7 * px and oy > 0.7 * px:
                fails.append(f"FAIL {name}: text {ti.get_text()!r} overlaps text {tj.get_text()!r}")
    return fails, texts


def _rasterize(pdf: Path, dpi: int = 200) -> Path:
    png = pdf.with_suffix(".png")
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf), str(png.with_suffix(""))],
                   check=True, capture_output=True)
    return png


def _pdf_page_pt(pdf: Path):
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _report(name: str, pdf: Path, fails: list, min_text):
    w, h = _pdf_page_pt(pdf)
    cls = width_class(w)
    lines = list(fails)
    cap = caption_source(name)
    if cap is not None:
        n_cap = caption_words(cap)
        if n_cap > CAPTION_MAX_WORDS:
            lines.append(f"FAIL {name}: caption of {n_cap} words (captions/{name}.tex), over the "
                         f"{CAPTION_MAX_WORDS}-word limit")
    if abs(w - WIDTHS_PT[cls]) > 1.5:
        lines.append(f"FAIL {name}: width {w:.1f} pt is neither the column ({COLUMN_PT:.1f}) nor the text width ({TEXT_PT:.1f})")
    if h > MAX_H_PT[cls] + 0.5:
        lines.append(f"FAIL {name}: height {h:.0f} pt over the {MAX_H_PT[cls]:.0f} pt {cls} budget")
    if min_text is not None and min_text < MIN_FONT_PT - 0.01:
        lines.append(f"FAIL {name}: smallest text {min_text:.2f} pt, under the {MIN_FONT_PT} pt floor")
    for ln in lines:
        print(ln)
    mt = f"text >= {min_text:.1f} pt" if min_text is not None else "text n/a"
    print(f"{'OK  ' if not lines else 'FAIL'} {name}: {w:.0f} x {h:.0f} pt [{cls}], {mt}, {len(lines)} finding(s); "
          f"now look at out/{name}.png and at the page mock (S.mock)")


def save_fig(fig, name: str, *, mock_page: bool = True):
    ""
    _plain_log_ticks(fig)
    catch = _LogCatch()
    mlog = logging.getLogger("matplotlib")
    mlog.addHandler(catch)
    pdf = OUT / f"{name}.pdf"
    try:
        fails, texts = check_fig(fig, name)
        sizes = [t.get_fontsize() for t in texts]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(pdf)
    finally:
        mlog.removeHandler(catch)
    for w in caught:
        fails.append(f"FAIL {name}: {w.message}")
    for m in sorted(set(catch.messages)):
        fails.append(f"FAIL {name}: matplotlib: {m}")
    plt.close(fig)
    _rasterize(pdf)
    _report(name, pdf, fails, min(sizes) if sizes else None)
    if mock_page:
        mock(name)
    return pdf


def _bind_sciviz_theme():
    try:
        from sciviz import DEFAULT_THEME as t
    except ImportError:
        return None
    t.font_family = FONT_STACK
    t.font_title, t.font_subtitle = 9.0, 8.5
    t.font_panel_title, t.font_panel_tag, t.font_section = FS["title"], FS["title"], 8.0
    t.font_label, t.font_small, t.font_tiny, t.font_micro = FS["label"], FS["tick"], FS["small"], FS["small"]
    t.font_math = 8.0
    try:
        from sciviz.core import _fonts
        _fonts._registry_cache.clear()
        _fonts._fallback_families.cache_clear()
    except (ImportError, AttributeError):
        pass
    return t


THEME = _bind_sciviz_theme()


def paper(body, width="column", **kwargs):
    ""
    from sciviz import Diagram
    kwargs.setdefault("min_effective_font_pt", MIN_FONT_PT)
    return Diagram.for_paper(body, target_width_pt=width_pt(width), **kwargs)


def save_diagram(diagram, name: str, *, mock_page: bool = True):
    ""
    pdf = OUT / f"{name}.pdf"
    fails = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagram.save(str(pdf), text_mode="hybrid")
        svg = diagram.render(embed_fonts=False)
    for w in caught:
        fails.append(f"FAIL {name}: {w.message}")
    _rasterize(pdf)
    w_pt, _ = _pdf_page_pt(pdf)
    canvas_w = None
    m = re.search(r'viewBox="[0-9.\-]+ [0-9.\-]+ ([0-9.]+) ', svg)
    if m:
        canvas_w = float(m.group(1))
    scale = (w_pt / canvas_w) if canvas_w else 1.0
    sizes = [float(s) * scale for s in re.findall(r'font-size="([0-9.]+)', svg)]
    words = len(re.findall(r"[A-Za-z][A-Za-z\-]+", " ".join(re.findall(r">([^<>]+)<", svg))))
    if words > 60:
        fails.append(f"FAIL {name}: {words} words in the diagram, over the 60-word limit")
    _report(name, pdf, fails, min(sizes) if sizes else None)
    if mock_page:
        mock(name)
    return pdf


def load(name: str):
    ""
    with open(DATA / f"{name}.json") as f:
        return json.load(f)


def save_keys(name: str, rows: list):
    ""
    bad = [r for r in rows if not (r.get("key") or r.get("source"))]
    for r in bad:
        print(f"FAIL {name}: plotted value without a key or a source: {r}")
    (OUT / f"{name}.keys.json").write_text(json.dumps(rows, indent=1) + "\n")


CAPTION_MAX_WORDS = 60

_VALUE_MACROS = {"val": ("display",), "nv": ("display",), "valint": ("display",),
                 "valn": ("value",), "valu": ("unit",), "valrange": ("range",), "valp": ("p_lands",)}
_VALUE_RE = re.compile(r"\\(?:(valas)\{([^{}]*)\}\{([^{}]*)\}|("
                       + "|".join(sorted(_VALUE_MACROS, key=len, reverse=True)) + r")\{([^{}]*)\})")
def _ledger_entries() -> dict:
    ""
    return {}


def _ledger_tex(e: dict, field: str) -> str:
    return re.sub(r"([%&#_])", r"\\\1", str(e[field]))


def caption_source(name: str):
    ""
    return None


def caption_words(text: str) -> int:
    ""
    return len(re.findall(r"\S+", _VALUE_RE.sub("N", text or "")))


def caption_value_keys(text: str) -> list:
    ""
    out = []
    for m in _VALUE_RE.finditer(text or ""):
        k = m.group(2) if m.group(1) else m.group(5)
        if k not in out:
            out.append(k)
    return out


def caption_keys(name: str) -> list:
    ""
    E = _ledger_entries()
    return [{"what": f"Caption: {k}", "value": E[k]["display"] if k in E else "pending", "key": k}
            for k in caption_value_keys(caption_source(name))]


def _brace_groups(text: str, start: int, n: int):
    ""
    out, i = [], start
    for _ in range(n):
        if i >= len(text) or text[i] != "{":
            return None
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
                depth += 1
            elif text[j] == "}" and text[j - 1] != "\\":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth:
            return None
        out.append(text[i + 1:j])
        i = j + 1
    return out, i


def adoption_forms(text: str) -> str:
    ""
    E, macro = _ledger_entries(), "\\whenadopted"
    while True:
        at = text.find(macro + "{")
        if at < 0:
            return text
        got = _brace_groups(text, at + len(macro), 4)
        if got is None:
            return text
        (_run, keys, adopted, earlier), end = got
        ok = all((E.get(k.strip()) or {}).get("adopted") for k in keys.split(",") if k.strip())
        text = text[:at] + (adopted if ok else earlier) + text[end:]


def mock_caption(text: str):
    ""
    E, missing = _ledger_entries(), []
    text = adoption_forms(text)

    def one(m):
        if m.group(1):
            if m.group(2) not in E:
                missing.append(m.group(2))
            return m.group(3)
        macro, key = m.group(4), m.group(5)
        e = E.get(key)
        if e is None or e.get("placeholder"):
            if e is None:
                missing.append(key)
            return "pending"
        for field in _VALUE_MACROS[macro]:
            if e.get(field) not in (None, ""):
                return _ledger_tex(e, field)
        return ""

    return _VALUE_RE.sub(one, text), missing


_MOCK_TEX = r"""\documentclass[journal]{IEEEtran}
\usepackage{graphicx}\usepackage{lipsum}
\renewcommand{\ref}[1]{N}\providecommand{\eqref}[1]{(N)}\renewcommand{\eqref}[1]{(N)}
\providecommand{\costonly}{cost-only}\providecommand{\bundleref}{the evidence bundle}
\begin{document}
\title{Page Mock of One Figure at Its Printed Size}\author{figures}\maketitle
%(float)s
\lipsum[1-14]
\end{document}
"""


def mock(name: str, caption: str = None) -> Path:
    ""
    if os.environ.get("FIG_MOCK") != "1":
        return None
    MOCK.mkdir(exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    w, _ = _pdf_page_pt(pdf)
    star = "" if width_class(w) == "column" else "*"
    cap = caption or caption_source(name) or "The figure at its printed size."
    cap, missing = mock_caption(cap)
    for k in missing:
        print(f"FAIL {name}: the caption prints the key {k!r}, which the ledger does not hold")
    flt = (f"\\begin{{figure{star}}}[!t]\\centering\\includegraphics{{{pdf}}}"
           f"\\caption{{{cap}}}\\end{{figure{star}}}")
    tex = MOCK / f"{name}_page.tex"
    tex.write_text(_MOCK_TEX % {"float": flt})
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={MOCK}",
                          str(tex)], capture_output=True, text=True, cwd=str(MOCK))
    page = MOCK / f"{name}_page.pdf"
    if run.returncode != 0 or not page.exists():
        print(f"FAIL {name}: the page mock did not compile: {run.stdout[-400:]}")
        return page
    first = "1" if star == "" else "2"
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "150", "-f", first, "-l", first, str(page),
                    str(MOCK / f"{name}_page")], check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "300", str(pdf), str(MOCK / f"{name}_zoom")],
                   check=True, capture_output=True)
    for ext in (".aux", ".log"):
        (MOCK / f"{name}_page{ext}").unlink(missing_ok=True)
    print(f"mock {name}: out/mock/{name}_page.png (the page, 150 dpi), out/mock/{name}_zoom.png (300 dpi)")
    return page


__all__ = ["COLUMN_PT", "TEXT_PT", "WIDTHS_PT", "MAX_H_PT", "MIN_FONT_PT", "FS", "LW", "MS", "C", "ROLE", "SERIES",
           "MARKERS", "tint", "setup", "figsize", "panel_title", "grid", "focal", "direct_label", "top_legend",
           "zero_mark", "text_width_pt", "thousands", "check_fig", "save_fig", "paper", "save_diagram", "load", "save_keys", "mock", "THEME",
           "CAPTION_MAX_WORDS", "caption_source", "caption_words", "caption_value_keys", "caption_keys",
           "mock_caption",
           "DATA", "OUT", "HERE", "PAPER", "REPO"]


def shade(color: str, amount: float = 0.3) -> str:
    ""
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v * (1.0 - amount)):02x}" for v in rgb)


def luma(color: str) -> float:
    ""
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


TONES4 = [C["blue"], C["teal"], tint(C["orange"], 0.25), shade(C["violet"], 0.40)]
TONES4_MARKERS = ["o", "s", "^", "D"]

__all__ += ["shade", "luma", "TONES4", "TONES4_MARKERS"]


def placeholder_forms(text: str) -> str:
    ""
    E, macro = _ledger_entries(), "\\tabph"
    while True:
        at = text.find(macro + "{")
        if at < 0:
            return text
        got = _brace_groups(text, at + len(macro), 3)
        if got is None:
            return text
        (key, held, value), end = got
        e = E.get(key.strip())
        text = text[:at] + (held if (e is None or e.get("placeholder")) else value) + text[end:]


_mock_caption_before_tabph = mock_caption


def mock_caption(text: str):
    ""
    return _mock_caption_before_tabph(placeholder_forms(text))


__all__ += ["placeholder_forms"]


INK_CLEAR_PT = None
_check_fig_line_boxes = check_fig


def _ink_extent(fig, r, t, bb, cache):
    ""
    from matplotlib.text import Text
    from matplotlib.textpath import TextPath
    s = t.get_text()
    if "\n" in s or t.get_rotation() % 360 or t.get_usetex():
        return bb.y0, bb.y1, False
    fp = t.get_fontproperties()
    key = (tuple(fp.get_family()), fp.get_size_in_points(), fp.get_weight(), fp.get_style())
    if key not in cache:
        probe = fig.text(0, 0, "x", fontproperties=fp, va="baseline")
        pb = Text.get_window_extent(probe, r)
        probe.remove()
        cache[key] = (pb.y1, -pb.y0)
    _, min_desc = cache[key]
    ismath = len(re.findall(r"(?<!\\)\$", s)) >= 2
    try:
        _, _, d = r.get_text_width_height_descent(s, fp, ismath)
        ink = TextPath((0, 0), s, size=fp.get_size_in_points(), prop=fp).get_extents()
    except Exception:
        return bb.y0, bb.y1, False
    px = fig.dpi / 72.0
    base = bb.y0 + max(d, min_desc)
    return base + ink.y0 * px, base + ink.y1 * px, True


def _ink_span(fig, r, t, bb, x_lo, x_hi):
    ""
    from matplotlib.textpath import TextPath
    s = t.get_text()
    fp = t.get_fontproperties()
    ismath = len(re.findall(r"(?<!\\)\$", s)) >= 2
    _, _, d = r.get_text_width_height_descent(s, fp, ismath)
    probe = fig.text(0, 0, "x", fontproperties=fp, va="baseline")
    pb = probe.get_window_extent(r)
    probe.remove()
    px = fig.dpi / 72.0
    base = bb.y0 + max(d, -pb.y0)
    v = TextPath((0, 0), s, size=fp.get_size_in_points(), prop=fp).vertices
    ys = [base + y * px for x, y in v if x_lo <= bb.x0 + x * px <= x_hi]
    return (min(ys), max(ys)) if ys else None


def _ink_gap_in_overlap(fig, r, ti, a, tj, b) -> float:
    ""
    x_lo, x_hi = max(a.x0, b.x0), min(a.x1, b.x1)
    sa, sb = _ink_span(fig, r, ti, a, x_lo, x_hi), _ink_span(fig, r, tj, b, x_lo, x_hi)
    if sa is None or sb is None:
        return float("inf")
    return max(sa[0], sb[0]) - min(sa[1], sb[1])


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_line_boxes(fig, name)
    if INK_CLEAR_PT is None:
        return fails, texts
    from matplotlib.text import Text
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    cache, boxes = {}, []
    for t in texts:
        try:
            bb = Text.get_window_extent(t, r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb, _ink_extent(fig, r, t, bb, cache)))
    kept = [f for f in fails if " overlaps text " not in f]
    for i in range(len(boxes)):
        ti, a, (a0, a1, one_a) = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, b, (b0, b1, one_b) = boxes[j]
            if min(a.x1, b.x1) - max(a.x0, b.x0) <= 0.7 * px:
                continue
            if one_a and one_b:
                gap = max(a0, b0) - min(a1, b1)
                if gap < INK_CLEAR_PT * px:
                    gap = _ink_gap_in_overlap(fig, r, ti, a, tj, b)
                if gap < INK_CLEAR_PT * px:
                    kept.append(f"FAIL {name}: text {ti.get_text()!r} stands {gap / px:.2f} pt from text "
                                f"{tj.get_text()!r}, under the {INK_CLEAR_PT} pt ink clearance")
            elif min(a.y1, b.y1) - max(a.y0, b.y0) > 0.7 * px:
                kept.append(f"FAIL {name}: text {ti.get_text()!r} overlaps text {tj.get_text()!r}")
    return kept, texts


__all__ += ["INK_CLEAR_PT"]


MIN_ARM_PT = None
ARMS = {}
_ARM_FINDINGS = []


def _arm_findings(name: str, rows: list) -> list:
    ""
    ARMS[name] = rows
    return [f"FAIL {name}: arrow into ({r['x']:.1f}, {r['y']:.1f}) pt: entering shaft {r['shaft']:.1f} pt behind a "
            f"{r['head']:.1f} pt head, under the {MIN_ARM_PT} pt minimum"
            for r in rows if r["shaft"] < MIN_ARM_PT - 1e-6]


def fig_arms(fig) -> list:
    ""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.text import Annotation
    patches = [p for p in fig.findobj(FancyArrowPatch)]
    patches += [a.arrow_patch for a in fig.findobj(Annotation) if a.arrow_patch is not None]
    px, H, rows, seen = fig.dpi / 72.0, fig.bbox.height, [], set()
    for p in patches:
        if id(p) in seen or not p.get_visible():
            continue
        seen.add(id(p))
        style = p.get_arrowstyle()
        spec = getattr(style, "arrow", None)
        if spec is None:
            begin, end = False, hasattr(style, "head_length")
        else:
            begin, end = spec.startswith("<"), spec.endswith(">")
        pos = getattr(p, "_posA_posB", None)
        if not (begin or end) or pos is None:
            continue
        try:
            (ax_, ay_), (bx_, by_) = p.get_transform().transform([tuple(map(float, pos[0])), tuple(map(float, pos[1]))])
        except Exception:
            continue
        head = float(getattr(style, "head_length", 0.4)) * p.get_mutation_scale()
        run = math.hypot(bx_ - ax_, by_ - ay_) / px - float(p.shrinkA) - float(p.shrinkB)
        shaft = max(0.0, run - head * (int(begin) + int(end)))
        cs = p.get_connectionstyle()
        straight = type(cs).__name__ == "Arc3" and not getattr(cs, "rad", 0.0)
        for on, (tx, ty) in ((begin, (ax_, ay_)), (end, (bx_, by_))):
            if on:
                rows.append({"x": tx / px, "y": (H - ty) / px, "run": run, "head": head, "shaft": shaft,
                             "straight": bool(straight)})
    return rows


def diagram_arms(diagram) -> list:
    ""
    svg = diagram.render(embed_fonts=False)
    from sciviz._arrow_check import arrow_ends
    w = re.search(r'<svg\b[^>]*?\bwidth="([0-9.]+)pt"', svg)
    vb = re.search(r'viewBox="[0-9.\-]+ [0-9.\-]+ ([0-9.]+) ', svg)
    scale = float(w.group(1)) / float(vb.group(1)) if (w and vb and float(vb.group(1)) > 0) else 1.0
    return [{"x": e.tip[0], "y": e.tip[1], "run": e.run, "head": e.head, "shaft": e.shaft, "straight": e.straight}
            for e in arrow_ends(svg, scale=scale)]


_check_fig_before_arms = check_fig


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_before_arms(fig, name)
    if MIN_ARM_PT is not None:
        fails = list(fails) + _arm_findings(name, fig_arms(fig))
    return fails, texts


_report_before_arms = _report


def _report(name: str, pdf: Path, fails: list, min_text):
    ""
    extra = list(_ARM_FINDINGS)
    del _ARM_FINDINGS[:]
    return _report_before_arms(name, pdf, list(fails) + extra, min_text)


_save_diagram_before_arms = save_diagram


def save_diagram(diagram, name: str, *, mock_page: bool = True):
    ""
    del _ARM_FINDINGS[:]
    if MIN_ARM_PT is not None:
        _ARM_FINDINGS.extend(_arm_findings(name, diagram_arms(diagram)))
    return _save_diagram_before_arms(diagram, name, mock_page=mock_page)


__all__ += ["MIN_ARM_PT", "ARMS", "fig_arms", "diagram_arms"]


JOIN = {"lw": 1.0, "radius": 2.6, "head": 3.6, "shaft": 10.1, "tip_gap": 1.0}
JOIN_TIP = 0.1 * JOIN["head"]
JOIN_MIN_SHAFT = JOIN["shaft"] + JOIN["lw"] / 2.0 - JOIN_TIP

JOIN_ARM = JOIN["head"] + JOIN_MIN_SHAFT
JOIN_PAD = JOIN_TIP + JOIN["tip_gap"]
JOIN_TEST_PT = 10.0 + JOIN["lw"] / 2.0


def join_theme(theme=None):
    ""
    t = THEME if theme is None else theme
    t.connector = float(JOIN["lw"])
    t.arrow_size = float(JOIN["head"]) / float(JOIN["lw"])
    return t


def join_arrow(ax, starts, x_join, x_end, y, *, color=None, zorder=3):
    ""
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path as MPath
    from sciviz.connect.trunk import round_corners
    color = color or C["ink2"]
    lw, r, head = float(JOIN["lw"]), float(JOIN["radius"]), float(JOIN["head"])
    code = {"M": [MPath.MOVETO], "L": [MPath.LINETO], "Q": [MPath.CURVE3, MPath.CURVE3]}
    for x0, y0 in starts:
        verts, codes = [], []
        for cmd in round_corners([(x0, y0), (x_join, y0), (x_join, y)], r):
            verts += list(cmd[1:])
            codes += code[cmd[0]]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=color, linewidth=lw, zorder=zorder,
                               capstyle="butt", joinstyle="round"))
    s = 1.0 if x_end >= x_join else -1.0
    base, tip = x_end - s * 0.9 * head, x_end + s * 0.1 * head
    ax.plot([x_join, tip - s * lw], [y, y], color=color, linewidth=lw, solid_capstyle="butt", zorder=zorder)
    ax.add_patch(Polygon([(base, y - head / 2.0), (tip, y), (base, y + head / 2.0)], closed=True, facecolor=color,
                         edgecolor="none", linewidth=0, zorder=zorder))
    fig = ax.figure
    if not hasattr(fig, "_kit_join_arms"):
        fig._kit_join_arms = []
    fig._kit_join_arms.append((ax, float(x_join), float(tip), float(y), head))
    return tip


_fig_arms_before_join = fig_arms


def fig_arms(fig) -> list:
    ""
    rows = _fig_arms_before_join(fig)
    px, H = fig.dpi / 72.0, fig.bbox.height
    for ax, x_join, tip, y, head in getattr(fig, "_kit_join_arms", []):
        (jx, _), (tx, ty) = ax.transData.transform([(x_join, y), (tip, y)])
        run = abs(tx - jx) / px
        rows.append({"x": tx / px, "y": (H - ty) / px, "run": run, "head": head, "shaft": max(0.0, run - head),
                     "straight": True})
    return rows


__all__ += ["JOIN", "JOIN_ARM", "JOIN_TIP", "JOIN_PAD", "JOIN_MIN_SHAFT", "JOIN_TEST_PT", "join_theme", "join_arrow"]


LOCAL_NUMBERS = None
_LOCAL_FIELDS = {"val": "defnum", "nv": "defnum", "valint": "defnum", "valn": "defbare", "valu": "defunit"}


def local_numbers() -> dict:
    ""
    out = {}
    text = ""
    for m in re.finditer(r"\\(defnum|defbare|defunit)(?=\{)", text):
        got = _brace_groups(text, m.end(), 2)
        if got is not None:
            (key, printed), _ = got
            out.setdefault(key, {})[m.group(1)] = printed
    return out


def local_forms(text: str):
    ""
    E, L, used = _ledger_entries(), local_numbers(), []

    def one(m):
        macro, key = m.group(1), m.group(2)
        field = _LOCAL_FIELDS.get(macro)
        if key in E or field is None or field not in L.get(key, {}):
            return m.group(0)
        used.append(key)
        return L[key][field]

    return re.sub(r"\\(" + "|".join(sorted(_LOCAL_FIELDS, key=len, reverse=True)) + r")\{([^{}]*)\}", one,
                  text or ""), used


_mock_caption_before_local = mock_caption


def mock_caption(text: str):
    ""
    return _mock_caption_before_local(local_forms(placeholder_forms(text))[0])


_caption_keys_before_local = caption_keys


def caption_keys(name: str) -> list:
    ""
    L = local_numbers()
    rows = _caption_keys_before_local(name)
    for r in rows:
        if r["key"] not in _ledger_entries() and "defnum" in L.get(r["key"], {}):
            r["value"] = L[r["key"]]["defnum"]
            r["source"] = f"draft-local: {LOCAL_NUMBERS.relative_to(PAPER)}"
    return rows


__all__ += ["LOCAL_NUMBERS", "local_numbers", "local_forms"]


class _AllDrawn:
    ""
    VALUE, PENDING = "value", "pending"
    DATA_FAMILY = "data file"

    def __init__(self):
        self.ASKED = []

    def ask(self, row: dict) -> dict:
        row["family"], row["state"] = self.DATA_FAMILY, self.VALUE
        row.pop("run", None)
        self.ASKED.append((row["family"], row["state"], str(row.get("key") or ""), str(row.get("what") or "")[:80]))
        return row

    def stamped(self, row: dict) -> bool:
        return row.get("state") in (self.VALUE, self.PENDING) and bool(row.get("family"))

    def switchable(self, fid: str) -> bool:
        return False

    def family(self, fid: str) -> dict:
        return {"id": fid, "state": "release"}

    def release_values_drawn(self) -> bool:
        return False

    def check(self, rows) -> list:
        return [f"row {i} ({str(r.get('what'))[:60]!r}) draws a value without asking" for i, r in enumerate(rows)
                if not self.stamped(r)]

    def drawn_together(self, rows: list) -> None:
        return None


FAMILY = _AllDrawn()
PENDING_WORD = "Pending"
PENDING_KEY_WORDS = "Release version, pending"
PENDING_STYLE = {"ink": C["gray"], "edge": C["gray"], "fill": C["white"], "dash": (0, (2.6, 1.6)), "lw": 0.8,
                 "hatch": "//////", "tag_fs": FS["tick"], "glyph_pt": 4.2}


PENDING_MARK_PT = 8.0
_MARKS = {}
_KEY_AT = {}


def ask(row: dict) -> dict:
    ""
    return FAMILY.ask(row)


def is_pending(row: dict) -> bool:
    return (row.get("state") if FAMILY.stamped(row) else ask(row)["state"]) == FAMILY.PENDING


def release_values_drawn() -> bool:
    ""
    return FAMILY.release_values_drawn()


def _value_artists(fig) -> list:
    ""
    out = []
    for ax in fig.axes:
        out += list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.texts) + list(ax.artists) \
            + list(ax.images)
    return out + list(fig.lines) + list(fig.patches) + list(fig.texts) + list(fig.artists) + list(fig.images)


class marks:
    ""

    def __init__(self, fig, ref, kind=None, series=None, tag_at=None, block=None, tag_size=None):
        self.fig, self.ref, self.kind, self.series, self.tag_at = fig, ref, kind, series, tag_at
        self.block = block
        self.tag_size = tag_size

    def __enter__(self):
        self.before = {id(a) for a in _value_artists(self.fig)}
        return self

    def __exit__(self, *exc):
        new = [a for a in _value_artists(self.fig) if id(a) not in self.before]
        add_marks(self.fig, self.ref, new, kind=self.kind, series=self.series, tag_at=self.tag_at, block=self.block,
                  tag_size=self.tag_size)
        return False


def add_marks(fig, ref, artists, *, kind=None, series=None, tag_at=None, block=None, tag_size=None):
    ""
    m = _MARKS.get((id(fig), ref))
    if m is None or m["fig"] is not fig:
        m = _MARKS[(id(fig), ref)] = {"fig": fig, "artists": [], "kind": None, "series": None, "tag_at": None,
                                      "block": None}
    m["artists"] += [a for a in artists if a is not None]
    m["kind"] = kind or m["kind"]
    m["series"] = series or m["series"]
    m["tag_at"] = tag_at or m["tag_at"]
    m["block"] = block or m.get("block")
    m["tag_size"] = tag_size or m.get("tag_size")
    return m


def relabel(fig, ref, artist, text):
    ""
    m = add_marks(fig, ref, [])
    m.setdefault("relabel", []).append((artist, text))
    return artist


def pending_key_at(fig, x, y, ha="right", *, ax=None, words=None, anchor=None, size=None, x_in=None):
    ""
    _KEY_AT[id(fig)] = (x, y, ha, ax, words, anchor, size, x_in)


def _kind_of(artists) -> str:
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    kinds = []
    for a in artists:
        if isinstance(a, Line2D):
            kinds.append("point" if a.get_marker() not in (None, "", "None", " ", "none") else "line")
        elif isinstance(a, PathCollection):
            kinds.append("point")
        elif isinstance(a, LineCollection):
            kinds.append("range")
        elif isinstance(a, Rectangle):
            kinds.append("bar")
        elif isinstance(a, Text):
            kinds.append("label")
        else:
            kinds.append("bar")
    for k in ("point", "bar", "range", "line", "label"):
        if k in kinds:
            return k
    return "label"


def pending_glyph(kind: str, size_pt: float = None):
    ""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Rectangle
    g = size_pt or PENDING_STYLE["glyph_pt"]
    st = PENDING_STYLE
    if kind == "label":
        return None
    w = {"point": g, "bar": g * 1.9, "range": g * 2.2, "line": g * 2.4, "cell": g}[kind]
    da = DrawingArea(w, g, 0, 0)
    if kind == "point":
        da.add_artist(Line2D([w / 2], [g / 2], marker="o", markersize=g, markerfacecolor=st["fill"],
                             markeredgecolor=st["edge"], markeredgewidth=0.9, linestyle="none"))
    elif kind == "bar":
        da.add_artist(Rectangle((0.4, g * 0.18), w - 0.8, g * 0.64, facecolor=st["fill"], edgecolor=st["edge"],
                                linewidth=st["lw"]))
    elif kind == "range":
        da.add_artist(Line2D([0.4, w - 0.4], [g / 2, g / 2], color=st["edge"], linewidth=st["lw"],
                             linestyle=st["dash"]))
        for x in (0.4, w - 0.4):
            da.add_artist(Line2D([x, x], [g * 0.2, g * 0.8], color=st["edge"], linewidth=st["lw"]))
    elif kind == "line":
        da.add_artist(Line2D([0, w], [g / 2, g / 2], color=st["edge"], linewidth=1.1, linestyle=st["dash"]))
    elif kind == "cell":
        da.add_artist(Rectangle((0.3, 0.3), g - 0.6, g - 0.6, facecolor=st["fill"], edgecolor=st["edge"],
                                linewidth=0.6, hatch=st["hatch"]))
    return da


def pending_tag(fig, x_frac, y_frac, kind="point", *, words=None, size=None, ground=True):
    ""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    st = PENDING_STYLE
    text = TextArea(words or PENDING_WORD, textprops=dict(color=st["ink"], fontsize=size or st["tag_fs"]))
    glyph = pending_glyph(kind)
    box = HPacker(children=[glyph, text] if glyph is not None else [text], align="center", pad=0, sep=2.2)
    ab = AnnotationBbox(box, (x_frac, y_frac), xycoords="figure fraction", box_alignment=(0.5, 0.5), frameon=ground,
                        pad=0.25, annotation_clip=False, zorder=8,
                        bboxprops=dict(facecolor=st["fill"], edgecolor="none", boxstyle="square,pad=0.25"))
    fig.add_artist(ab)
    return ab


def _gray(a, kind):
    ""
    from matplotlib.collections import Collection, LineCollection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox
    from matplotlib.patches import FancyArrowPatch, Patch
    from matplotlib.text import Annotation, Text
    st = PENDING_STYLE
    if isinstance(a, Line2D):
        if a.get_marker() not in (None, "", "None", " ", "none"):
            a.set_markerfacecolor(st["fill"])
            a.set_markeredgecolor(st["edge"])
            if a.get_markeredgewidth() < 0.5:
                a.set_markeredgewidth(0.8)
        if a.get_linestyle() not in ("None", "none", "", " "):
            a.set_color(st["edge"])
            a.set_linestyle(st["dash"])
            if a.get_marker() not in (None, "", "None", " ", "none"):
                a.set_markeredgecolor(st["edge"])
        a.set_alpha(None)
    elif isinstance(a, Annotation):
        a.set_color(st["ink"])
        a.set_fontweight("normal")
        if a.arrow_patch is not None:
            a.arrow_patch.set_color(st["edge"])
    elif isinstance(a, Text):
        a.set_color(st["ink"])
        a.set_fontweight("normal")
    elif isinstance(a, FancyArrowPatch):
        a.set_color(st["edge"])
    elif isinstance(a, Patch):
        a.set_alpha(None)
        from matplotlib.patches import Rectangle
        if kind != "cell" and not isinstance(a, Rectangle) and a.get_facecolor()[3] > 0:
            ext = a.get_window_extent()
            size_pt = max(ext.width, ext.height) * 72.0 / a.figure.dpi if a.figure is not None else 0.0
            edged = a.get_edgecolor()[3] > 0 and a.get_linewidth() > 0
            if size_pt < PENDING_MARK_PT or edged:
                a.set_facecolor(st["fill"])
                a.set_edgecolor(st["edge"])
                a.set_linewidth(max(a.get_linewidth(), 0.8 if size_pt < PENDING_MARK_PT else 0.6))
                return
            a.set_facecolor(tint(C["gray"], 0.88))
            a.set_edgecolor("none")
            return
        if kind == "cell":
            a.set_facecolor(st["fill"])
            a.set_edgecolor(st["edge"])
            a.set_hatch(st["hatch"])
            a.set_linewidth(0.6)
        else:
            fc = a.get_facecolor()
            a.set_facecolor(st["fill"] if fc[3] > 0 else "none")
            a.set_edgecolor(st["edge"])
            a.set_linewidth(max(a.get_linewidth(), 0.6))
    elif isinstance(a, PathCollection):
        cloud = len(a.get_offsets()) > 50
        a.set_alpha(None)
        a.set_facecolors("none")
        a.set_edgecolors(C["gray2"] if cloud else st["edge"])
        a.set_linewidths(0.2 if cloud else 0.35)
    elif isinstance(a, LineCollection):
        a.set_alpha(None)
        a.set_color(st["edge"])
        a.set_linestyle(st["dash"])
    elif isinstance(a, Collection):
        a.set_alpha(None)
        a.set_facecolor(st["fill"])
        a.set_edgecolor(st["edge"])
    elif isinstance(a, AnnotationBbox):
        for c in a.get_children():
            _gray(c, kind)
        for t in a.findobj(Text):
            if t is not a:
                _gray(t, kind)


def _union_bbox(fig, artists):
    ""
    import numpy as np
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    boxes = []
    for a in artists:
        try:
            if isinstance(a, Collection) and len(a.get_offsets()):
                pts = a.get_offset_transform().transform(np.asarray(a.get_offsets(), dtype=float))
                bb = Bbox.from_extents(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            elif isinstance(a, Line2D):
                xy = a.get_transform().transform(np.asarray(a.get_xydata(), dtype=float))
                xy = xy[np.all(np.isfinite(xy), axis=1)]
                if not len(xy):
                    continue
                bb = Bbox.from_extents(xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
            else:
                bb = a.get_window_extent(r)
        except Exception:
            continue
        if bb is None or not all(map(math.isfinite, bb.extents)) or bb.width < 0 or bb.height < 0:
            continue
        ax = getattr(a, "axes", None)
        if ax is not None and a.get_clip_on():
            pb = ax.get_window_extent()
            bb = Bbox.from_extents(max(bb.x0, pb.x0), max(bb.y0, pb.y0), min(bb.x1, pb.x1), min(bb.y1, pb.y1))
            if bb.width < 0 or bb.height < 0:
                continue
        boxes.append(bb)
    if not boxes:
        return None
    return Bbox.union(boxes)


def apply_pending(fig, rows, name="figure") -> list:
    ""
    for r in rows:
        if not FAMILY.stamped(r):
            ask(r)
    fails = []
    mine = {ref: m for (_f, ref), m in _MARKS.items() if m["fig"] is fig}
    by_ref = {}
    for r in rows:
        if "ref" in r:
            if r["ref"] is not None:
                by_ref.setdefault(r["ref"], []).append(r)
        elif FAMILY.switchable(r["family"]):
            fails.append(f"FAIL {name}: the value of {str(r.get('what'))[:70]!r} ({r['family']}) is drawn by no "
                         f"registered marks, so its pending form cannot be drawn")
    for ref, rs in by_ref.items():
        if ref not in mine or not (mine[ref]["artists"] or mine[ref].get("relabel")):
            fails.append(f"FAIL {name}: the marks {ref!r} of {len(rs)} row(s) registered no artist")
        if len({r["state"] for r in rs}) > 1:
            FAMILY.drawn_together(rs)
        if len({r["state"] for r in rs}) > 1:
            fails += _kept_shared(name, f"marks {ref!r}", rs)
            print(f"NOTE {name}: the marks {ref!r} draw values of " + ", ".join(sorted({f"{r['family']} "
                  f"({r['state']})" for r in rs})) + ": drawn pending, since one of their values is")
            _stamp_shared(rs)
    for ref in mine:
        if ref not in by_ref:
            fails.append(f"FAIL {name}: the marks {ref!r} draw no row's value")
    pend = [ref for ref, rs in by_ref.items() if ref in mine and any(r["state"] == FAMILY.PENDING for r in rs)]
    fams = sorted({r["family"] for r in rows if r["state"] == FAMILY.PENDING})
    gray = release_values_drawn()
    if not pend:
        print(f"{name}: family census: {len(rows)} rows asked, none pending")
        return fails
    fig.canvas.draw()
    for ax in fig.axes:
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    form = "gray" if gray else "tag"
    for ref in pend:
        for artist, text in mine[ref].get("relabel", []):
            _relabel_now(fig, artist, text, gray)
    for r in rows:
        if r["state"] == FAMILY.PENDING:
            r["drawn"] = "the release value in the gray pending form" if gray else "the pending tag, no value"
    tags = 0
    if gray:
        for ref in pend:
            m = mine[ref]
            kind = m["kind"] or _kind_of(m["artists"])
            for a in m["artists"]:
                _gray(a, kind)
        W_, H_ = fig.bbox.width * 72.0 / fig.dpi, fig.bbox.height * 72.0 / fig.dpi
        x_pt, y_pt, ha, k_ax, k_words, k_anchor, k_size, k_x_in = _KEY_AT.get(
            id(fig), (W_ - 1.0, H_ - 6.0, "right", None, None, None, None, None))
        if k_anchor is not None and k_x_in == "points":
            bb = k_anchor.get_window_extent(fig.canvas.get_renderer())
            y_pt = (bb.y0 + bb.y1) / 2.0 * 72.0 / fig.dpi + y_pt
        elif k_ax is not None and k_anchor is not None:
            bb = k_anchor.get_window_extent(fig.canvas.get_renderer())
            x_pt = k_ax.transAxes.transform((x_pt, 0.0))[0] * 72.0 / fig.dpi
            y_pt = (bb.y0 + bb.y1) / 2.0 * 72.0 / fig.dpi + y_pt
        elif k_ax is not None:
            dx, dy = k_ax.transAxes.transform((x_pt, y_pt))
            x_pt, y_pt = dx * 72.0 / fig.dpi, dy * 72.0 / fig.dpi
        elif k_anchor is not None:
            bb = k_anchor.get_window_extent(fig.canvas.get_renderer())
            x_pt, y_pt = bb.x0 * 72.0 / fig.dpi + x_pt, (bb.y0 + bb.y1) / 2.0 * 72.0 / fig.dpi + y_pt
        from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
        kinds = sorted({mine[ref]["kind"] or _kind_of(mine[ref]["artists"]) for ref in pend} - {"label"})
        glyph = pending_glyph(kinds[0] if len(kinds) == 1 else "point")
        text = TextArea(k_words or PENDING_KEY_WORDS, textprops=dict(color=PENDING_STYLE["ink"],
                                                                      fontsize=k_size or FS["small"], linespacing=1.15))
        box = HPacker(children=[glyph, text] if glyph is not None else [text], align="center", pad=0, sep=2.2)
        fig.add_artist(AnnotationBbox(box, (x_pt / W_, y_pt / H_), xycoords="figure fraction",
                                      box_alignment=({"right": 1.0, "left": 0.0, "center": 0.5}[ha], 0.5),
                                      frameon=False, pad=0.0, annotation_clip=False, zorder=8))
    else:
        groups = {}
        whole_blocks = _whole_blocks(mine, pend)
        for ref in pend:
            m = mine[ref]
            fam = by_ref[ref][0]["family"]
            axs = {id(a.axes) for a in m["artists"] if getattr(a, "axes", None) is not None}
            panel = next((a.axes for a in m["artists"] if getattr(a, "axes", None) is not None), None)
            key = m["series"] or (id(panel) if len(axs) <= 1 else "figure", fam)
            if m.get("block") in whole_blocks:
                key = (_BLOCK, m["block"])
            if not m["artists"]:
                continue
            g = groups.setdefault(key, {"artists": [], "kinds": [], "tag_at": None, "panel": panel, "axes": [],
                                        "refs": []})
            g["refs"].append(ref)
            g["axes"] += [a.axes for a in m["artists"] if getattr(a, "axes", None) is not None
                          and all(a.axes is not b for b in g["axes"])]
            g["artists"] += m["artists"]
            g["kinds"].append(m["kind"] or _kind_of(m["artists"]))
            g["tag_at"] = g["tag_at"] or m["tag_at"]
            g["tag_size"] = g.get("tag_size") or m.get("tag_size")
        boxes = {k: _union_bbox(fig, g["artists"]) for k, g in groups.items()}

        whole = set()
        for ax in fig.axes:
            refs = [ref for ref, m in mine.items() if any(getattr(a, "axes", None) is ax for a in m["artists"])]
            if refs and all(ref in pend for ref in refs):
                whole.add(id(ax))
        for g in groups.values():
            for a in g["artists"]:
                if not getattr(a, "_pending_tag", False):
                    a.set_visible(False)
        _unanchor(fig, set(pend))
        inv = fig.transFigure.inverted()
        obstacles = _obstacles(fig, [m for ref, m in mine.items() if ref not in pend])
        tagged_axes = set()
        for k, g in groups.items():
            kinds = [x for x in g["kinds"] if x != "label"] or ["label"]
            kind = max(set(kinds), key=kinds.count)
            ground = kind != "label"
            if all(_tagged_in_place(mine[ref]) for ref in g["refs"]):
                continue
            if (g["tag_at"] is None and not (isinstance(k, tuple) and k[0] == _BLOCK) and len(g["axes"]) > 1
                    and all(id(a) in whole for a in g["axes"])
                    and all(a.get_window_extent().width >= 1.1 * _tag_size(fig, kind)[0] for a in g["axes"])):
                for a in g["axes"]:
                    if id(a) in tagged_axes:
                        continue
                    tagged_axes.add(id(a))
                    pb = a.get_window_extent()
                    size = _tag_size(fig, kind)
                    disp = _free_spot(fig, ((pb.x0 + pb.x1) / 2.0, (pb.y0 + pb.y1) / 2.0), pb, size, obstacles)
                    fx, fy = inv.transform(disp)
                    ab = pending_tag(fig, fx, fy, kind, ground=ground)
                    obstacles.append(ab.get_window_extent(fig.canvas.get_renderer()))
                    tags += 1
                continue
            if g["tag_at"] is None and kind == "label" and boxes[k] is not None:
                bb = boxes[k]
                g["tag_at"] = (None, (bb.x0 + bb.x1) / 2.0 * 72.0 / fig.dpi, (bb.y0 + bb.y1) / 2.0 * 72.0 / fig.dpi,
                               "points")
            if g["tag_at"] is not None:
                ax, x, y, coords = g["tag_at"]
                if coords == "data":
                    disp = ax.transData.transform((x, y))
                elif coords == "axes":
                    disp = ax.transAxes.transform((x, y))
                else:
                    disp = (x * fig.dpi / 72.0, y * fig.dpi / 72.0)
            else:
                bb = boxes[k]
                if bb is None:
                    continue
                disp = ((bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0)
                region = bb
                if isinstance(k, tuple) and k[0] == _BLOCK:
                    region = _block_region(fig, k[1], bb)
                    disp = ((region.x0 + region.x1) / 2.0, (region.y0 + region.y1) / 2.0)
                    pb = region = region.padded(0.0, _tag_size(fig, kind)[1] / 2.0)
                elif g["axes"]:
                    from matplotlib.transforms import Bbox
                    pb = Bbox.union([a.get_window_extent() for a in g["axes"]])
                    if all(id(a) in whole for a in g["axes"]):
                        disp, region = ((pb.x0 + pb.x1) / 2.0, (pb.y0 + pb.y1) / 2.0), pb
                    disp = (min(max(disp[0], pb.x0 + 0.15 * pb.width), pb.x1 - 0.15 * pb.width),
                            min(max(disp[1], pb.y0 + 0.1 * pb.height), pb.y1 - 0.1 * pb.height))
                size = _tag_size(fig, kind)
                spot = _free_spot(fig, disp, region, size, obstacles)
                if spot == disp and g["axes"] and not _free_spot(fig, disp, region, size, obstacles, test=True):
                    spot = _free_spot(fig, disp, pb.padded(size[0] / 2.0, size[1] / 2.0), size, obstacles)
                disp = spot
            fx, fy = inv.transform(disp)
            ab = pending_tag(fig, fx, fy, kind, ground=ground, size=g.get("tag_size"))
            obstacles.append(ab.get_window_extent(fig.canvas.get_renderer()))
            tags += 1
    print(f"{name}: family census: {len(rows)} rows asked, {sum(r['state'] == FAMILY.PENDING for r in rows)} pending "
          f"({', '.join(fams)}), form {form}, {len(pend)} groups of marks, {tags} tag(s)")
    return fails


def _tag_size(fig, kind):
    ""
    px = fig.dpi / 72.0
    w = text_width_pt(fig, PENDING_WORD, PENDING_STYLE["tag_fs"]) + 2.2
    g = PENDING_STYLE["glyph_pt"]
    w += {"label": -2.2, "point": g, "bar": g * 1.9, "range": g * 2.2, "line": g * 2.4, "cell": g}[kind]
    return (w + 2 * 0.25 * PENDING_STYLE["tag_fs"]) * px, (1.25 * PENDING_STYLE["tag_fs"]) * px


def _obstacles(fig, kept):
    ""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and (t.axes is None or t.axes.get_visible()):
            try:
                out.append(Text.get_window_extent(t, r))
            except Exception:
                pass
    for m in kept:
        bb = _union_bbox(fig, [a for a in m["artists"] if a.get_visible()])
        if bb is not None:
            out.append(bb)
    return out


def _free_spot(fig, want, region, size, obstacles, clear_pt=1.5, test=False):
    ""
    from matplotlib.transforms import Bbox
    w, h = size
    c = clear_pt * fig.dpi / 72.0

    def free(x, y):
        box = Bbox.from_extents(x - w / 2 - c, y - h / 2 - c, x + w / 2 + c, y + h / 2 + c)
        return not any(box.overlaps(o) for o in obstacles if o.width > 0 and o.height > 0)

    if test:
        return free(*want)
    if free(*want):
        return want
    x0, x1 = region.x0 + w / 2, region.x1 - w / 2
    y0, y1 = region.y0 + h / 2, region.y1 - h / 2
    if x1 < x0:
        x0 = x1 = (region.x0 + region.x1) / 2
    if y1 < y0:
        y0 = y1 = (region.y0 + region.y1) / 2
    n = 24
    spots = [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * j / n) for i in range(n + 1) for j in range(n + 1)]
    spots.sort(key=lambda p: (p[0] - want[0]) ** 2 + (p[1] - want[1]) ** 2)
    return next((p for p in spots if free(*p)), want)


_save_fig_before_pending = save_fig


_save_keys_before_pending = save_keys


def save_keys(name: str, rows: list):
    ""


    shared = "hidden_with"
    asked = [dict(r, state=FAMILY.VALUE) if (r.get(shared) and r.get("state") == FAMILY.PENDING) else r for r in rows]
    for f in FAMILY.check(asked):
        print(f"FAIL {name}: {f}")
    out = [{k: v for k, v in r.items() if k != "ref"} for r in rows]
    return _save_keys_before_pending(name, out)


__all__ += ["FAMILY", "PENDING_WORD", "PENDING_KEY_WORDS", "PENDING_STYLE", "ask", "is_pending",
            "release_values_drawn", "marks", "add_marks", "relabel", "pending_key_at", "pending_glyph", "pending_tag",
            "apply_pending"]


_CENSUS_FAILS = {}
_check_fig_before_pending = check_fig


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_before_pending(fig, name)
    return fails + _CENSUS_FAILS.pop(id(fig), []), texts


def save_fig(fig, name: str, *, mock_page: bool = True, rows=None):
    ""
    if rows is not None:
        _CENSUS_FAILS[id(fig)] = apply_pending(fig, rows, name)
    return _save_fig_before_pending(fig, name, mock_page=mock_page)


_save_diagram_before_pending = save_diagram


def save_diagram(diagram, name: str, *, mock_page: bool = True, rows=None):
    ""
    if rows is not None:
        for r in rows:
            if not FAMILY.stamped(r):
                ask(r)
            if "ref" not in r and FAMILY.switchable(r["family"]):
                print(f"FAIL {name}: the value of {str(r.get('what'))[:70]!r} ({r['family']}) is drawn by no named "
                      f"elements, so its pending form cannot be drawn")
        by_ref = {}
        for r in rows:
            if r.get("ref") is not None:
                by_ref.setdefault(r["ref"], []).append(r)
        for ref, rs in by_ref.items():
            if len({r["state"] for r in rs}) > 1:
                FAMILY.drawn_together(rs)
            if len({r["state"] for r in rs}) > 1:
                for f in _kept_shared(name, f"elements {ref!r}", rs):
                    print(f)
                print(f"NOTE {name}: the elements {ref!r} draw values of " + ", ".join(sorted({f"{r['family']} "
                      f"({r['state']})" for r in rs})) + ": drawn pending, since one of their values is")
                _stamp_shared(rs)
        n = sum(r["state"] == FAMILY.PENDING for r in rows)
        print(f"{name}: family census: {len(rows)} rows asked, {n} pending"
              + (f" ({', '.join(sorted({r['family'] for r in rows if r['state'] == FAMILY.PENDING}))}), form "
                 f"{'gray' if release_values_drawn() else 'tag'}" if n else ""))
    return _save_diagram_before_pending(diagram, name, mock_page=mock_page)


_caption_keys_before_pending = caption_keys


def caption_keys(name: str) -> list:
    ""
    return [dict(r, ref=None) for r in _caption_keys_before_pending(name)]


def sv_blank(element, stand_in=None):
    ""
    from sciviz.elements._blank import Blank
    return Blank(element, stand_in=stand_in)


def sv_tag(kind="point", *, size=None, width=None, height=None, align="center"):
    ""
    from sciviz import Box, FixedSize, Row, Text
    st = PENDING_STYLE
    fs = size or st["tag_fs"]
    g = st["glyph_pt"]
    glyph = {"point": lambda: Box(width=g, height=g, radius=g / 2.0, fill=st["fill"], stroke=st["edge"],
                                  stroke_width=0.9),
             "bar": lambda: Box(width=1.9 * g, height=0.64 * g, fill=st["fill"], stroke=st["edge"],
                                stroke_width=st["lw"]),
             "cell": lambda: Box(width=g, height=g, fill=st["fill"], stroke=st["edge"], stroke_width=0.6)}.get(kind)
    word = Text(PENDING_WORD, size=fs, color=st["ink"])
    el = Row(glyph(), word, gap=2.2, align="center") if glyph else word
    if width is not None or height is not None:
        el = FixedSize(el, width=width, height=height, align=align)
    return el


def sv_ink(color, pending: bool):
    ""
    return PENDING_STYLE["ink"] if pending else color


__all__ += ["sv_blank", "sv_tag", "sv_ink"]


PENDING_TOKEN = "{pending}"

SHARED_FIELD = "hidden_with"
_ANCHORED = {}


def _stamp_shared(rows):
    ""
    pend = [r for r in rows if r["state"] == FAMILY.PENDING and not r.get(SHARED_FIELD)]
    if not pend:
        return
    fams = sorted({r["family"] for r in pend})
    for r in rows:
        if r["state"] != FAMILY.PENDING:
            r["state"] = FAMILY.PENDING
            r["run"] = pend[0].get("run")
            r[SHARED_FIELD] = fams


def _relabel_now(fig, artist, text, gray=False):
    ""
    if PENDING_TOKEN not in text:
        artist.set_text(text)
        return
    if gray:
        artist.set_color(PENDING_STYLE["ink"])
        artist.set_fontweight("normal")
        return
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    before, _, after = text.partition(PENDING_TOKEN)
    before, after = before.rstrip(), after.lstrip()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r)
    ha = artist.get_horizontalalignment()
    size = artist.get_fontsize()
    style = dict(color=artist.get_color(), fontsize=size, fontweight=artist.get_fontweight(),
                 fontstyle=artist.get_fontstyle(), fontfamily=artist.get_fontfamily())
    parts = ([TextArea(before, textprops=style)] if before else []) + \
        [TextArea(PENDING_WORD, textprops=dict(style, color=PENDING_STYLE["ink"], fontweight="normal"))] + \
        ([TextArea(after, textprops=style)] if after else [])
    run = HPacker(children=parts, align="baseline", pad=0, sep=0.28 * size)
    x = {"left": bb.x0, "center": (bb.x0 + bb.x1) / 2.0, "right": bb.x1}[ha]
    xy = (x / fig.bbox.width, bb.y0 / fig.bbox.height)
    ab = AnnotationBbox(run, xy, xycoords="figure fraction", box_alignment=({"left": 0.0, "center": 0.5,
                        "right": 1.0}[ha], 0.0), frameon=False, pad=0.0, annotation_clip=False,
                        zorder=artist.get_zorder())
    fig.add_artist(ab)
    artist.set_text("")
    artist._pending_run = ab


def anchored(fig, ref, label, *, leaders=(), column=None, order=None, pitch=None):
    ""
    _ANCHORED.setdefault(id(fig), []).append({"fig": fig, "ref": ref, "label": label, "leaders": list(leaders),
                                              "column": column, "order": order, "pitch": pitch})
    return label


def _unanchor(fig, pend):
    ""
    from matplotlib.text import Annotation
    from matplotlib.transforms import Affine2D, ScaledTranslation
    entries = [e for e in _ANCHORED.get(id(fig), []) if e["fig"] is fig and e["ref"] in pend]
    cols = {}
    for k, e in enumerate(entries):
        for a in e["leaders"]:
            a.set_visible(False)
        if isinstance(e["label"], Annotation) and e["label"].arrow_patch is not None:
            e["label"].arrowprops, e["label"].arrow_patch = None, None
        if e["column"] is None:
            raise ValueError(f"an anchored label of {e['ref']!r} names no column")
        ax, x, y, coords, ha = e["column"]
        cols.setdefault((id(ax), x, y, coords, ha), []).append((e["order"] if e["order"] is not None else k, k, e))
    for (_, x, y, coords, ha), es in cols.items():
        es.sort(key=lambda t: (t[0], t[1]))
        drop = 0.0
        for _, _, e in es:
            t = e["label"]
            ax = e["column"][0]
            base = ax.transAxes if coords == "axes" else Affine2D().scale(1.0 / 72.0) + fig.dpi_scale_trans
            if isinstance(t, Annotation):
                t.xy, t.xycoords = (x, y), base
                t.set_anncoords("offset points")
                t.xyann = (0.0, -drop)
            else:
                t.set_transform(base + ScaledTranslation(0.0, -drop / 72.0, fig.dpi_scale_trans))
                t.set_position((x, y))
            t.set_horizontalalignment(ha)
            t.set_verticalalignment("top")
            drop += e["pitch"] or 1.25 * t.get_fontsize()


def withheld(rows) -> bool:
    ""
    return (not release_values_drawn()) and any(is_pending(dict(r)) for r in rows)


def frame_range(lo, hi, rows, *, scale="linear", step=None):
    ""
    if not withheld(rows):
        return lo, hi
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    if scale in ("log", "symlog"):
        lin = scale == "symlog" and a <= 0
        if b <= 1.0 and lin:
            return lo, hi
        ea, eb = (None if lin else math.ceil(math.log10(a) - 1e-9)), math.floor(math.log10(b) + 1e-9)
        if ea is not None and ea >= eb:
            ea, eb = math.floor(math.log10(a) + 1e-9), math.ceil(math.log10(b) - 1e-9)
        a, b = (a if lin else 10.0 ** ea), 10.0 ** eb
    else:
        if not step:
            raise ValueError("frame_range: a linear range needs its step")
        ka, kb = math.ceil(a / step - 1e-9), math.floor(b / step + 1e-9)
        if ka >= kb:
            ka, kb = math.floor(a / step + 1e-9), math.ceil(b / step - 1e-9)
        a, b = ka * step, kb * step
    return (a, b) if lo <= hi else (b, a)


__all__ += ["PENDING_TOKEN", "anchored", "withheld", "frame_range"]


_BLOCK = "block"
_BLOCKS = {}


CATEGORY_SOURCE = "statement: a declared category of its row"


def block(fig, name, *, across=None):
    ""
    _BLOCKS[(id(fig), name)] = {"fig": fig, "across": across}


def _whole_blocks(mine, pend) -> set:
    ""
    names = {}
    for ref, m in mine.items():
        if m.get("block"):
            names.setdefault(m["block"], []).append(ref)
    return {b for b, refs in names.items() if all(r in pend and mine[r]["artists"] for r in refs)}


def _block_region(fig, name, bb):
    ""
    from matplotlib.transforms import Bbox
    b = _BLOCKS.get((id(fig), name))
    if b is None or b["fig"] is not fig or b["across"] is None:
        return bb
    pb = b["across"].get_window_extent()
    return Bbox.from_extents(pb.x0, bb.y0, pb.x1, bb.y1)


def kept(row) -> bool:
    ""
    if not FAMILY.stamped(row):
        ask(row)
    if row["state"] != FAMILY.VALUE or row.get(SHARED_FIELD):
        return False
    return FAMILY.family(row["family"]).get("state") != "release"


def _kept_shared(name, what, rows) -> list:
    ""
    return [f"FAIL {name}: the {what} draw the kept value {str(r.get('what'))[:60]!r} ({r['family']}) with a pending "
            f"value: a kept value's mark is drawn, so it names marks of its own" for r in rows if kept(r)]


def reach(end, rows, frame_end):
    ""
    return frame_end if withheld(rows) else end


def category(row) -> dict:
    ""
    r = {k: v for k, v in row.items() if k not in ("key", "ref", "family", "state", "run", "drawn")}
    if row.get("key"):
        r["category_of"] = row["key"]
    r["source"] = CATEGORY_SOURCE + (f"; {row['source']}" if row.get("source") else "")
    r["ref"] = None
    return r


__all__ += ["block", "kept", "reach", "category", "CATEGORY_SOURCE"]


def family_forms(text: str) -> str:
    ""
    text = text.replace("\\familyversion{", "\\familyform@v{")
    for macro, n in (("\\familyform@v", 1), ("\\familyform", 4)):
        while True:
            at = text.find(macro + "{")
            if at < 0:
                break
            got = _brace_groups(text, at + len(macro), n)
            if got is None:
                break
            groups, end = got
            state = FAMILY.family(groups[0].strip()).get("state")
            forms = ("release", "paper", "paper") if n == 1 else tuple(groups[1:])
            text = text[:at] + {"release": forms[0], "pending": forms[1]}.get(state, forms[2]) + text[end:]
    return text


_mock_caption_before_family = mock_caption


def mock_caption(text: str):
    ""
    return _mock_caption_before_family(family_forms(text))


__all__ += ["family_forms"]


TAG_RULE = {"length": 5.0, "lw": 0.7, "gap": 1.8}


def _tag_box(size=None, ink=None, words=None):
    ""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import DrawingArea, HPacker, TextArea
    st = PENDING_STYLE
    fs = size or st["tag_fs"]
    k = fs / st["tag_fs"]
    length, lw = TAG_RULE["length"] * k, TAG_RULE["lw"]
    rule = DrawingArea(length, lw, 0, 0)
    rule.add_artist(Line2D([0.0, length], [lw / 2.0, lw / 2.0], color=ink or st["edge"], linewidth=lw,
                           solid_capstyle="butt"))
    word = TextArea(words or PENDING_WORD, textprops=dict(color=ink or st["ink"], fontsize=fs))
    return HPacker(children=[rule, word], align="baseline", pad=0, sep=TAG_RULE["gap"] * k)


def pending_tag(fig, x_frac, y_frac, kind="point", *, words=None, size=None, ground=True):
    ""
    from matplotlib.offsetbox import AnnotationBbox
    st = PENDING_STYLE
    box = _tag_box(size, words=words)
    ab = AnnotationBbox(box, (x_frac, y_frac), xycoords="figure fraction", box_alignment=(0.5, 0.5), frameon=ground,
                        pad=0.25, annotation_clip=False, zorder=8,
                        bboxprops=dict(facecolor=st["fill"], edgecolor="none", boxstyle="square,pad=0.25"))
    fig.add_artist(ab)
    return ab


def _tag_size(fig, kind):
    ""
    px = fig.dpi / 72.0
    fs = PENDING_STYLE["tag_fs"]
    w = TAG_RULE["length"] + TAG_RULE["gap"] + text_width_pt(fig, PENDING_WORD, fs)
    return (w + 2 * 0.25 * fs) * px, (1.25 * fs) * px


def sv_tag(kind="point", *, size=None, width=None, height=None, align="center"):
    ""
    from sciviz import Box, Column, FixedSize, Row, Spacer, Text
    st = PENDING_STYLE
    fs = size or st["tag_fs"]
    k = fs / st["tag_fs"]
    length, lw = TAG_RULE["length"] * k, TAG_RULE["lw"]
    word = Text(PENDING_WORD, size=fs, color=st["ink"])
    h = word.measure(THEME).h
    base = word.first_baseline(THEME)
    rule = Column(Spacer(length, max(base - lw, 0.0)), Box(width=length, height=lw, fill=st["edge"], stroke="none"),
                  Spacer(length, max(h - base, 0.0)), gap=0.0, align="start")
    el = Row(rule, word, gap=TAG_RULE["gap"] * k, align="center")
    if width is not None or height is not None:
        el = FixedSize(el, width=width, height=height, align=align)
    return el


def _accent(color) -> bool:
    ""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return max(r, g, b) - min(r, g, b) > 0.12


def relabel(fig, ref, artist, text, *, as_tag=False, tag_size=None):
    ""
    m = add_marks(fig, ref, [])
    m.setdefault("relabel", []).append((artist, text))
    if as_tag and PENDING_TOKEN in text:
        m["in_place"] = True
    if tag_size is not None:
        artist._tag_size = tag_size
    return artist


def _tagged_in_place(m) -> bool:
    ""
    arts = m.get("artists") or []
    return bool(m.get("in_place")) or (bool(arts) and all(getattr(a, "_pending_tag", False) for a in arts))


def _relabel_now(fig, artist, text, gray=False):
    ""
    if PENDING_TOKEN not in text:
        artist.set_text(text)
        if not gray and _accent(artist.get_color()):
            artist.set_color(PENDING_STYLE["ink"])
            artist.set_fontweight("normal")
        return
    if gray:
        artist.set_color(PENDING_STYLE["ink"])
        artist.set_fontweight("normal")
        return
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    before, _, after = text.partition(PENDING_TOKEN)
    before, after = before.rstrip(), after.lstrip()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r)
    ha = artist.get_horizontalalignment()
    size = artist.get_fontsize()
    quiet = _accent(artist.get_color())
    style = dict(color=PENDING_STYLE["ink"] if quiet else artist.get_color(), fontsize=size,
                 fontweight="normal" if quiet else artist.get_fontweight(), fontstyle=artist.get_fontstyle(),
                 fontfamily=artist.get_fontfamily())
    parts = ([TextArea(before, textprops=style)] if before else []) + [_tag_box(getattr(artist, "_tag_size", size))] + \
        ([TextArea(after, textprops=style)] if after else [])
    run = HPacker(children=parts, align="baseline", pad=0, sep=0.28 * size)
    x = {"left": bb.x0, "center": (bb.x0 + bb.x1) / 2.0, "right": bb.x1}[ha]
    xy = (x / fig.bbox.width, bb.y0 / fig.bbox.height)
    ab = AnnotationBbox(run, xy, xycoords="figure fraction", box_alignment=({"left": 0.0, "center": 0.5,
                        "right": 1.0}[ha], 0.0), frameon=False, pad=0.0, annotation_clip=False,
                        zorder=artist.get_zorder())
    fig.add_artist(ab)
    artist.set_text("")
    artist._pending_run = ab


DESCRIPTOR = {"size": FS["small"], "color": C["ink2"], "asc": 0.76, "desc": 0.293, "air": 1.6, "gap": 1.0}


def _runs(line, size, color):
    if isinstance(line, str):
        line = [(line, {})]
    return [(t, dict(dict(color=color, fontsize=size), **p)) for t, p in line]


def _line_sizes(lines, size):
    return [max(p["fontsize"] for _, p in _runs(ln, size, DESCRIPTOR["color"])) for ln in lines]


def _baselines(lines, size):
    ""
    sz = _line_sizes(lines, size)
    d = DESCRIPTOR
    base = [0.0] * len(sz)
    base[-1] = d["desc"] * sz[-1]
    for i in range(len(sz) - 2, -1, -1):
        base[i] = base[i + 1] + d["desc"] * sz[i] + d["air"] + d["asc"] * sz[i + 1]
    return base, sz


def descriptor_height(fig, lines, *, size=None) -> float:
    ""
    base, sz = _baselines(lines, size or DESCRIPTOR["size"])
    return base[0] + DESCRIPTOR["asc"] * sz[0]


def descriptor(ax, lines, *, x=0.0, gap=None, size=None, color=None, ha="left"):
    ""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    size = size or DESCRIPTOR["size"]
    color = color or DESCRIPTOR["color"]
    gap = DESCRIPTOR["gap"] if gap is None else gap
    base, sz = _baselines(lines, size)
    out = []
    for ln, b, s in zip(lines, base, sz):
        box = HPacker(children=[TextArea(t, textprops=p) for t, p in _runs(ln, size, color)], align="baseline", pad=0,
                      sep=0)
        ab = AnnotationBbox(box, (x, 1.0), xycoords="axes fraction",
                            xybox=(0.0, gap + b - DESCRIPTOR["desc"] * s), boxcoords="offset points",
                            box_alignment=({"left": 0.0, "center": 0.5, "right": 1.0}[ha], 0.0), frameon=False,
                            pad=0.0, annotation_clip=False, zorder=6)
        ax.add_artist(ab)
        out.append(ab)
    return out


def descriptor_x(block, x):
    ""
    for ab in block:
        ab.xy = (x, ab.xy[1])


__all__ += ["TAG_RULE", "relabel", "descriptor", "descriptor_height", "descriptor_x", "DESCRIPTOR"]


def tag_label(ax, x, y, *, transform=None, size=None, ha="left", va="center", zorder=4):
    ""
    from matplotlib.offsetbox import AnnotationBbox
    fs = size or PENDING_STYLE["tag_fs"]
    box = _tag_box(fs)
    vy = {"center": 0.5, "bottom": 0.0, "top": 1.0, "baseline": 0.0}[va]
    ab = AnnotationBbox(box, (x, y), xycoords=transform if transform is not None else ax.transData,
                        xybox=(0.0, -0.24 * fs if va == "baseline" else 0.0), boxcoords="offset points",
                        box_alignment=({"left": 0.0, "center": 0.5, "right": 1.0}[ha], vy), frameon=False, pad=0.0,
                        annotation_clip=False, zorder=zorder)
    ab._pending_tag = True
    ax.add_artist(ab)
    return ab


__all__ += ["tag_label"]


NAME_CENTRE_PT = 0.25
NAME_AIR_PT = 1.0
_AXIS_NAMES = {}


def axis_name(fig, axes, text, *, x, size, color=None, **kw):
    ""
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    w_pt = fig.get_size_inches()[0] * 72.0
    y0 = min(a.get_position().y0 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    r = fig.canvas.get_renderer()
    out, xi = [], x
    for line in text.split("\n"):
        t = fig.text(xi / w_pt, (y0 + y1) / 2.0, line, rotation=90, ha="left", va="center", fontsize=size,
                     color=color or C["ink"], **kw)
        out.append(t)
        xi += t.get_window_extent(r).width * 72.0 / fig.dpi
    _AXIS_NAMES.setdefault(id(fig), []).append((out, axes))
    return out


def name_findings(fig, name: str, texts) -> list:
    ""
    from matplotlib.text import Text
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    out = []
    for lines, axes in _AXIS_NAMES.get(id(fig), []):
        a0 = min(a.get_window_extent(r).y0 for a in axes)
        a1 = max(a.get_window_extent(r).y1 for a in axes)
        for t in lines:
            bb = Text.get_window_extent(t, r)
            off = ((bb.y0 + bb.y1) - (a0 + a1)) / 2.0 / px
            label = t.get_text()
            if abs(off) > NAME_CENTRE_PT:
                out.append(f"FAIL {name}: axis name {label!r} stands {off:+.2f} pt off its axes' centre "
                           f"(at most {NAME_CENTRE_PT} pt)")
            for o in texts:
                if any(o is s_ for s_ in lines) or not o.get_visible():
                    continue
                try:
                    ob = Text.get_window_extent(o, r)
                except Exception:
                    continue
                if ob.width <= 0 or ob.height <= 0 or min(bb.x1, ob.x1) - max(bb.x0, ob.x0) <= 0.7 * px:
                    continue
                gap = max(bb.y0, ob.y0) - min(bb.y1, ob.y1)
                if gap < NAME_AIR_PT * px:
                    out.append(f"FAIL {name}: axis name {label!r} stands {gap / px:.2f} pt from text "
                               f"{o.get_text().replace(chr(10), ' ')!r} (at least {NAME_AIR_PT} pt)")
    return out


_check_fig_before_names = check_fig


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_before_names(fig, name)
    if id(fig) in _AXIS_NAMES:
        fails = list(fails) + name_findings(fig, name, texts)
        del _AXIS_NAMES[id(fig)]
    return fails, texts


__all__ += ["axis_name", "name_findings", "NAME_CENTRE_PT", "NAME_AIR_PT"]


TITLE_RUN_GAP = 6.0
_TITLE_RUNS = {}


def title_run(ax, title, runs, *, x=1.0, color=None, size=None):
    ""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    from matplotlib.text import Text
    size = size or DESCRIPTOR["size"]
    parts = _runs(runs, size, color or DESCRIPTOR["color"])
    areas = [TextArea(t, textprops=p) for t, p in parts]
    box = HPacker(children=areas, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(box, (x, title.get_position()[1]), xycoords=title.get_transform(), xybox=(0.0, 0.0),
                        boxcoords="offset points", box_alignment=(1.0, 0.0), frameon=False, pad=0.0,
                        annotation_clip=False, zorder=6)
    ax.add_artist(ab)
    fig = ax.get_figure()
    desc = -box.get_bbox(fig.canvas.get_renderer()).y0 * 72.0 / fig.dpi
    ab.xybox = (0.0, -desc)
    texts = [t for a in areas for t in a.findobj(Text) if t.get_text()]
    _TITLE_RUNS.setdefault(id(fig), []).append((ab, title, texts))
    return ab


def title_run_findings(fig, name: str) -> list:
    ""
    from matplotlib.text import Text
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    out = []
    for ab, title, texts in _TITLE_RUNS.get(id(fig), []):
        tb = Text.get_window_extent(title, r)
        rb = ab.get_window_extent(r)
        words = "".join(t.get_text() for t in texts)
        gap = (rb.x0 - tb.x1) / px
        if gap < TITLE_RUN_GAP:
            out.append(f"FAIL {name}: the run {words!r} stands {gap:.2f} pt after its title {title.get_text()!r} "
                       f"(at least {TITLE_RUN_GAP} pt)")
        t_base = title.get_transform().transform(title.get_position())[1]
        r_base = texts[0].get_transform().transform((0.0, 0.0))[1]
        if abs(t_base - r_base) / px > 0.1:
            out.append(f"FAIL {name}: the run {words!r} stands {(r_base - t_base) / px:+.2f} pt off its title's "
                       f"baseline")
    return out


_check_fig_before_runs = check_fig


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_before_runs(fig, name)
    if id(fig) in _TITLE_RUNS:
        fails = list(fails) + title_run_findings(fig, name)
        del _TITLE_RUNS[id(fig)]
    return fails, texts


__all__ += ["title_run", "title_run_findings", "TITLE_RUN_GAP"]


MOVED_LABEL_AIR_PT = 6.0
_MOVED = {}
_unanchor_before_moved = _unanchor


def _unanchor(fig, pend):
    ""
    _unanchor_before_moved(fig, pend)
    _MOVED[id(fig)] = [e["label"] for e in _ANCHORED.get(id(fig), []) if e["fig"] is fig and e["ref"] in pend]


def moved_label_findings(fig, name: str) -> list:
    ""
    import numpy as np
    labels = [t for t in _MOVED.get(id(fig), []) if t.figure is fig and t.get_visible()]
    if not labels:
        return []
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    out = []
    for t in labels:
        bb = t.get_window_extent(r)
        for ax in fig.axes:
            for ln in ax.lines:
                if not ln.get_visible() or ln.get_linewidth() <= 0 or ln.get_linestyle() in ("None", "none", " ", ""):
                    continue
                xy = np.asarray(ln.get_transform().transform(np.asarray(ln.get_xydata(), dtype=float)))
                xy = xy[np.all(np.isfinite(xy), axis=1)]
                if len(xy) < 2:
                    continue
                for x, y in (xy[0], xy[-1]):
                    if not bb.x0 <= x <= bb.x1:
                        continue
                    gap = bb.y0 - y if y <= bb.y0 else (y - bb.y1 if y >= bb.y1 else 0.0)
                    if gap < MOVED_LABEL_AIR_PT * px:
                        out.append(f"FAIL {name}: a drawn line ends {gap / px:.1f} pt from the moved label "
                                   f"{t.get_text()!r}, inside its width: the label reads as that line's name")
    return out


_check_fig_before_moved = check_fig


def check_fig(fig, name: str) -> list:
    ""
    fails, texts = _check_fig_before_moved(fig, name)
    if id(fig) in _MOVED:
        fails = list(fails) + moved_label_findings(fig, name)
        del _MOVED[id(fig)]
    return fails, texts


__all__ += ["moved_label_findings", "MOVED_LABEL_AIR_PT"]


LABELS = HERE / "labels.tsv"


@functools.lru_cache(maxsize=1)
def printed_words() -> dict:
    ""
    out = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        old, new = line.split("\t")[:2]
        if old in out:
            raise SystemExit(f"labels.tsv: {old!r} given twice")
        out[old] = new
    return out


def printed(obj):
    ""
    words = printed_words()
    if isinstance(obj, str):
        return words.get(obj, obj)
    if isinstance(obj, list):
        return [printed(v) for v in obj]
    if isinstance(obj, dict):
        return {k: printed(v) for k, v in obj.items()}
    return obj


_load_record_words = load


def load(name: str):
    ""
    return printed(_load_record_words(name))


__all__ += ["LABELS", "printed_words", "printed"]
