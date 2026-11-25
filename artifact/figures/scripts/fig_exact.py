""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.MIN_FONT_PT = 5.5
S.FS["title"] = 7.5

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation

NAME = "fig_exact"
HEIGHT = 139.4
ROW_FS = "row"
FS = {"row": 6.0, "small": 6.0, "sub": 5.5, "tick": 6.0, "title": S.FS["title"], "focal": 8.0}


MARK = 3.0
MARK_CLEAR = 0.8
FOCAL_ZERO = True
HEAD_GAP = 2.0
LIFT = 2.7
DROP = 0.0
BAR = 0.2
BAR_PT = 2.05
LINE_W = 1.2
TOP_AIR, FOOT_AIR = 1.6, 0.9
TITLE_PAD = 2.4
TICK_LEN, TICK_PAD = 1.6, 0.8
SUB_DROP = 1.3
MIN_GAP = 7.0
RATIO = 1.132
WSPACE = 0.074
EDGE = 0.5
HEAD_RISE = 3.2
AXIS_ROW_H = 5.6


REASON_AT = "mark"
REASON_CLEAR = 1.2
S.INK_CLEAR_PT = REASON_CLEAR


PRINTED = {"Simulator runs": "Simulators and golden model", "All kept": "All conditions kept",
           "Randomized test, 9 to 64 steps": "Deployments, 9 to 64 time steps"}


def _pt(fig, px):
    return px * 72.0 / fig.dpi


def _extent(t):
    return Text.get_window_extent(t, t.figure.canvas.get_renderer())


def row_text(ax, y, name, number, tag=("", ""), focal=False):
    ""
    fs = FS[ROW_FS]
    kw = dict(xycoords=("axes fraction", "data"), textcoords="offset points", va="baseline")
    left = ax.annotate(name, xy=(0.0, y), xytext=(0.0, LIFT), ha="left", fontsize=fs, color=S.C["ink"], **kw)
    stem, sub = tag
    if stem:
        x = _pt(ax.figure, _extent(left).width) + 3.0
        left = ax.annotate(stem, xy=(0.0, y), xytext=(x, LIFT), ha="left", fontsize=fs, color=S.C["gray"], **kw)
        if sub:
            x += _pt(ax.figure, _extent(left).width) + 0.3
            left = ax.annotate(sub, xy=(0.0, y), xytext=(x, LIFT - SUB_DROP), ha="left", fontsize=FS["sub"],
                               color=S.C["gray"], **kw)
    head, rest = number if isinstance(number, tuple) else ("", number)
    right = ax.annotate(rest, xy=(1.0, y), xytext=(0.0, LIFT), ha="right", fontsize=fs, color=S.C["ink"], **kw)
    row_text.numbers = []
    row_text.rest = right
    if head:
        w = _pt(ax.figure, _extent(right).width)
        right = ax.annotate(head, xy=(1.0, y), xytext=(-w - HEAD_GAP, LIFT), ha="right",
                            fontsize=FS["focal"] if focal else fs, color=S.C["blue"] if focal else S.C["ink"],
                            fontweight="bold", **kw)
        row_text.numbers.append(right)
    return left, right


def count_text(k, n):
    ""
    return ("0", f"of {n:,}") if k == 0 else f"{k:,} of {n:,}"


def zero_mark(ax, x, y, inset=False):
    ""
    n = len(ax.lines)
    S.zero_mark(ax, x, y, size=MARK)
    for ln in ax.lines[n:]:
        ln.set_clip_on(False)
        if inset:
            ln.set_transform(ln.get_transform()
                             + ScaledTranslation(MARK / 2.0 / 72.0, 0.0, ax.figure.dpi_scale_trans))
    _MARKS.append((ax, x, y, MARK / 2.0 if inset else 0.0, MARK / 2.0))


_MARKS = []


def ink_outline(t):
    ""
    fig = t.figure
    r = fig.canvas.get_renderer()
    fp = t.get_fontproperties()
    bb = _extent(t)
    probe = fig.text(0, 0, "x", fontproperties=fp, va="baseline")
    line_desc = -_extent(probe).y0
    probe.remove()
    _, _, d = r.get_text_width_height_descent(t.get_text(), fp, False)
    px = fig.dpi / 72.0
    path = TextPath((0, 0), t.get_text(), size=fp.get_size_in_points(), prop=fp)
    return [poly * px + (bb.x0, bb.y0 + max(d, line_desc)) for poly in path.to_polygons(closed_only=False)]


def mark_report(fig):
    ""
    import numpy as np
    fig.canvas.draw()
    px = fig.dpi / 72.0
    texts = [t for ax in fig.axes for t in list(ax.texts) + [ax._left_title] if t.get_text().strip()]
    inks = [(t, ink_outline(t)) for t in texts]
    least = None
    for ax, x, y, dx, rad in _MARKS:
        c = ax.transData.transform((x, y)) + (dx * px, 0.0)
        for t, polys in inks:
            bb = _extent(t)
            if (bb.x0 - c[0] > (rad + 3.0) * px or c[0] - bb.x1 > (rad + 3.0) * px
                    or bb.y0 - c[1] > (rad + 3.0) * px or c[1] - bb.y1 > (rad + 3.0) * px):
                continue
            for poly in polys:
                a, b = poly[:-1], poly[1:]
                ab = b - a
                u = np.clip(((c - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
                gap = float(np.hypot(*(a + ab * u[:, None] - c).T).min()) / px - rad
                if least is None or gap < least[0]:
                    least = (gap, t.get_text())
                if gap < MARK_CLEAR:
                    print(f"FAIL {NAME}: a mark stands {gap:.2f} pt from the ink of {t.get_text()!r}, under "
                          f"{MARK_CLEAR} pt")
    if least:
        print(f"marks {NAME}: {len(_MARKS)} marks of {MARK} pt, least white to a text's ink {least[0]:.2f} pt "
              f"({least[1]!r}), floor {MARK_CLEAR} pt")


def keep_marks_clear(fig):
    ""
    from sciviz import inkcenter

    fig.canvas.draw()
    px = fig.dpi / 72.0
    discs = [tuple(ax.transData.transform((x, y)) + (dx * px, 0.0)) + (rad,) for ax, x, y, dx, rad in _MARKS]
    lines = {}
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip() and hasattr(t, "xyann"):
                lines.setdefault((id(ax), round(float(t.xy[1]), 9)), []).append(t)
    inkcenter.keep_clear_of_discs(list(lines.values()), discs, MARK_CLEAR)


def glyph_height(text, size):
    ""
    return TextPath((0, 0), text, size=size,
                    prop=FontProperties(family=plt.rcParams["font.family"])).get_extents().height


def unit_head(ax, i, unit):
    ""
    rise = LIFT + glyph_height("0", FS[ROW_FS]) + HEAD_RISE
    return ax.annotate(unit[:1].upper() + unit[1:], xy=(0.0, i + DROP), xycoords=("axes fraction", "data"),
                       xytext=(0.0, rise), textcoords="offset points", ha="left", va="baseline",
                       fontsize=FS["small"], color=S.C["ink2"])


def pending_mark(ax, x, y):
    ""
    from matplotlib.transforms import offset_copy
    tr = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=0.0, y=LIFT, units="points")
    S.tag_label(ax, 1.0, y, transform=tr, size=FS[ROW_FS], ha="right", va="baseline")


def rows_axis(ax, n):
    ax.set_ylim(n - 0.5 + 0.1, -0.5 - 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=TICK_LEN, pad=TICK_PAD, labelsize=FS["tick"], labelbottom=False)


def title(ax, tag, text):
    ""
    t = S.panel_title(ax, tag, text, pad=TITLE_PAD)
    t.set_fontsize(FS["title"])
    return t


def axis_row(ax, lo, hi, name):
    ""
    kw = dict(xycoords="axes fraction", textcoords="offset points", va="baseline", fontsize=FS["tick"],
              color=S.C["ink"])
    tallest = max(glyph_height(t, FS["tick"]) for t in (lo, hi))
    if tallest > AXIS_ROW_H + 0.05:
        print(f"FAIL {NAME}: an axis end label is {tallest:.1f} pt high, over the {AXIS_ROW_H} pt of the axis row")
    drop = TICK_LEN + TICK_PAD + AXIS_ROW_H
    return (ax.annotate(lo, xy=(0.0, 0.0), xytext=(0.0, -drop), ha="left", **kw),
            ax.annotate(name, xy=(0.5, 0.0), xytext=(0.0, -drop), ha="center", **kw),
            ax.annotate(hi, xy=(1.0, 0.0), xytext=(0.0, -drop), ha="right", **kw))


def fit_rows(fig, axes, n):
    ""
    fig.canvas.draw()
    over = LIFT + glyph_height("0", FS[ROW_FS]) + TOP_AIR
    under = MARK / 2.0 + FOOT_AIR
    for ax in axes:
        pitch = (_pt(fig, ax.get_window_extent().height) - over - under) / max(n - 1, 1)
        ax.set_ylim(n - 1 + DROP + under / pitch, DROP - over / pitch)
        for bar in ax.patches:
            if bar.get_height() > 0:
                mid = bar.get_y() + bar.get_height() / 2.0
                bar.set_height(BAR_PT / pitch)
                bar.set_y(mid - LINE_W / 2.0 / pitch)
    global BAR
    BAR = 2.0 * (BAR_PT - LINE_W / 2.0) / pitch
    return pitch


def panel_families(ax, fams, meta, slots):
    done = [f for f in fams if f["status"] == "measured"]
    lo_e = math.floor(math.log10(min(f["compared"] for f in done))) - 1
    top = max(f["compared"] for f in done)
    left, right = 10.0 ** lo_e, 10.0 ** math.ceil(math.log10(top))
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    rows_axis(ax, slots)
    pairs = []
    fig = ax.figure
    for i, f in enumerate(fams):
        y = i + DROP

        name = PRINTED.get(f["label"], f["label"])
        if f["status"] != "measured":
            pairs.append(row_text(ax, y, name, ""))
            with S.marks(fig, f"a/{i}", series="a"):
                pending_mark(ax, left, y)
            continue
        pairs.append(row_text(ax, y, name, count_text(f["differ"], f["compared"]), focal=FOCAL_ZERO))
        S.add_marks(fig, f"a/{i}", row_text.numbers, series="a")
        S.relabel(fig, f"a/{i}", row_text.rest, S.PENDING_TOKEN, as_tag=True, tag_size=FS["sub"])
        with S.marks(fig, f"a/{i}", series="a"):
            ax.hlines(y, left, f["compared"], color=S.C["gray2"], lw=LINE_W, zorder=2)
            zero_mark(ax, f["compared"], y)
    hi_e = math.ceil(math.log10(top))
    ax.set_xticks([10.0 ** e for e in range(lo_e, hi_e + 1)])
    ax.minorticks_off()
    axis_row(ax, S._log_tick_label(left), S._log_tick_label(right), meta["xlabel"])
    title(ax, "a", meta["title"])
    return pairs


def control_groups(ctrl):
    ""
    groups = [g for g in ([ctrl["baseline"]], ctrl["removed"], ctrl["other"]) if g]
    heads, last = [], None
    for g in groups:
        units = {c["unit"] for c in g}
        if len(units) != 1:
            raise SystemExit(f"{NAME}: panel (b): the rows of one group count more than one unit: {sorted(units)}")
        unit = units.pop()
        heads.append(unit if unit != last else None)
        last = unit
    return groups, heads


def control_slots(ctrl, slots):
    ""
    groups, heads = control_groups(ctrl)
    spare = slots - sum(len(g) for g in groups) - sum(h is not None for h in heads)
    if spare < 0:
        raise SystemExit(f"{NAME}: panel (b): no spare slot for the unit headers ({-spare} short)")
    gaps = [0] * len(groups)
    if len(groups) > 1:
        gaps[1] += spare // 2
        gaps[-1] += spare - spare // 2
    at, head_at, i = [], [], 0
    for g, head, gap in zip(groups, heads, gaps):
        i += gap + (head is not None)
        if head is not None:
            head_at.append((i, head))
        at += list(range(i, i + len(g)))
        i += len(g)
    return at, head_at


def panel_controls(ax, ctrl, meta, slots):
    ""
    rows = [ctrl["baseline"]] + ctrl["removed"] + ctrl["other"]
    at, head_at = control_slots(ctrl, slots)
    headed = {i for i, _ in head_at}
    apart = {id(c) for c in ctrl["other"]}
    removed = {id(c) for c in ctrl["removed"]}
    ax.set_xlim(0.0, 100.0)
    rows_axis(ax, slots)
    pairs, rules, runs = [], [], []
    fig = ax.figure
    for n, c in enumerate(rows):
        i = at[n]
        y = i + DROP
        tag = (c["tag_stem"], c["tag_sub"])
        mk = S.marks(fig, f"b/{n}", series="b")
        name = PRINTED.get(c["label"], c["label"])
        if c["status"] != "measured":
            pairs.append(row_text(ax, y, name, "", tag))
            with mk:
                pending_mark(ax, 0.0, y)
        elif c["differ"] == 0:


            number = f"0 of {c['of']:,}" if id(c) in removed else count_text(c["differ"], c["of"])
            pairs.append(row_text(ax, y, name, number, tag))
            with mk:
                zero_mark(ax, 0.0, y, inset=True)
            if c.get("reason"):
                if runs and runs[-1][1] == c["reason"] and runs[-1][0] == i - 1:
                    runs[-1][0] = i
                    runs[-1][2].append(f"b/{n}")
                else:
                    runs.append([i, c["reason"], [f"b/{n}"]])
        else:
            number = count_text(c["differ"], c["of"]) if c["number"] == "count" else f"{c['share_pct']:.0f}%"
            pairs.append(row_text(ax, y, name, number, tag))
            with mk:
                ax.barh(y, c["share_pct"], height=BAR, color=S.C["gray"], edgecolor="none", zorder=3)
        S.add_marks(fig, f"b/{n}", row_text.numbers, series="b")
        if c["status"] == "measured":
            S.relabel(fig, f"b/{n}", row_text.rest, S.PENDING_TOKEN, as_tag=True, tag_size=FS["sub"])
        if id(c) in apart:
            rules.append((i, at[n - 1], i in headed))
    heads = [unit_head(ax, i, unit) for i, unit in head_at]

    S.pending_key_at(fig, 1.0, 1.2, ha="right", ax=ax, words="Release version", size=FS["sub"], anchor=heads[0])
    ax.set_xticks([0, 25, 50, 75, 100])
    axis_row(ax, "0", "100", meta["xlabel"])
    title(ax, "b", meta["title"])
    return pairs, rules, runs


def _ink_y(t):
    ""
    fp = t.get_fontproperties()
    ink = TextPath((0, 0), t.get_text(), size=fp.get_size_in_points(), prop=fp).get_extents()
    base = t.xyann[1]
    return base + ink.y0, base + ink.y1


def reason_lines(ax, runs):
    ""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    reach = {}
    for slot, words, refs in runs:
        y = slot + DROP
        row = [t for t in ax.texts if t.get_text().strip() and hasattr(t, "xyann") and abs(t.xy[1] - y) < 1e-9]
        kw = dict(xycoords=("axes fraction", "data"), textcoords="offset points", va="baseline",
                  fontsize=FS["sub"], color=S.C["gray"])
        t = (ax.annotate(words, xy=(0.0, y), xytext=(MARK + HEAD_GAP, 0.0), ha="left", **kw) if REASON_AT == "mark"
             else ax.annotate(words, xy=(1.0, y), xytext=(0.0, 0.0), ha="right", **kw))
        fig.canvas.draw()
        span = Text.get_window_extent(t, r)
        over = [_ink_y(o)[0] for o in row
                if min(span.x1, Text.get_window_extent(o, r).x1) > max(span.x0, Text.get_window_extent(o, r).x0)]
        lo, hi = _ink_y(t)
        top = (min(over) if over else LIFT) - REASON_CLEAR
        t.xyann = (t.xyann[0], top - hi)
        for ref in refs:
            S.add_marks(fig, ref, [t], series="b")
        reach[slot] = -(top - hi + lo)
    return reach


def hairlines(ax, rules, reach=None):
    ""
    fig = ax.figure
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    pitch = _pt(fig, ax.get_window_extent().height) / abs(hi - lo)
    digits = glyph_height("0", FS[ROW_FS])
    head = HEAD_RISE + glyph_height("A", FS["small"])
    for i, above, headed in rules:
        text_top = LIFT + digits + (head if headed else 0.0)
        under = max(pitch * BAR / 2.0, (reach or {}).get(above, 0.0))
        ax.axhline(i + DROP - (text_top + pitch * (i - above) - under) / 2.0 / pitch, color=S.C["gray2"],
                   lw=S.LW["grid"], zorder=1)


def gap_report(fig, panels):
    ""
    fig.canvas.draw()
    worst = {}
    for tag, pairs in panels:
        for left, right in pairs:
            gap = _pt(fig, _extent(right).x0 - _extent(left).x1)
            worst[tag] = min(gap, worst.get(tag, gap))
            if gap < MIN_GAP:
                print(f"FAIL {NAME}: panel ({tag}): a row name ends {gap:.1f} pt before its number "
                      f"({right.get_text()!r}), under {MIN_GAP:.0f} pt")
    a, b = fig.axes[0], fig.axes[1]
    for ax in (a, b):
        over = _pt(fig, _extent(ax._left_title).x1 - ax.get_window_extent().x1)
        if over > 0.3:
            print(f"FAIL {NAME}: the title {ax._left_title.get_text()!r} runs {over:.1f} pt past its panel")
    gutter = _pt(fig, b.get_window_extent().x0 - a.get_window_extent().x1)
    least = ", ".join(f"{g:.1f} pt in ({t})" for t, g in worst.items())
    print(f"rows {NAME}: text {FS[ROW_FS]} pt, least name to number gap {least}, "
          f"gutter {gutter:.1f} pt, panels {_pt(fig, a.get_window_extent().width):.1f} and "
          f"{_pt(fig, b.get_window_extent().width):.1f} pt")


def keys_rows(d) -> list:
    rows = []

    def add(what, value, item, ref):
        row = {"what": what, "value": value, "key": ", ".join(k["key"] for k in item["keys"])}
        if item.get("derived"):
            row["derivation"] = item["derived"]
        row["ref"] = ref
        rows.append(row)

    for i, f in enumerate(d["families"]):
        add(f"(a) {f['label']}: compared instances ({f['unit']})", f["compared"], f, f"a/{i}")
        add(f"(a) {f['label']}: instances that differ", f["differ"], f, f"a/{i}")
        if f.get("window_range"):
            r = f["window_range"]

            rows.append(S.category({"what": f"(a) {f['label']}: the window lengths (steps, T in the text) in the "
                                            f"label", "value": f"{r['lo']} to {r['hi']}",
                                    "key": ", ".join(k["key"] for k in f["keys"]),
                                    "derivation": ", checked against ".join(r["basis"])}))
    c = d["controls"]
    for n, r in enumerate([c["baseline"]] + c["removed"] + c["other"]):
        add(f"(b) {r['label']}: differing of {r['unit']}", f"{r['differ']} of {r['of']}", r, f"b/{n}")
        add(f"(b) {r['label']}: differing share (%), the bar's length", r["share_pct"], r, f"b/{n}")
    return rows


def main():
    S.setup()
    plt.rcParams["text.hinting"] = "none"
    d = S.load(NAME)
    fig, (a, b) = plt.subplots(1, 2, figsize=S.figsize("column", HEIGHT),
                               gridspec_kw={"width_ratios": [RATIO, 1.0], "wspace": WSPACE})
    fig.get_layout_engine().set(w_pad=EDGE / 72.0, h_pad=0.3 / 72.0)
    c = d["controls"]
    groups, heads = control_groups(c)
    slots = max(len(d["families"]),
                sum(len(g) for g in groups) + sum(h is not None for h in heads))
    pa = panel_families(a, d["families"], d["panels"]["a"], slots)
    pb, rules, runs = panel_controls(b, c, d["panels"]["b"], slots)
    pitch = fit_rows(fig, (a, b), slots)
    print(f"rows {NAME}: pitch {pitch:.2f} pt")
    reach = reason_lines(b, runs)
    hairlines(b, rules, reach)
    gap_report(fig, [("a", pa), ("b", pb)])
    keep_marks_clear(fig)
    mark_report(fig)
    rows = keys_rows(d) + S.caption_keys(NAME)
    S.save_fig(fig, NAME, rows=rows)
    S.save_keys(NAME, rows)


if __name__ == "__main__":
    main()
