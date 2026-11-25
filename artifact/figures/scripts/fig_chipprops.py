""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.MIN_FONT_PT = 6.0
S.INK_CLEAR_PT = 1.5


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

NAME = "fig_chipprops"
D = S.load(NAME)

HEIGHT_PT = 94.6
DESC_GAP = 1.0
PRINTED = {}

FS = {"focal": 8.0, "label": 6.8, "tick": 6.5, "small": 6.0}
LINE_EM = 1.362
EDGE, RIGHT, COL_GAP = 0.6, 4.2, 1.0
YLAB_GAP, TICK_PAD, TICK_LEN = 1.2, 1.0, 2.0
TITLE_H, TITLE_PAD, FOOT, XLABEL_UP = 10.3, 0.4, 21.7, 1.27
MARK_PT = 3.0
SHAPE = ["o", "s", "^", "D"]
Y_PAD_A = 5.0


Y_B = (1.0, 1.3e6)


BW_CLEAR, BW_EDGE = 4.0, 1.5

GUIDE_FOOT = 1.6

NAME_TOP = -0.4

GUIDE_CLEAR, VALUE_UP = 3.5, 4.2
CORNER_UP, CORNER_DOWN = 2.8, 3.6
CORNER_LEFT = 11.6


FOCAL_PITCH, CORNER_PITCH = 7.5, 7.0
BLOCK_GAP = 6.0

FOCAL_GAP = 3.0

FOCAL_DOWN = 0.25

KEY_IN, KEY_PAD = 3.5, 1.2

LEADER_IN, LEADER_DOWN, LEADER_END = 4.5, 1.8, 1.2

KEY_PITCH = 0.0

LINE_GAP = 3.0
KEY_UP = 1.6
X_PAD = 0.88
SUB_DROP = 1.5
SUB_GAP = 0.3
SYMBOL_LEAD = 0.5
VIOLET, AMBER = S.ROLE["changing"], S.ROLE["admission"]
HOLLOW = dict(linestyle="none", marker="o", markersize=4.6, markerfacecolor=S.C["white"], markeredgecolor=AMBER,
              markeredgewidth=1.1)
SHADE = [0.0, 0.25, 0.45, 0.62]


def spread(values, gap):
    ""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(200):
        moved = False
        for i in range(len(pos) - 1):
            d = pos[i + 1] - pos[i]
            if d < gap - 1e-9:
                pos[i] -= (gap - d) / 2.0
                pos[i + 1] += (gap - d) / 2.0
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for j, i in enumerate(order):
        out[i] = pos[j]
    return out


def end_label(ax, y_data, y_text, text, color, x_end=None, text_color=None, **kw):
    ""
    trans = ax.get_yaxis_transform()
    if x_end is not None and abs(y_text - y_data) > 1e-9:
        x_ax = ax.transAxes.inverted().transform(ax.transData.transform((x_end, y_data)))[0]
        ax.plot([x_ax + 0.03, 1.02], [y_data, y_text], transform=trans, color=color, linewidth=0.6,
                clip_on=False, solid_capstyle="round", zorder=2)
    return ax.annotate(text, xy=(1.0, y_text), xycoords=trans, xytext=(4.5, 0), textcoords="offset points",
                       ha="left", va="center", fontsize=S.FS["label"], color=text_color or S.C["ink"],
                       annotation_clip=False, **kw)


def _extent(t):
    return Text.get_window_extent(t, t.figure.canvas.get_renderer())


def parts_xlabel(ax, parts, fs=None, sub_fs=None, end_pt=None, **label_kw):
    ""
    fig, fs = ax.figure, fs or S.FS["label"]
    sub_fs = sub_fs or S.FS["small"]
    px = fig.dpi / 72.0
    plain = " ".join(t for t, k in parts if k != "sub")
    ax.set_xlabel(plain, fontsize=fs, **label_kw)
    fig.canvas.draw()
    probe = fig.text(0.5, 0.5, plain, fontsize=fs, ha="center", va="baseline")
    descent = fig.bbox.height * 0.5 - _extent(probe).y0
    probe.remove()
    drop = (ax.bbox.y0 - (_extent(ax.xaxis.label).y0 + descent)) / px
    ax.set_xlabel("")
    space = S.text_width_pt(fig, "n n", fs) - S.text_width_pt(fig, "nn", fs)
    kw = dict(xy=(0.5, 0.0), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline",
              color=S.C["ink"], annotation_clip=False)
    x, texts = 0.0, []
    for i, (text, kind) in enumerate(parts):
        sub = kind == "sub"
        x += (SUB_GAP if sub else space + (SYMBOL_LEAD if kind == "symbol" else 0.0)) if i else 0.0
        t = ax.annotate(text, xytext=(x, -drop - (SUB_DROP if sub else 0.0)), fontsize=sub_fs if sub else fs,
                        fontstyle="italic" if kind == "symbol" else "normal", **kw)
        x += _extent(t).width / px
        texts.append((t, t.xyann))
    fig.canvas.draw()
    lift = max(0.0, -min(_extent(t).y0 for t, _ in texts) / px)
    shift = x / 2.0 if end_pt is None else x - ax.bbox.width / px / 2.0 - end_pt
    for t, (x0, y0) in texts:
        t.xyann = (x0 - shift, y0 + lift)
    return [t for t, _ in texts]


def main():
    S.setup()
    W = S.width_pt("column")
    fig = plt.figure(figsize=S.figsize("column", HEIGHT_PT), layout="none")
    fig.canvas.draw()
    conv, end = D["converter"], D["endurance"]


    ylab_h = LINE_EM * FS["label"]
    tick_a = S.text_width_pt(fig, "80", FS["tick"])
    tick_b = S.text_width_pt(fig, "$10^{4}$", FS["tick"])
    left_a, left_b = (EDGE + ylab_h + YLAB_GAP + t + TICK_PAD + TICK_LEN for t in (tick_a, tick_b))
    pw = (W - left_a - left_b - COL_GAP - RIGHT) / 2.0
    desc_b = [end["anchor"]["label"]]
    desc_h = S.descriptor_height(fig, desc_b, size=FS["small"])
    title_h = TITLE_H + 2.1 + DESC_GAP + desc_h
    ph = HEIGHT_PT - title_h - FOOT
    x_a = left_a
    x_b = left_a + pw + COL_GAP + left_b
    ax_a = fig.add_axes([x_a / W, FOOT / HEIGHT_PT, pw / W, ph / HEIGHT_PT])
    ax_b = fig.add_axes([x_b / W, FOOT / HEIGHT_PT, pw / W, ph / HEIGHT_PT])
    for ax, left, tick_w in ((ax_a, left_a, tick_a), (ax_b, left_b, tick_b)):
        ax.tick_params(axis="both", labelsize=FS["tick"], pad=TICK_PAD, length=TICK_LEN)
        ax.tick_params(axis="both", which="minor", length=TICK_LEN * 0.6)
        ax.xaxis.set_label_coords(0.5, -(FOOT - XLABEL_UP) / ph)
        ax.yaxis.set_label_coords(-(TICK_LEN + TICK_PAD + tick_w + YLAB_GAP) / pw, 0.5)

    def y_name(ax, text, left):
        ""
        S.axis_name(fig, ax, text, x=ax.get_position().x0 * W - left + EDGE, size=FS["label"])

    def title(ax, tag, text, left):
        return S.panel_title(ax, tag, text, pad=TITLE_PAD + 2.1 + DESC_GAP + desc_h, x=-(left - EDGE) / pw)


    bits = conv["bits"]
    b_hi, b_lo = max(bits), min(bits)
    ax_a.set_xlim(b_hi + 0.35, b_lo - 0.25)
    loss_hi = max(p["loss_pp"] for s in conv["series"] for p in s["points"])
    ax_a.set_ylim(-Y_PAD_A, loss_hi + Y_PAD_A)
    S.grid(ax_a)
    marks, handles = [], {}
    for i, s in reversed(list(enumerate(conv["series"]))):
        col = S.tint(VIOLET, SHADE[i]) if SHADE[i] else VIOLET
        pts = sorted(s["points"], key=lambda p: -p["bits"])
        xs, ys = [p["bits"] for p in pts], [p["loss_pp"] for p in pts]
        with S.marks(fig, f"a/{i}"):
            ax_a.plot(xs, ys, color=col, linewidth=S.LW["line"], zorder=3 + 0.1 * (4 - i), solid_capstyle="round",
                      clip_on=False)
        marks += [(x, y, col, SHAPE[i], i) for x, y in zip(xs, ys)]
        handles[i] = Line2D([], [], color=col, linewidth=S.LW["line"], marker=SHAPE[i], markersize=MARK_PT,
                            markerfacecolor=col, markeredgecolor=S.C["white"], markeredgewidth=0.4)
    for x, y, col, shape, i in sorted(marks, key=lambda m: -m[1]):
        with S.marks(fig, f"a/{i}"):
            ax_a.plot([x], [y], linestyle="none", marker=shape, markersize=MARK_PT, markerfacecolor=col,
                      markeredgecolor=S.C["white"], markeredgewidth=0.4, zorder=5, clip_on=False)

    order = range(len(conv["series"]))
    key_a = ax_a.legend([handles[i] for i in order], [conv["series"][i]["label"] for i in order], loc="upper left",
                        bbox_to_anchor=(KEY_IN / pw, 1.0), frameon=True, fancybox=False, facecolor=S.C["white"],
                        edgecolor="none", framealpha=1.0, fontsize=FS["small"], handlelength=1.7,
                        handletextpad=0.5, labelspacing=KEY_PITCH, borderaxespad=0.0,
                        borderpad=KEY_PAD / FS["small"])
    key_a.set_zorder(2.5)
    ax_a.set_xticks(range(b_lo, b_hi + 1))
    ax_a.set_yticks([0, 20, 40, 60, 80])
    ax_a.set_xlabel(conv["x_label"], fontsize=FS["label"], va="bottom")
    y_name(ax_a, conv["y_label"], left_a)
    title(ax_a, "a", conv["title"], left_a)


    x_hi = end["x_max"]
    y_lo, y_hi = Y_B
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(X_PAD, x_hi)
    ax_b.set_ylim(y_lo, y_hi)
    lo_c, hi_c = end["corners"]
    keep = [j for j, k in enumerate(hi_c["k"]) if k > 0]
    xs = [hi_c["x"][j] for j in keep]
    ax_b.fill_between(xs, [max(hi_c["k"][j], y_lo) for j in keep], [max(lo_c["k"][j], y_lo) for j in keep],
                      color=S.tint(AMBER, 0.78), linewidth=0, zorder=1)
    for c in (lo_c, hi_c):
        ax_b.plot(xs, [c["k"][j] for j in keep], color=AMBER, linewidth=S.LW["line"], zorder=3)
    bw = end["bandwidth"]
    with S.marks(fig, "b/bandwidth"):
        ax_b.plot(xs, [bw["k"][j] for j in keep], color=S.C["gray"], linewidth=S.LW["thin"], linestyle=(0, (4, 2)),
                  zorder=2)


    names = []
    xa = end["anchor"]["x"]
    k_bw = bw["anchor"]["k"]

    def wide(text):
        return S.text_width_pt(fig, text, FS["small"])

    value_w = wide(bw["anchor"]["display"])
    room = math.log10(xa / X_PAD) / math.log10(x_hi / X_PAD) * pw - value_w / 2.0 - BW_CLEAR - BW_EDGE
    bw_label = PRINTED.get(bw["label"], bw["label"])
    words = bw_label.split()
    bw_lines = [bw_label]
    if wide(bw_label) > room:
        bw_lines = min(([" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))),
                       key=lambda two: max(wide(t) for t in two))
    names.append(S.direct_label(ax_b, xa, k_bw, "\n".join(bw_lines), color=S.C["gray"],
                                dx=-(value_w / 2.0 + BW_CLEAR), dy=VALUE_UP, ha="right", va="bottom",
                                size=FS["small"]))


    if S.withheld([{"what": "(b) batch for the declared rate (write bandwidth) at that overflow, and its line",
                    "key": bw["key"], "source": bw["source"]}]):
        names[-1].set_visible(False)


    k_hi, k_lo = hi_c["anchor"]["k"], lo_c["anchor"]["k"]


    bw_row = {"what": "(b) batch for the declared rate (write bandwidth) at that overflow, and its line",
              "key": bw["key"], "source": bw["source"]}
    foot = 1.0 - (-NAME_TOP + len(bw_lines) * 1.25 * FS["small"] + GUIDE_FOOT) / ph
    with S.marks(fig, "b/overflow"):
        ax_b.plot([xa, xa], [y_lo, S.reach(k_bw, [bw_row], y_lo * (y_hi / y_lo) ** foot)], color=S.C["gray2"],
                  linewidth=0.6, zorder=1.5)
    with S.marks(fig, "b/anchor", tag_at=(ax_b, 0.5, 0.62, "axes")):
        ax_b.plot([xa, xa], [k_hi, k_lo], color=AMBER, linewidth=S.LW["heavy"], solid_capstyle="butt", zorder=3.5)
        ax_b.plot([xa, xa], [k_hi, k_lo], zorder=5, **HOLLOW)
    with S.marks(fig, "b/bandwidth"):
        ax_b.plot([xa], [k_bw], linestyle="none", marker="o", markersize=4.2, markerfacecolor=S.C["white"],
                  markeredgecolor=S.C["gray"], markeredgewidth=0.8, zorder=5)
    S.descriptor(ax_b, desc_b, x=-(left_b - EDGE) / pw, gap=DESC_GAP, size=FS["small"], color=S.C["ink"])

    with S.marks(fig, "b/bandwidth"):
        names.append(S.direct_label(ax_b, xa, k_bw, bw["anchor"]["display"], color=S.C["gray"], dx=0.0,
                                    dy=VALUE_UP, ha="center", va="bottom", size=FS["small"]))


    def x_at(pt):
        ""
        return X_PAD * (x_hi / X_PAD) ** (pt / pw)

    def y_pt(k):
        ""
        return math.log10(k / y_lo) / math.log10(y_hi / y_lo) * ph

    rise = math.log10(1.0 / X_PAD) / math.log10(x_hi / X_PAD) * pw
    def two_lines(text):
        ""
        return text.split(" ", 1)

    corner_lines = two_lines(lo_c["label"])
    corner_ws = [S.text_width_pt(fig, ln, FS["small"]) for ln in corner_lines]
    corner_at = x_at(rise + CORNER_LEFT + max(corner_ws))
    corner_y = lo_c["per_overflow"] * (corner_at - 1.0)
    for i, (ln, w) in enumerate(zip(corner_lines, corner_ws)):
        t_corner = S.direct_label(ax_b, corner_at, corner_y, ln, color=AMBER, dx=-max(corner_ws),
                                  dy=CORNER_UP + (len(corner_lines) - 1 - i) * CORNER_PITCH, ha="left", va="bottom",
                                  size=FS["small"])
        names.append(t_corner)
    names.append(S.direct_label(ax_b, x_hi, hi_c["k"][-1], rf"$10^{{{round(math.log10(hi_c['cycles']))}}}$",
                                color=AMBER, dx=-0.8, dy=-CORNER_DOWN, ha="right", va="top", size=FS["small"]))


    px = fig.dpi / 72.0
    focal_lines = two_lines(end["anchor"]["focal"])
    focal_ws = [S.text_width_pt(fig, ln, FS["focal"], "bold") for ln in focal_lines]
    x_right = math.log10(xa / X_PAD) / math.log10(x_hi / X_PAD) * pw - GUIDE_CLEAR
    x_left = x_right - focal_ws[0]

    between = x_right - max(focal_ws) - (rise + CORNER_LEFT + max(corner_ws))
    if between < BLOCK_GAP:
        raise SystemExit(f"{NAME}: the focal label of (b) starts {between:.1f} pt after the low-endurance line's "
                         f"name (least {BLOCK_GAP})")
    under = y_pt(k_lo) + HOLLOW["markersize"] / 2.0 + HOLLOW["markeredgewidth"] / 2.0
    over = y_pt(bw["anchor"]["k"] / (xa - 1.0) * (x_at(x_left) - 1.0))
    cap = 0.76 * FS["focal"] + FOCAL_PITCH
    room = (over - under - cap) / 2.0
    if room - FOCAL_DOWN < FOCAL_GAP:
        raise SystemExit(f"{NAME}: the focal label of (b) has {room - FOCAL_DOWN:.1f} pt under it and "
                         f"{room + FOCAL_DOWN:.1f} pt over it (least {FOCAL_GAP})")
    base = under + room - FOCAL_DOWN
    up = base - y_pt(k_lo)
    with S.marks(fig, "b/anchor"):
        for i, ln in enumerate(focal_lines):
            t = S.focal(ax_b, ln, (xa, k_lo), (-GUIDE_CLEAR, up + (len(focal_lines) - 1 - i) * FOCAL_PITCH),
                        color=AMBER, textcoords="offset points", ha="right", va="baseline", arrow=False)
            t.set_fontsize(FS["focal"])
            names.append(t)
        ax_b.annotate("", xy=(xa, k_lo), xytext=(-(GUIDE_CLEAR + LEADER_IN), up - LEADER_DOWN),
                      textcoords="offset points",
                      arrowprops=dict(arrowstyle="-", color=AMBER, linewidth=0.7, shrinkA=0.0,
                                      shrinkB=HOLLOW["markersize"] / 2.0 + HOLLOW["markeredgewidth"] / 2.0
                                      + LEADER_END))
    print(f"{NAME}: focal label of (b): {FS['focal']} pt on two lines, last baseline {base:.1f} pt over the axis, "
          f"{room - FOCAL_DOWN:.1f} pt under its ink, {room + FOCAL_DOWN:.1f} pt over it, {between:.1f} pt after "
          f"the low-endurance line's name")

    ax_b.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: f"{v:g}"))
    ax_b.xaxis.set_minor_locator(FixedLocator([2, 4, 5, 6, 7, 8, 9, 20]))
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    majors = [10.0 ** e for e in range(0, round(math.log10(y_hi)) + 1, 2)
              if 10.0 ** e <= y_hi / 10.0 ** 0.35]
    ax_b.yaxis.set_major_locator(FixedLocator(majors))
    ax_b.yaxis.set_major_formatter(FuncFormatter(
        lambda v, pos=None: "1" if v == 1 else rf"$10^{{{round(math.log10(v))}}}$"))
    ax_b.yaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(1, math.floor(math.log10(y_hi)) + 1, 2)]))
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    parts_xlabel(ax_b, end["x_label_parts"], fs=FS["label"], sub_fs=FS["small"], va="bottom")
    y_name(ax_b, end["y_label"], left_b)
    title_b = title(ax_b, "b", end["title"], left_b)

    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    fig.legend([Line2D([], [], **HOLLOW)], ["Estimated"], loc="lower right",
               bbox_to_anchor=(inv.transform((ax_b.get_window_extent(r).x1, 0))[0],
                               inv.transform((0, _extent(title_b).y0 + KEY_UP * fig.dpi / 72.0))[1]),
               bbox_transform=fig.transFigure, frameon=False, fontsize=FS["small"], handlelength=1.0,
               handletextpad=0.3, borderaxespad=0.0, borderpad=0.0)


    fig.canvas.draw()
    curves = {"reference": [bw["k"][j] for j in keep], "low endurance": [lo_c["k"][j] for j in keep],
              "high endurance": [hi_c["k"][j] for j in keep]}
    least, per_name = None, []
    for t in names:
        bb, fs = _extent(t), t.get_fontsize()
        own = None
        y0, y1 = bb.y0 + 0.05 * fs * px, bb.y1 - 0.25 * fs * px
        for cname, ks in curves.items():
            pts = ax_b.transData.transform([(x, max(k, y_lo)) for x, k in zip(xs, ks)])
            gap = None
            for (xa_, ya_), (xb_, yb_) in zip(pts[:-1], pts[1:]):
                for i in range(9):
                    x, y = xa_ + (xb_ - xa_) * i / 8.0, ya_ + (yb_ - ya_) * i / 8.0
                    d = math.hypot(max(bb.x0 - x, 0.0, x - bb.x1), max(y0 - y, 0.0, y - y1)) / px
                    gap = d if gap is None else min(gap, d)
            if least is None or gap < least[0]:
                least = (gap, t.get_text(), cname)
            if own is None or gap < own[0]:
                own = (gap, cname)
            if gap < LINE_GAP:
                print(f"FAIL {NAME}: the name {t.get_text()!r} of (b) is {gap:.1f} pt from the {cname} line, "
                      f"under {LINE_GAP} pt")
        per_name.append(f"{t.get_text()!r} {own[0]:.2f} pt ({own[1]})")
    print(f"{NAME}: least gap of a name of (b) to a line: {least[0]:.1f} pt ({least[1]!r}, the {least[2]} line)")
    print(f"{NAME}: every name's least gap to a line: " + ", ".join(per_name))


    rows = []
    for i, s in enumerate(conv["series"]):
        for p in s["points"]:
            row = {"what": f"(a) {s['label']}: accuracy lost at {p['bits']} converter bits (pp)",
                   "value": round(p["loss_pp"], 4), "source": s["source"], "key": p["key"], "ref": f"a/{i}"}
            if p["key"] is None:
                row["checked_against"] = p["checked_against"]
            rows.append(row)
    for c in end["corners"]:
        rows.append({"what": f"(b) closed form at {c['cycles']:g} cycles: minimum batch per unit of overflow "
                             f"beyond 1 (the line)", "value": c["per_overflow"], "source": c["source"]})
        rows.append({"what": f"(b) {end['anchor']['label']} at {c['cycles']:g} cycles: closed form "
                             f"{c['anchor']['k']:.3f}, printed as batch {c['anchor']['batch']}",
                     "value": c["anchor"]["batch"], "key": c["anchor"]["key"], "source": c["anchor"]["source"],
                     "ref": "b/anchor"})
    rows.append({"what": f"(b) {end['anchor']['label']}: overflow", "value": xa,
                 "key": end["anchor"]["key"], "ref": "b/overflow"})
    rows.append({"what": "(b) batch for the declared rate (write bandwidth) at that overflow, and its line", "value": k_bw, "key": bw["key"],
                 "source": bw["source"], "ref": "b/bandwidth"})
    rows += S.caption_keys(NAME)
    S.pending_key_at(fig, 0.04, 0.29, ha="left", ax=ax_a, words="Release version")


    S.save_fig(fig, NAME, rows=rows)
    S.save_keys(NAME, rows)


if __name__ == "__main__":
    main()
