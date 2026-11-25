""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.MIN_FONT_PT = 6.0
S.INK_CLEAR_PT = 1.0
S.FS["title"] = 7.5

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

NAME = "fig_fronts"


FS = {"focal": 8.0, "label": 6.8, "tick": 6.5, "small": 6.0, "axis": 6.5, "number": 6.0}


BOX = 1.362
LAYOUT = {
    "tick": 1.8, "tick_pad": 0.8, "label_pad": -1.6,


    "title_pad": 2.3, "title_pad_low": 1.6,


    "pitch": 9.0,
    "top_h": 41.5,
    "rows_gap": 4.5,
    "edge": 0.2,
}
LAYOUT["band"] = LAYOUT["pitch"]
LAYOUT["low_h"] = 4 * LAYOUT["pitch"]

LINES = (6.4, 14.6)

MARK = {
    "pick": 4.4, "ring": 4.0, "ring_same": 6.4, "pool": 2.3,
    "heavy": 1.7, "arrow": 6.2,
    "seed": 2.0, "mean": 3.6, "bound": 4.6,
    "ratio": 3.8,
}
LAYOUT["x_block"] = (LAYOUT["tick"] + LAYOUT["tick_pad"] + BOX * FS["number"] + LAYOUT["label_pad"]
                     + BOX * FS["axis"])
LAYOUT["title_up"] = (BOX - 0.293) * S.FS["title"]
LAYOUT["low_b"] = LAYOUT["x_block"] + LAYOUT["edge"]
LAYOUT["top_b"] = (LAYOUT["low_b"] + LAYOUT["low_h"] + LAYOUT["band"] + LAYOUT["title_pad_low"] + LAYOUT["title_up"]
                   + LAYOUT["rows_gap"] + LAYOUT["x_block"])
S.setup()
plt.rcParams.update({"axes.labelsize": FS["axis"], "xtick.labelsize": FS["number"], "ytick.labelsize": FS["number"],
                     "axes.labelpad": LAYOUT["label_pad"], "axes.titlepad": LAYOUT["title_pad"],
                     "xtick.major.pad": LAYOUT["tick_pad"], "ytick.major.pad": 1.2,
                     "xtick.major.size": LAYOUT["tick"], "ytick.major.size": 2.2, "xtick.minor.size": 1.2})
D = S.load(NAME)
ORDER, CELLS = D["order"], D["cells"]

BLUE, GRAY, GRAY2, GRAY3, INK, INK2 = (S.C[k] for k in ("blue", "gray", "gray2", "gray3", "ink", "ink2"))
OVER = S.tint(S.C["gray"], 0.86)

H_PT = round(LAYOUT["top_b"] + LAYOUT["top_h"] + LAYOUT["title_pad"] + LAYOUT["title_up"] + LAYOUT["edge"], 1)
W_PT = S.TEXT_PT
MU = "µ"
DASH = (3.2, 1.8)

WORDS = {
    "budget_line": "Energy budget",
    "over_budget": "Over budget",
    "cell_suffix": " budgets",
    "gain_title": "Gain at equal budgets",


    "legend": ("Five seeds", "Seeds", "Seed 0 gain"),
    "pool": "Pool networks",
    "seed": "Seed 0",
    "ratio_axis": "Sequential-to-joint energy",

    "ratio_band": "Matched accuracy",
    "same_read": "Same as {row}",

    "pending": "Pending",
}
FOCAL_DX = 5.2
GAP_TICK = 6.0


DESCRIPTIVE_BAR = {"height": 3.2, "edge": 0.7}
AXIS_AIR = 1.0
KEY_UP = 0.9
YLABEL_PAD = -0.4

PANEL_E_DRAWN = ("five_seed", "five_seed_pending", "front_gap")


def check_panel_e():
    ""
    marks = tuple(D["panels"]["e"]["marks"])
    if marks != PANEL_E_DRAWN:
        raise SystemExit(f"(e): the data lists the marks {marks}, this script draws {PANEL_E_DRAWN}")
    for cid in ORDER:
        c = CELLS[cid]
        want = (c["at_cap"]["multi"]["network"], c["at_cap"]["single"]["network"], c["cap_uJ"]["value"],
                c["area_budget_mm2"]["value"])
        for mark in marks:
            s = (c.get(mark) or {}).get("selection")
            if c.get(mark) is not None and (s is None or (s["multi"], s["single"], s["energy_budget_uJ"],
                                                          s["area_budget_mm2"]) != want):
                raise SystemExit(f"(e) {cid}: the mark {mark} reads a selection made under other budgets than its cell's")


def names():
    ""
    out = {}
    for cid in ORDER:
        c = CELLS[cid]
        same = [o for o in ORDER if CELLS[o]["batch"]["value"] == c["batch"]["value"]]
        name = f"Batch {int(c['batch']['value'])}"
        if len(same) > 1:
            top = max(float(CELLS[o]["caps"]["energy_mult"]) for o in same)
            name += ", loose" if float(c["caps"]["energy_mult"]) == top else ", tight"
        out[cid] = name
    return out


def ax_pt(fig, left, bottom, width, height, **kw):
    ""
    return fig.add_axes([left / W_PT, bottom / H_PT, width / W_PT, height / H_PT], **kw)


def span_xlabel(fig, ax, axes, text):
    ""
    ax.set_xlabel(text)
    fig.canvas.draw()
    bb = ax.xaxis.label.get_window_extent()
    y = ax.transAxes.inverted().transform((0, bb.y1))[1]
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    xm = ax.transAxes.inverted().transform(fig.transFigure.transform(((x0 + x1) / 2, 0)))[0]
    ax.xaxis.set_label_coords(xm, y)


def layout_reading(fig, tops, gain_ax, thin, leg, pool_key, band0, focal_end):
    ""
    fig.canvas.draw()
    r, pt = fig.canvas.get_renderer(), 72.0 / fig.dpi
    box = lambda a: a.get_window_extent(r)
    title = lambda a: box(a._left_title)
    lows = (gain_ax, thin[0])
    readings = (
        ("upper axis name to the lower titles (boxes)", (box(tops[0].xaxis.label).y0 - max(title(a).y1 for a in lows)) * pt, -0.6),
        ("key of (e) to the band of (f)", (box(band0).x0 - box(leg).x1) * pt, 3.0),
        ("right end of (e) to the tick numbers of (f)", (min(t.get_window_extent(r).x0 for t in thin[0].get_yticklabels() if t.get_text()) - gain_ax.get_window_extent(r).x1) * pt, 6.0),
        ("panels of (f) to the panels of (c) and (d), left and right ends", max(abs(thin[0].get_window_extent(r).x0 - tops[2].get_window_extent(r).x0), abs(thin[-1].get_window_extent(r).x1 - tops[3].get_window_extent(r).x1)) * -pt, -0.05),
        ("title of (a) to its key", (box(pool_key).x0 - title(tops[0]).x1) * pt, 6.0),
        ("gain label of (a) to the panel's right end", (tops[0].get_window_extent(r).x1 - box(focal_end).x1) * pt, 1.0),
        ("upper titles to the page's upper edge (boxes)", (fig.bbox.height - max(title(a).y1 for a in tops)) * pt, 0.0),
        ("lower axis names to the page's lower edge (boxes)", min(box(a.xaxis.label).y0 for a in lows) * pt, 0.0),
        ("axis name of (e) to the axis name of (f)", (box(thin[0].xaxis.label).x0 - box(gain_ax.xaxis.label).x1) * pt, 6.0),
    )
    for what, v, least in readings:
        print(f"{'layout' if v >= least - 0.05 else 'FAIL layout'} {NAME}: {what}: {v:.1f} pt (least {least} pt)")


def plain(v, _=None):
    return f"{v:,g}"


def split_at(front, cap):
    ""
    pts = list(zip(front["x"], front["y"]))
    under = [p for p in pts if p[0] <= cap]
    at = under[-1][1]
    return under + [(cap, at)], [(cap, at)] + [p for p in pts if p[0] > cap]


def riser_to(front, level):
    ""
    pts = list(zip(front["x"], front["y"]))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 >= level - 1e-9 and y0 < level - 1e-9:
            return x1, y0
    raise ValueError(f"the front never reaches {level}")


def member_dots(ax, pts, color):
    if pts:
        ax.plot(*zip(*pts), linestyle="none", marker="o", markersize=MARK["pool"], markerfacecolor=color, markeredgewidth=0,
                zorder=2)


def fronts(ax, part, *, cap=None):
    ""
    fm, fs = part["front_multi"], part["front_single"]
    if cap is None:
        ax.plot(fm["x"], fm["y"], color=BLUE, linewidth=S.LW["line"], zorder=3, solid_joinstyle="miter")
        ax.plot(fs["x"], fs["y"], color=GRAY, linewidth=0.9, zorder=4, dashes=DASH)
        return
    m = part["members"]
    for single, color, far in ((False, S.tint(BLUE, 0.55), S.tint(BLUE, 0.78)), (True, GRAY2, S.tint(GRAY2, 0.5))):
        mine = [(x, y) for x, y, s in zip(m["x"], m["y"], m["single"]) if bool(s) == single]
        member_dots(ax, [p for p in mine if p[0] <= cap], color)
        member_dots(ax, [p for p in mine if p[0] > cap], far)
    m_in, m_out = split_at(fm, cap)
    s_in, s_out = split_at(fs, cap)
    ax.plot(*zip(*m_out), color=S.tint(BLUE, 0.55), linewidth=1.1, zorder=2.2, solid_joinstyle="miter")
    ax.plot(*zip(*s_out), color=S.tint(GRAY, 0.45), linewidth=0.9, zorder=2.3, dashes=DASH)
    ax.plot(*zip(*m_in), color=BLUE, linewidth=MARK["heavy"], zorder=3, solid_joinstyle="miter",
            solid_capstyle="butt")
    ax.plot(*zip(*s_in), color=GRAY, linewidth=1.1, zorder=4, dashes=DASH)


PAD_PT = 1.5


def free_right_edge(fig, ax, cap, part, labels):
    ""
    k = 72.0 / fig.dpi
    to_pt = lambda x, y: ax.transData.transform((x, y)) * k
    top = ax.get_window_extent().y1 * k
    cap_x = to_pt(cap, 0.0)[0]
    obstacles = []
    m_in, _ = split_at(part["front_multi"], cap)
    half = MARK["heavy"] / 2.0
    for (xa, ya), (xb, yb) in zip(m_in, m_in[1:]):
        (pa, qa), (pb, qb) = to_pt(xa, ya), to_pt(xb, yb)
        obstacles.append((min(pa, pb) - half, max(pa, pb) + half, min(qa, qb) - half, max(qa, qb) + half))
    rad = MARK["pool"] / 2.0
    for x, y in zip(part["members"]["x"], part["members"]["y"]):
        if x <= cap:
            p, q = to_pt(x, y)
            obstacles.append((p - rad, p + rad, q - rad, q + rad))
    bands = []
    for text, size, weight, depth in labels:
        base = top - depth
        bands.append((S.text_width_pt(fig, text, size, weight), base - 0.293 * size, base + (BOX - 0.293) * size))
    edge = cap_x - 5.0
    for _ in range(len(obstacles) + 1):
        hits = [o for o in obstacles for w, b0, b1 in bands
                if o[2] < b1 and o[3] > b0 and o[1] > edge - w - PAD_PT and o[0] < edge + PAD_PT]
        if not hits:
            break
        edge = min(o[0] for o in hits) - PAD_PT
    left_end = edge - max(w for w, _, _ in bands) - ax.get_window_extent().x0 * k
    if left_end < 1.0:
        print(f"FAIL layout {NAME}: a label stack on the upper lines has no free place ({left_end:.1f} pt)")
    return edge - cap_x


def main():
    check_panel_e()
    name = names()
    fig = plt.figure(figsize=S.figsize("text", H_PT))
    fig.set_layout_engine("none")
    keys = []

    yticks = [65, 70, 75]
    left, right = 29.0, 7.0


    top_b, top_h = LAYOUT["top_b"], LAYOUT["top_h"]
    gap = 9.0
    pw = (W_PT - left - right - 3 * gap) / 4
    e_all = [x for cid in ORDER for x in CELLS[cid]["energy"]["members"]["x"]]


    c0 = CELLS[ORDER[0]]
    focal_text = f"+{c0['front_gap']['text']} pp"
    need = (FOCAL_DX + S.text_width_pt(fig, focal_text, FS["focal"], "bold") + 2.4
            + S.text_width_pt(fig, WORDS["seed"], FS["small"]) + 1.5)
    x_lo, r_need = min(e_all) * 0.8, need / pw
    u_hi = (math.log10(c0["cap_uJ"]["value"]) - r_need * math.log10(x_lo)) / (1.0 - r_need)
    xlim = (x_lo, max(max(e_all) * 1.85, 10 ** u_hi))
    decades = range(math.floor(math.log10(xlim[0])), math.ceil(math.log10(xlim[1])) + 1)
    x_major = [10.0 ** e for e in decades if xlim[0] <= 10.0 ** e <= xlim[1]]
    x_minor = [m * 10.0 ** e for e in decades for m in range(2, 10) if xlim[0] <= m * 10.0 ** e <= xlim[1]]

    picks = [CELLS[cid]["at_cap"]["single"]["value"] for cid in ORDER]
    best = max(CELLS[cid]["energy"]["front_multi"]["y"][-1] for cid in ORDER)

    ylim = (min(picks) - 2.0, best + 3.9)

    ylim_time = (ylim[0], best + 1.0)
    pt_pp = LAYOUT["top_h"] / (ylim[1] - ylim[0])
    tops = []
    for i, cid in enumerate(ORDER):
        c = CELLS[cid]
        ax = ax_pt(fig, left + i * (pw + gap), top_b, pw, top_h)
        tops.append(ax)
        cap = c["cap_uJ"]["value"]
        lo, hi = c["at_cap"]["single"]["value"], c["at_cap"]["multi"]["value"]
        same = abs(hi - lo) < 1e-9
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axvspan(cap, xlim[1], color=OVER, linewidth=0, zorder=0)
        fronts(ax, c["energy"], cap=cap)
        ax.axvline(cap, color=GRAY, linewidth=S.LW["thin"], zorder=2.6)


        ax.plot([cap], [lo], linestyle="none", marker="o", markersize=MARK["ring_same"] if same else MARK["ring"], markerfacecolor="white",
                markeredgecolor=GRAY, markeredgewidth=1.0, zorder=6)
        ax.plot([cap], [hi], linestyle="none", marker="o", markersize=MARK["pick"], markerfacecolor=BLUE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=7)
        fs5 = c["five_seed"]

        fs5 = fs5 if fs5 and not fs5.get("descriptive") else None
        if fs5 or not same:


            fgap = c["front_gap"]
            if abs((hi - lo) - float(fgap["value"])) > 0.005 or hi <= lo:
                raise SystemExit(f"{cid}: the selections at the energy budget span {hi - lo:.2f} pp; the fronts' gap of record is "
                                 f"{fgap['text']} pp ({fgap['key']})")


            arrow_fits = (hi - lo) * pt_pp >= 3.0 + 2.8 + 0.4 * MARK["arrow"] + 1.0
            if fs5 and arrow_fits:
                ax.annotate("", xy=(cap, hi), xytext=(cap, lo), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.0, shrinkA=3.0, shrinkB=2.8,
                                            mutation_scale=MARK["arrow"]))
            text = f"+{fgap['text']} pp"
            if i == 0:


                gain_w = S.text_width_pt(fig, text, FS["focal"], "bold")
                S.focal(ax, text, (cap, (lo + hi) / 2), xytext=(FOCAL_DX, -0.357 * FS["focal"]),
                        textcoords="offset points", arrow=False, ha="left",
                        va="baseline").set_fontsize(FS["focal"])
                focal_end = S.direct_label(ax, cap, (lo + hi) / 2, WORDS["seed"], color=BLUE,
                                           dx=FOCAL_DX + gain_w + 2.4, dy=-0.357 * FS["focal"], va="baseline",
                                           size=FS["small"])
            else:


                weight = "bold" if fs5 else "normal"
                dx = free_right_edge(fig, ax, cap, c["energy"], ((text, FS["label"], weight, LINES[0]),
                                                                 (WORDS["seed"], FS["small"], "normal", LINES[1])))
                S.direct_label(ax, cap, ylim[1], text, color=BLUE, dx=dx, dy=-LINES[0], ha="right", va="baseline",
                               size=FS["label"], fontweight=weight)
                S.direct_label(ax, cap, ylim[1], WORDS["seed"], color=BLUE, dx=dx, dy=-LINES[1], ha="right",
                               va="baseline", size=FS["small"])
            keys.append({"what": f"{name[cid]}: gain label at the energy budget (the gap of the two drawn selections, "
                                 f"seed 0, development split, in pp)",
                         "value": fgap["text"], "key": fgap["key"]})
        ax.xaxis.set_major_locator(FixedLocator(x_major))
        ax.xaxis.set_major_formatter(FuncFormatter(plain))
        ax.xaxis.set_minor_locator(FixedLocator(x_minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        S.grid(ax)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", labelpad=YLABEL_PAD)
            rx, plateau = riser_to(c["energy"]["front_multi"], hi)
            if (ylim[1] - plateau) * pt_pp - LINES[1] < 0.2 * FS["tick"] + 2.0:
                raise SystemExit("the second label line of (a) runs into the multi-pass front under it")
            S.direct_label(ax, rx, ylim[1], "Multi-pass", color=BLUE, dx=-3.5, dy=-LINES[1], ha="right", va="baseline",
                           size=FS["tick"])
            S.direct_label(ax, cap, lo, "Single-pass", color=INK2, dx=FOCAL_DX, dy=-7.0, size=FS["tick"])


            S.direct_label(ax, cap, ylim[1], WORDS["budget_line"], color=INK2, dx=-4, dy=-LINES[0], ha="right",
                           va="baseline", size=FS["small"])


            S.direct_label(ax, cap, ylim[1], WORDS["over_budget"], color=INK2, dx=4, dy=-LINES[0], va="baseline",
                           size=FS["small"])

            pool = [ax.plot([], [], linestyle="none", marker="o", markersize=MARK["pool"], markerfacecolor=col,
                            markeredgewidth=0)[0] for col in (S.tint(BLUE, 0.55), GRAY2)]
            pool_key = ax.legend([tuple(pool)], [WORDS["pool"]], loc="lower right",
                                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                                 bbox_to_anchor=(1.0, 1.0 + (LAYOUT["title_pad"] - BOX * 0.293 * FS["small"]) / top_h),
                                 fontsize=FS["small"], frameon=False, handlelength=1.3, handletextpad=0.35,
                                 borderaxespad=0.0, borderpad=0.0, labelcolor=INK2)
        else:
            ax.tick_params(labelleft=False)
        S.panel_title(ax, "abcd"[i], name[cid] + (WORDS["cell_suffix"] if "," in name[cid] else ""))
        keys += [
            {"what": f"{name[cid]}: energy budget ({MU}J)", "value": cap, "key": c["cap_uJ"].get("key"),
             "source": c["cap_uJ"]["source"]},
            {"what": f"{name[cid]}: trained members, accuracy against energy", "value": len(c["energy"]["members"]["x"]),
             "source": c["energy"]["members"]["source"]},
            {"what": f"{name[cid]}: single-pass energy front", "value": c["energy"]["front_single"]["y"][-1],
             "source": c["energy"]["front_single"]["source"]},
            {"what": f"{name[cid]}: multi-pass energy front", "value": c["energy"]["front_multi"]["y"][-1],
             "source": c["energy"]["front_multi"]["source"]},
            {"what": f"{name[cid]}: single-pass selection at the energy budget (%)", "value": lo,
             "source": c["at_cap"]["single"]["source"], "network": c["at_cap"]["single"].get("network")},
            {"what": f"{name[cid]}: multi-pass selection at the energy budget (%)", "value": hi,
             "source": c["at_cap"]["multi"]["source"], "network": c["at_cap"]["multi"].get("network")},
        ]

    span_xlabel(fig, tops[0], tops, f"Energy per inference ({MU}J)")


    low_b, low_h = LAYOUT["low_b"], LAYOUT["low_h"]
    band = LAYOUT["band"]
    pad_t = LAYOUT["title_pad_low"] + band


    rows, y_at = {}, low_h
    for cid in ORDER:
        height = low_h / len(ORDER)
        rows[cid] = y_at - height / 2.0
        y_at -= height
    rlim = (0.0, low_h)

    def band_label(ax, x, text, ha):
        return ax.annotate(text, xy=(x, 1), xycoords="axes fraction", xytext=(0, KEY_UP), textcoords="offset points",
                           ha=ha, va="bottom", fontsize=FS["small"], color=INK2)


    g_l = left + 2 * (pw + gap)
    e_l = 53.0
    e_w = g_l - 41.0 - e_l
    ax = ax_pt(fig, e_l, low_b, e_w, low_h)
    lane_seeds, lane_mean, lane_zero = 3.2, 0.0, -3.4


    def seeds_drawn(cid):
        f5 = CELLS[cid]["five_seed"]
        return f5["seeds"]["values"] if f5 and not f5.get("same_read_as") and not f5.get("descriptive") else []

    def upper_marks(cid):
        f5 = CELLS[cid]["five_seed"]
        if f5 and f5.get("descriptive"):
            return [(f5["lower95"]["value"], f5["mean"]["value"])]
        return [(v, v) for v in seeds_drawn(cid)]


    g_max = max([hi for cid in ORDER for _, hi in upper_marks(cid)]
                + [CELLS[cid]["front_gap"]["value"] for cid in ORDER])
    ax.axvline(0, color=GRAY2, linewidth=S.LW["axis"], zorder=1)

    label_col = max([max(seeds_drawn(cid)) for cid in ORDER if seeds_drawn(cid)] or [0.0])
    for cid in ORDER:
        c, y = CELLS[cid], rows[cid]
        fs5, fgap = c["five_seed"], c["front_gap"]
        pend = c.get("five_seed_pending")


        ym = y + lane_mean
        y0 = y + lane_zero if fs5 and y + lane_zero - GAP_TICK / 2.0 >= AXIS_AIR else y
        if fs5:
            sv = fs5["seeds"]["values"]
            mean, low = fs5["mean"]["value"], fs5["lower95"]["value"]
            again = fs5.get("same_read_as")
            desc = bool(fs5.get("descriptive"))
            if not again and not desc:
                ax.plot(sv, [y + lane_seeds] * len(sv), linestyle="none", marker="o", markersize=MARK["seed"],
                        markerfacecolor=S.tint(BLUE, 0.45), markeredgewidth=0, zorder=3)
            if desc:
                hb = DESCRIPTIVE_BAR["height"]
                ax.add_patch(Rectangle((low, ym - hb / 2.0), mean - low, hb, facecolor="white", edgecolor=BLUE,
                                       linewidth=DESCRIPTIVE_BAR["edge"], joinstyle="miter", zorder=4))
            else:
                ax.plot([low, mean], [ym] * 2, color=BLUE, linewidth=1.2, solid_capstyle="butt", zorder=4)
            ax.plot([low], [ym], linestyle="none", marker="|", markersize=MARK["bound"], markeredgewidth=1.2,
                    color=BLUE, zorder=4)
            ax.plot([mean], [ym], linestyle="none", marker="o", markersize=MARK["mean"], markerfacecolor=BLUE,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            if again:

                S.direct_label(ax, low, ym, WORDS["same_read"].format(row=name[again][0].lower() + name[again][1:]),
                               color=INK2, dx=-5, ha="right", size=FS["small"],
                               path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")])
            else:


                box = BOX * FS["label"] / 2.0
                reach = ([] if desc else sv) + [mean] + [
                    x for x, half in ((fgap["value"], GAP_TICK / 2.0),) if abs(y0 - ym) < box + half]
                S.direct_label(ax, max(reach), ym, fs5["mean"]["text"], color=BLUE, dx=5, size=FS["label"])
            same = f" (the read of {name[again]}: the same pair)" if again else ""
            kind = " (descriptive: the pair was declared after its networks' test reads)" if desc else ""
            keys += [
                {"what": f"{name[cid]}: five-seed mean gain, test images (pp){same}{kind}", "value": fs5["mean"]["text"],
                 "key": fs5["mean"]["key"], "source": fs5["mean"]["source"]},
                {"what": f"{name[cid]}: one-sided 95% lower bound (pp){same}{kind}", "value": low,
                 "key": fs5["lower95"]["key"], "source": fs5["lower95"]["source"]},
            ]
            if not again and not desc:
                keys.append({"what": f"{name[cid]}: the five seeds' gains (pp)", "value": sv,
                             "source": fs5["seeds"]["source"]})
        elif pend:

            S.direct_label(ax, label_col, y + lane_mean, WORDS["pending"], color=GRAY, dx=5, size=FS["small"])
            keys.append({"what": f"{name[cid]}: five-seed gain of the cell's selection, test images", "value":
                         pend["text"], "key": pend["key"], "source": pend["source"]})


        ax.plot([fgap["value"]], [y0], linestyle="none", marker="|", markersize=GAP_TICK, markeredgewidth=1.0,
                color=GRAY, zorder=4)
        keys += [
            {"what": f"{name[cid]}: gap of the fronts at the energy budget, seed 0 (pp)", "value": fgap["text"],
             "key": fgap["key"]},
        ]
    ax.set_xlim(-0.3, g_max + 1.3)
    ax.set_ylim(*rlim)
    pt_gain = e_w / (g_max + 1.6)
    for upper, lower in zip(ORDER, ORDER[1:]):
        if CELLS[upper]["five_seed"] and upper_marks(lower):
            side = min(max(lo - s0, s0 - hi, 0.0) for lo, hi in upper_marks(lower)
                       for s0 in (CELLS[upper]["front_gap"]["value"],)) * pt_gain
            what = "bar" if (CELLS[lower]["five_seed"] or {}).get("descriptive") else "seeds"
            print(f"{'layout' if side >= 4.0 else 'FAIL layout'} {NAME}: seed-0 marks of {name[upper]} to the {what} of "
                  f"{name[lower]}, sideways: {side:.1f} pt (least 4.0 pt)")
    ax.set_yticks([rows[cid] for cid in ORDER], [name[cid] for cid in ORDER])
    ax.tick_params(axis="y", length=0, labelsize=FS["tick"])
    ax.xaxis.set_major_locator(FixedLocator([0, 2, 4, 6]))
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    S.grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy gain (pp)")
    S.panel_title(ax, "e", WORDS["gain_title"], x=-(e_l - left) / e_w, pad=pad_t)
    h1, = ax.plot([], [], linestyle="none", marker="o", markersize=MARK["mean"], markerfacecolor=BLUE,
                  markeredgecolor="white", markeredgewidth=0.5)
    h3, = ax.plot([], [], linestyle="none", marker="|", markersize=GAP_TICK, markeredgewidth=1.0, color=GRAY)
    h4, = ax.plot([], [], linestyle="none", marker="o", markersize=MARK["seed"], markerfacecolor=S.tint(BLUE, 0.45),
                  markeredgewidth=0)

    leg = ax.legend([h1, h4, h3], list(WORDS["legend"]), loc="lower left",
                    bbox_to_anchor=(-(e_l - left) / e_w, 1.0 + KEY_UP / low_h), ncol=len(WORDS["legend"]),
                    fontsize=FS["small"],
                    frameon=False, handlelength=0.9, handletextpad=0.3, columnspacing=1.3, borderaxespad=0.0,
                    borderpad=0.0, labelcolor=INK2)
    gain_ax = ax


    g_gap = 5.0
    g_w = (W_PT - right - g_l - 3 * g_gap) / 4
    t_all = [x for cid in ORDER for x in CELLS[cid]["time"]["members"]["x"]]
    tlim = (min(t_all) * 0.6, max(t_all) * 1.3)


    t_dec = range(math.floor(math.log10(tlim[0])), math.ceil(math.log10(tlim[1])) + 1)
    t_major = [m * 10.0 ** e for e in t_dec for m in (1, 3) if tlim[0] <= m * 10.0 ** e <= tlim[1]]
    t_minor = [m * 10.0 ** e for e in t_dec for m in (2, 4, 5, 6, 7, 8, 9) if tlim[0] <= m * 10.0 ** e <= tlim[1]]
    thin = []
    for i, cid in enumerate(ORDER):
        c = CELLS[cid]
        ax = ax_pt(fig, g_l + i * (g_w + g_gap), low_b, g_w, low_h)
        thin.append(ax)
        fronts(ax, c["time"])
        ax.set_xscale("log")
        ax.set_xlim(*tlim)
        ax.set_ylim(*ylim_time)
        ax.xaxis.set_major_locator(FixedLocator(t_major))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _=None: plain(round(v, 12))))
        ax.xaxis.set_minor_locator(FixedLocator(t_minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        S.grid(ax)
        band = band_label(ax, 0.5, name[cid], "center")
        if i == 0:
            band0 = band
        if i == 0:
            ax.set_ylabel("Accuracy (%)", labelpad=YLABEL_PAD)
            S.panel_title(ax, "f", "Time fronts", pad=pad_t)
        else:
            ax.tick_params(labelleft=False)

        kept = "" if c["time"].get("formed_at_nominal_point", True) else " (kept: not formed at the nominal point)"
        keys += [
            {"what": f"{name[cid]}: single-pass time front{kept}", "value": c["time"]["front_single"]["y"][-1],
             "source": c["time"]["front_single"]["source"]},
            {"what": f"{name[cid]}: multi-pass time front{kept}", "value": c["time"]["front_multi"]["y"][-1],
             "source": c["time"]["front_multi"]["source"]},
            {"what": f"{name[cid]}: time range of the fronts' members (ms)",
             "value": [min(c["time"]["members"]["x"]), max(c["time"]["members"]["x"])],
             "source": c["time"]["members"]["source"]},
        ]
    span_xlabel(fig, thin[0], thin, "Time per inference (ms)")

    layout_reading(fig, tops, gain_ax, thin, leg, pool_key, band0, focal_end)
    S.pending_key_at(fig, -8.0, 0.0, ha="right", anchor=band0,
                     words="Release version")
    S.save_fig(fig, NAME, rows=keys)
    S.save_keys(NAME, keys)


if __name__ == "__main__":
    main()
