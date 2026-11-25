""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.INK_CLEAR_PT = 1.0

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Affine2D

NAME = "fig_card"
D = S.load(NAME)
S.setup()


W, H = S.COLUMN_PT, 107.5
LEFT, GUTTER, RIGHT = 40.0, 29.0, 5.0
BOTTOM, H_RES, GAP, TOP = 21.6, 13.5, 3.5, 12.0

PW = (W - LEFT - GUTTER - RIGHT) / 2.0
DESC_GAP = 1.2
GEO_PAD = 3.0


FS_TICK = S.FS["small"]
FS_NAME = S.FS["small"]

TITLE_PAD, TICK_PAD = 2.8, 1.5
TICK_LEN = 2.2
DOT, DOT_RES = 2.1, 1.7
UNIT, UNIT_WORDS = 1e6, "million cycles"
TITLE_AIR = 1.0
LABEL_AIR = 1.5
CORNER_LABELS = []

fig = plt.figure(figsize=S.figsize("column", H))
fig.set_layout_engine("none")


def descriptor_lines(g):
    ""
    lines_, neurons_, bits_ = (g["geometry"][k] for k in ("axon_lines", "neurons", "weight_bits"))
    return [f"{lines_:,} × {neurons_:,}, {bits_}-bit, {g['counts']['label']}"]


def focal_run(g):
    ""
    focal = dict(fontsize=S.FS["focal"], fontweight="bold", color=S.C["blue"])
    sign = dict(focal, path_effects=[patheffects.withStroke(linewidth=0.55, foreground=S.C["blue"])])
    return [("R² ", focal), ("\u2265", sign), (" " + g["r2"]["label"], focal)]


DESC_H = max(S.descriptor_height(fig, descriptor_lines(g)) for g in D["geometries"])
print(f"descriptor {NAME}: block {DESC_H:.2f} pt high")
TOP += 0.5 + DESC_GAP + DESC_H
H_PAR = H - BOTTOM - H_RES - GAP - TOP
DESCS = []
PT = Affine2D().scale(1.0 / 72.0) + fig.dpi_scale_trans


def pt_axes(x, y, w, h, **kw):
    ""
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def padded(lo, hi, frac):
    span = hi - lo
    return lo - frac * span, hi + frac * span


def bound_label(ax, symbol, value, xy):
    ""
    kw = dict(fontsize=S.FS["focal"], fontweight="bold", color=S.C["blue"])
    sign = TextArea("\u2265", textprops=dict(kw, path_effects=[patheffects.withStroke(linewidth=0.55,
                                                                                    foreground=S.C["blue"])]))
    box = HPacker(children=[TextArea(symbol, textprops=kw), sign, TextArea(value, textprops=kw)], align="baseline",
                  pad=0, sep=2.2)
    ab = AnnotationBbox(box, xy, xycoords="axes fraction", box_alignment=(0.0, 1.0), frameon=False, pad=0, zorder=6)
    ax.add_artist(ab)
    return ab


def check_corner_labels():
    ""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    out = []
    for ax_, art, side in CORNER_LABELS:
        (lx0, ly0), (lx1, ly1) = (ax_.transData.transform(p) * k for p in
                                  ((ax_.get_xlim()[0], ax_.get_ylim()[0]), (ax_.get_xlim()[1], ax_.get_ylim()[1])))

        bb = Text.get_window_extent(art, r) if isinstance(art, Text) else art.get_window_extent(r)
        cx, cy = ((bb.x1, bb.y0) if side == "above" else (bb.x0, bb.y1))
        cx, cy = cx * k, cy * k
        dx, dy = lx1 - lx0, ly1 - ly0
        dist = ((cx - lx0) * dy - (cy - ly0) * dx) / (dx * dx + dy * dy) ** 0.5
        air = (-dist if side == "above" else dist) - DOT / 2.0
        text = art.get_text() if hasattr(art, "get_text") else "R squared"
        out.append((air, f"{text!r} ({side})".replace("\\n", " / ")))
    print(f"corner labels {NAME}: " + ", ".join(f"{w} {a:.2f} pt" for a, w in out) + f" (least {LABEL_AIR})")
    bad = [w for a, w in out if a < LABEL_AIR]
    if bad:
        raise SystemExit(f"labels closer than {LABEL_AIR} pt to the marks on the line: {bad}")


def group(n):
    return f"{n:,}"


rows = []


TITLES = {"published": "Published geometry", "widened": "Wider generated core"}
PRINTED_TITLES = {"published": "Published core", "widened": "Wider core"}
NAMES = []
NAME_X = 0.6

TITLE_GAP = 6.0
PANELS = []


def place_titles():
    ""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    starts = []
    for title, axes_, x0, named in PANELS:
        ticks = [t.get_window_extent(r).x0 * k for a_ in axes_ for t in a_.get_yticklabels()
                 if t.get_visible() and t.get_text().strip()]
        start = min(ticks + ([NAME_X] if named else []))
        title.set_x((start - x0) / PW)
        starts.append(start)
    fig.canvas.draw()
    ends = [t.get_window_extent(r).x1 * k for t, _, _, _ in PANELS]
    for d, (title, _, x0, _) in zip(DESCS, PANELS):
        S.descriptor_x(d, title.get_position()[0])
    fig.canvas.draw()
    for i, (end, start) in enumerate(zip(ends, starts)):
        limit = starts[i + 1] - TITLE_GAP if i + 1 < len(starts) else W - 0.5
        if end > limit:
            raise SystemExit(f"the title of panel {i + 1} ends at {end:.1f} pt, past {limit:.1f} pt")

    floor_ = min([t.get_window_extent(r).y0 * k for t, _, _, _ in PANELS]
                 + [a.get_window_extent(r).y0 * k for d in DESCS for a in d])
    top = max(n.get_window_extent(r).y1 * k for n in NAMES)
    if top > floor_ - TITLE_AIR:
        raise SystemExit(f"the axis names of the left edge reach {top:.1f} pt, the titles start at {floor_:.1f} pt")
    print(f"axis names {NAME}: top {top:.2f} pt, titles from {floor_:.2f} pt (air {floor_ - top:.2f} pt)")
    print("titles " + NAME + ": " + ", ".join(f"({tag}) from {a:.2f} to {b:.2f} pt" for tag, a, b in
                                             zip("ab", starts, ends)))
res_all = [p["residual_pct"] for g in D["geometries"] for p in g["points"]]
res_lim = max(abs(min(res_all)), abs(max(res_all)))
res_tick = round(res_lim, 1)
res_ylim = (-1.45 * res_tick, 1.45 * res_tick)

for col, (g, tag) in enumerate(zip(D["geometries"], "ab")):
    x0 = LEFT + col * (PW + GUTTER)
    ax = pt_axes(x0, BOTTOM + H_RES + GAP, PW, H_PAR)
    rx = pt_axes(x0, BOTTOM, PW, H_RES, sharex=ax)
    xs = [p["modelled_cycles"] / UNIT for p in g["points"]]
    ys = [p["measured_cycles"] / UNIT for p in g["points"]]
    lo, hi = padded(min(xs + ys), max(xs + ys), 0.13)


    ax.plot([lo, hi], [lo, hi], color=S.C["gray"], linewidth=S.LW["thin"], zorder=2, solid_capstyle="butt")
    ref = f"card/{tag}"
    with S.marks(fig, ref, series=tag):
        ax.plot(xs, ys, linestyle="none", marker="o", markersize=DOT, markerfacecolor=S.C["blue"],
                markeredgecolor=S.C["white"], markeredgewidth=0.2, zorder=3, clip_on=False)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.tick_params(axis="x", labelbottom=False)
    title = TITLES[g["realization"]]
    PANELS.append((S.panel_title(ax, tag, PRINTED_TITLES[g["realization"]], pad=TITLE_PAD + 0.5 + DESC_GAP + DESC_H,
                                 y=1.0), (ax, rx), x0,
                   col == 0))
    with S.marks(fig, ref, series=tag):
        S.title_run(ax, PANELS[-1][0], focal_run(g))
    for a_ in (ax, rx):
        a_.tick_params(labelsize=FS_TICK, length=TICK_LEN)
    rx.tick_params(axis="x", pad=TICK_PAD)


    lab = g["r2"]
    counts = g["counts"]
    with S.marks(fig, ref, series=tag):
        DESCS.append(S.descriptor(ax, descriptor_lines(g), x=0.0, gap=DESC_GAP))


    for yv in (-res_tick, res_tick):
        rx.axhline(yv, color=S.C["gray3"], linewidth=S.LW["grid"], zorder=1)
    rx.axhline(0.0, color=S.C["gray"], linewidth=S.LW["thin"], zorder=2)
    with S.marks(fig, ref, series=tag):
        rx.plot(xs, [p["residual_pct"] for p in g["points"]], linestyle="none", marker="o", markersize=DOT_RES,
                markerfacecolor=S.C["blue"], markeredgecolor=S.C["white"], markeredgewidth=0.15, zorder=3)
    rx.set_ylim(*res_ylim)
    rx.set_yticks([-res_tick, res_tick])
    rx.set_yticklabels([f"\u2212{res_tick:g}%", f"+{res_tick:g}%"])
    if col == 0:


        for label, axes_ in ((f"Measured\n({UNIT_WORDS})", ax), ("Residual", rx)):
            NAMES.extend(S.axis_name(fig, axes_, label, x=NAME_X, size=FS_NAME))


    k_panel = len(rows)
    rows += [
        {"what": f"{title}: geometry label", "value": g["geometry"]["value"], "key": g["geometry"]["key"]},
        {"what": f"{title}: R squared label (at least)", "value": lab["label"], "key": lab["key"],
         "record_key": lab["record_key"], "source": lab["source"], "key_display": lab["display"],
         "derivation": lab["derivation"]},
        {"what": f"{title}: pass count label", "value": counts["label"],
         "source": "; ".join(counts[k]["source"] for k in ("samples", "passes_per_inference", "runs")),
         "derivation": counts["derivation"]},
        {"what": f"{title}: residual ticks (percent)", "value": res_tick, "source": "data/fig_card.json points "
         "residual_pct: the largest magnitude over both panels, to one decimal"},
    ]
    for f in g["fits"]:
        for name in ("intercept", "slope", "r2", "n"):
            rows.append({"what": f"{title}: pass {f['pass_index']} fit {name}", "value": f[name]["value"],
                         "source": f[name]["source"]})
    for p in g["points"]:
        what = f"{title}: pass {p['pass_index']} of sample {p['sample']}"
        rows.append({"what": f"{what}, measured (cycles)", "value": p["measured_cycles"],
                     "source": p["measured_source"]})
        rows.append({"what": f"{what}, modelled (cycles)", "value": p["modelled_cycles"],
                     "source": f"{p['modelled_source']} ({p['words_source']})"})
    for r_ in rows[k_panel:]:
        r_["ref"] = None if r_["what"].endswith("residual ticks (percent)") else ref

fig.text((LEFT + (W - RIGHT)) / 2.0 / W, 0.0 / H, f"Fitted pass time ({UNIT_WORDS})", ha="center", va="bottom",
         fontsize=FS_NAME, color=S.C["ink"])

place_titles()
check_corner_labels()
rows += S.caption_keys(NAME)
S.save_fig(fig, NAME, rows=rows)
S.save_keys(NAME, rows)
