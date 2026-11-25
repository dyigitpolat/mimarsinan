""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.INK_CLEAR_PT = 1.5
S.MIN_FONT_PT = 6.0
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

NAME = "fig_block"
D = S.load(NAME)
ROWS = D["rows"]
FLOOR = D["floor"]["hours"]["value"]
RATE = D["rate"]["per_second"]["value"]
KEYS = []

TONE = {"accent": S.C["blue"], "quiet": S.C["ink2"]}
CORE = S.ROLE["cores"]
FS = {"title": S.FS["title"], "focal": 8.0, "name": 7.0, "label": 7.0, "tick": 6.5, "small": 6.0}
N_GLYPH = 4
EDGE = 0.6
GAP_WHITE = 0.9
SQ = 2.7
GAP = GAP_WHITE + EDGE
BAR = 4.4
CAP_A, CAP_B = 5.5, 4.0
MEW = 1.1
MARK_FOOT = 5.2
TEXT_FOOT = 10.7
LABEL_DX = 4.0
MS_B = 3.8
REACH_LABEL = "Required rate met"
REACH_DX = 2.0

BATCH_LABEL = {"every": "All"}

PRINTED = {"Required life, 5 years": "Service life, 5 years"}
FIT_ONE = "One core per tile"
NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four"}
LATE = []
ROW_BAND = S.tint(S.C["gray3"], 0.45)
BATCH_HEAD = "Batch"
BATCH_DX = -4.5
BATCH_HEAD_UP = 0.3
VALUE_DX = 2.0
VALUE_CLEAR = 1.5
RATE_DASH = (3, 2)
LANE_PT, LANE_ROOM = 6.0, 6.4
LANE_UP = 1.0
KEY_W, KEY_GAP = 9.0, 3.0


W_PT = S.COLUMN_PT
ROW_PT, HEAD_PT = 18.4, 8.3
BOTTOM, TOP = 21.0, 10.0
TITLE_PAD, XLABEL_UP = 0.3, 0.2
ROW_H = [max(ROW_PT, (len(r["rate_lines"]) - 1) * LANE_PT + LANE_ROOM) for r in ROWS]
BODY = sum(ROW_H)
H_PT = BOTTOM + BODY + HEAD_PT + TOP
NAME_GAP = 1.6
GLYPH_H = 2 * SQ + GAP
FIT_DROP = 0.5
PLOT_W = 88.7
X_LAB, X_A, X_B = (0.6, 63.0), (67.5, 67.5 + PLOT_W), (W_PT - 1.5 - PLOT_W, W_PT - 1.5)
FIT_GAP = 2.4


REF = [None]


def keyrow(what, v, key=None, source=None):
    KEYS.append({"what": what, "value": v, **({"key": key} if key else {}), **({"source": source} if source else {}),
                 **({"ref": REF[0]} if REF[0] else {})})


def axes_pt(fig, x):
    ""
    return fig.add_axes([x[0] / W_PT, BOTTOM / H_PT, (x[1] - x[0]) / W_PT, (H_PT - BOTTOM - TOP) / H_PT])


def pt_patch(ax, x, y, dx, dy, w, h, **kw):
    ""
    tr = (mtransforms.Affine2D().scale(1 / 72.0) + ax.figure.dpi_scale_trans
          + mtransforms.ScaledTranslation(x, y, ax.transData))
    p = Rectangle((dx, dy), w, h, transform=tr, clip_on=False, **kw)
    ax.add_patch(p)
    return p


def fit_blocks(share):
    ""
    n_cores = max(1, round(N_GLYPH * share))
    return n_cores, N_GLYPH // n_cores


def fit_glyph(ax, x, y, dx, dy, share, color=CORE):
    ""
    n_cores, span = fit_blocks(share)
    for i in range(N_GLYPH):
        pt_patch(ax, x, y, dx + i * (SQ + GAP), dy + SQ + GAP, SQ, SQ, facecolor=S.C["white"],
                 edgecolor=S.C["gray"], linewidth=EDGE)
    w = span * SQ + (span - 1) * GAP
    for j in range(n_cores):
        pt_patch(ax, x, y, dx + j * span * (SQ + GAP), dy, w, SQ, facecolor=color, edgecolor=color, linewidth=EDGE)
    return N_GLYPH * (SQ + GAP) - GAP


def drawn_pt(ax, lo, hi, y):
    ""
    (x0, _), (x1, _) = ax.transData.transform([(lo, y), (hi, y)])
    return abs(x1 - x0) * 72.0 / ax.figure.dpi


def whisker(ax, lo, hi, y, color, *, cap, ms, reach=None):
    ""
    ax.plot([lo, hi], [y, y], color=color, lw=S.LW["thin"], solid_capstyle="butt", zorder=3)
    if reach is not None and hi >= reach:
        ax.plot([max(lo, reach), hi], [y, y], color=color, lw=S.LW["heavy"], solid_capstyle="butt", zorder=3)
    if drawn_pt(ax, lo, hi, y) > ms + MEW:
        ax.plot([lo, hi], [y, y], ls="none", marker="|", ms=cap, mew=S.LW["thin"], color=color, zorder=3)


def mark(ax, x, y, color, ms):
    ""
    ax.plot([x], [y], marker="o", ms=ms, mfc=S.C["white"], mec=color, mew=MEW, zorder=5)


def baseline_px(t, r):
    ""
    d = max(r.get_text_width_height_descent(u, t.get_fontproperties(), ismath=False)[2] for u in (t.get_text(), "lp"))
    return t.get_window_extent(r).y0 + d


def forms_key(fig, x_pt, lines, color):
    ""
    r, px = fig.canvas.get_renderer(), fig.dpi / 72.0
    fs = FS["small"]
    cap = TextPath((0, 0), "E", size=fs, prop=FontProperties(family=plt.rcParams["font.family"])).get_extents().height
    to_fig = fig.transFigure.inverted()
    for form, name, beside in lines:
        base = baseline_px(beside, r)
        (x0, y), (x1, _), (xt, yt) = to_fig.transform([(x_pt * px, base + cap / 2.0 * px),
                                                       ((x_pt + KEY_W) * px, base + cap / 2.0 * px),
                                                       ((x_pt + KEY_W + KEY_GAP) * px, base)])
        kw = dict(transform=fig.transFigure, color=color)
        if form == "estimate":
            fig.add_artist(Line2D([(x0 + x1) / 2.0], [y], ls="none", marker="o", ms=S.MS, mfc=S.C["white"],
                                  mec=color, mew=MEW, **kw))
        else:
            fig.add_artist(Line2D([x0, x1], [y, y], lw=S.LW["thin"], solid_capstyle="butt", **kw))
            fig.add_artist(Line2D([x0, x1], [y, y], ls="none", marker="|", ms=CAP_B, mew=S.LW["thin"], **kw))
        fig.text(xt, yt, name, ha="left", va="baseline", fontsize=fs, color=S.C["ink2"])


S.setup()
plt.rcParams["figure.constrained_layout.use"] = False
fig = plt.figure(figsize=S.figsize("column", H_PT))
lab, a, b = axes_pt(fig, X_LAB), axes_pt(fig, X_A), axes_pt(fig, X_B)
n = len(ROWS)
tops = [BODY - sum(ROW_H[:i]) for i in range(n)]
ys = [t - h / 2.0 for t, h in zip(tops, ROW_H)]
ROW_TOP = BODY
YLIM = (0.0, BODY + HEAD_PT)
HEAD_UP = 0.6
HEAD_Y = BODY + 0.5 * HEAD_PT + HEAD_UP

for ax in (a, b):
    ax.patch.set_visible(False)
for i in range(1, n, 2):
    fig.add_artist(Rectangle((0.0, (BOTTOM + tops[i] - ROW_H[i]) / H_PT), 1.0, ROW_H[i] / H_PT,
                             transform=fig.transFigure, facecolor=ROW_BAND, edgecolor="none", zorder=-1))


def cap_half(size):
    ""
    prop = FontProperties(family=plt.rcParams["font.family"])
    return TextPath((0, 0), "H", size=size, prop=prop).get_extents().height / 2.0


def math_rise(text, size):
    ""
    prop = FontProperties(family=plt.rcParams["font.family"], size=size)
    return float(MathTextParser("path").parse(text, 72, prop)[3][0][-1])


def math_ink(text, size):
    ""
    prop = FontProperties(family=plt.rcParams["font.family"], size=size)
    box = TextPath((0, 0), text, size=size, prop=prop).get_extents()
    return box.x1, box.y1


def batch_name(ax, y, text):
    ""
    S.direct_label(ax, 0, y, text, dx=BATCH_DX, dy=-cap_half(FS["small"]), ha="right", va="baseline",
                   size=FS["small"], color=S.C["ink2"])


lab.set_xlim(0, 1); lab.set_ylim(*YLIM); lab.axis("off")
for i, (row, y, rh) in enumerate(zip(ROWS, ys, ROW_H)):
    text_dy = TEXT_FOOT - rh / 2.0
    glyph_dy = text_dy - NAME_GAP - GLYPH_H
    S.direct_label(lab, 0, y, row["name"], dx=0, dy=text_dy, va="baseline", size=FS["name"])
    REF[0] = f"fit/{i}"
    _m = S.marks(fig, REF[0], series="fit").__enter__()
    fit = row["fit"]
    share = fit["cores_per_tile"]["value"]
    gw = fit_glyph(lab, 0, y, 0.6, glyph_dy, share)
    keyrow(f"{row['name']}: cores per tile (pictogram)", share, source=fit["cores_per_tile"]["source"])
    if "cores" in fit:
        per_core = 1.0 / share
        if abs(per_core - round(per_core)) > 1e-9 or fit_blocks(share)[1] != round(per_core):
            raise SystemExit(f"{NAME}: {row['name']}: {share:g} cores per tile is no whole count of tiles per core "
                             f"drawn by the pictogram")

        text = f"{NUMBER_WORDS.get(round(per_core), str(round(per_core)))} tiles a core"
        REF[0] = None
        keyrow(f"{row['name']}: cores (not printed, the label prints the tiles per core)",
               fit["cores"]["value"], key=fit["cores"]["key"])
        REF[0] = f"fit/{i}"
        LATE.append((f"{row['name']}: tiles per core (the label)", round(per_core),
                     fit["cores_per_tile"]["source"], REF[0]))
    else:
        if share != 1:
            raise SystemExit(f"{NAME}: {row['name']} has {share} cores per tile and no core count")
        if fit_blocks(share) != (N_GLYPH, 1):
            raise SystemExit(f"{NAME}: {row['name']}: {FIT_ONE!r} beside {fit_blocks(share)[0]} core blocks under "
                             f"{N_GLYPH} tiles")
        text = FIT_ONE
    print(f"{NAME}: {row['name']}: pictogram of {N_GLYPH} tiles over {fit_blocks(share)[0]} core block(s), each under "
          f"{fit_blocks(share)[1]} tile(s) ({share:g} cores per tile in the data), {GAP_WHITE} pt of white between blocks, "
          f"label {text!r}")
    S.direct_label(lab, 0, y, text, dx=gw + FIT_GAP, dy=glyph_dy + GLYPH_H / 2.0 - FIT_DROP, va="center", size=FS["small"],
                   color=S.C["ink2"])
    _m.__exit__(None, None, None)
REF[0] = None

for dx, text, face, edge in ((0.6, "Tile", S.C["white"], S.C["gray"]), (20.6, "Core", CORE, CORE)):
    pt_patch(lab, 0, HEAD_Y, dx, -SQ / 2, SQ, SQ, facecolor=face, edgecolor=edge, linewidth=EDGE)
    S.direct_label(lab, 0, HEAD_Y, text, dx=dx + SQ + 1.6, size=FS["small"], color=S.C["ink2"])


XA = (1e3, 3e12)
a.set_xscale("log"); a.set_xlim(*XA); a.set_ylim(*YLIM)
a.axvspan(XA[0], FLOOR, ymax=(ROW_TOP - YLIM[0]) / (YLIM[1] - YLIM[0]), color=S.C["gray3"], lw=0, zorder=0)

a.plot([FLOOR, FLOOR], [YLIM[0], ROW_TOP], color=S.C["ink"], lw=S.LW["thin"], zorder=6, solid_capstyle="butt")

floor_label = PRINTED.get(D["floor"]["label"], D["floor"]["label"])
floor_w = S.text_width_pt(fig, floor_label, FS["tick"])
floor_dx = max(0.0, 0.5 + floor_w / 2.0 - drawn_pt(a, XA[0], FLOOR, HEAD_Y))
S.direct_label(a, FLOOR, HEAD_Y, floor_label, dx=floor_dx, ha="center", size=FS["tick"], color=S.C["ink"])
keyrow("required service life (h)", FLOOR, key=D["floor"]["hours"]["key"])
for i, (row, y, rh) in enumerate(zip(ROWS, ys, ROW_H)):
    REF[0] = f"a/{i}"
    _m = S.marks(fig, REF[0], series="a").__enter__()
    col = TONE[row["tone"]]
    lt = row["lifetime_h"]
    lo, nom, hi = (lt[c]["value"] for c in ("low", "nominal", "high"))
    accepted = lo >= FLOOR
    ym = y - rh / 2.0 + MARK_FOOT
    label_dy = TEXT_FOOT - MARK_FOOT
    if accepted:
        a.plot([FLOOR, nom], [ym, ym], color=S.tint(col, 0.62), lw=BAR, solid_capstyle="butt", zorder=2)
    if hi > lo:
        whisker(a, lo, hi, ym, col, cap=CAP_A, ms=S.MS + 0.4)
    mark(a, nom, ym, col, ms=S.MS + 0.4)
    label = row["verdict"]["label"]
    if row is ROWS[0]:
        S.focal(a, label, (FLOOR, ym), (LABEL_DX, label_dy), textcoords="offset points", arrow=False, ha="left",
                va="baseline").set_fontsize(FS["focal"])
    else:
        S.direct_label(a, FLOOR, ym, label, color=col, dx=LABEL_DX, dy=label_dy, ha="left", va="baseline",
                       size=FS["label"])
    for c in ("low", "nominal", "high"):
        keyrow(f"{row['name']}: lifetime, {c} corner (h)", lt[c]["value"], key=lt[c]["key"])
    keyrow(f"{row['name']}: verdict", row["verdict"]["value"], key=row["verdict"]["key"])
    _m.__exit__(None, None, None)
REF[0] = None
a.set_xticks([1e3, 1e6, 1e9, 1e12]); a.minorticks_off()
a.set_xlabel("Chip lifetime (h)", fontsize=FS["label"], va="bottom")
S.panel_title(a, "a", "Lifetime", pad=TITLE_PAD)


XB = (-22, 106)
b.set_xlim(*XB); b.set_ylim(*YLIM)
rate_foot = YLIM[0]

S.direct_label(b, RATE, HEAD_Y, D["rate"]["label"], dx=0, ha="center", size=FS["tick"], color=S.C["ink2"])
S.direct_label(b, 0, HEAD_Y, BATCH_HEAD, dx=BATCH_DX, dy=BATCH_HEAD_UP, ha="right", size=FS["small"],
               color=S.C["ink2"])
keyrow("declared rate (per second)", RATE, key=D["rate"]["per_second"]["key"])
for i, (row, y, rh) in enumerate(zip(ROWS, ys, ROW_H)):
    REF[0] = f"b/{i}"
    _m = S.marks(fig, REF[0], series="b").__enter__()
    col = TONE[row["tone"]]
    lines = row["rate_lines"]
    m = len(lines)
    names = [BATCH_LABEL.get(ln["batch"], f"{ln['batch']}") for ln in lines]
    for j, ln in enumerate(lines):
        yy = y + LANE_UP + LANE_PT * ((m - 1) / 2.0 - j)
        r = ln["rate"]
        unresolved = r["high"] < 0.01 * RATE
        if m == 1:
            yy = y - rh / 2.0 + MARK_FOOT
        whisker(b, r["low"], r["high"], yy, col, cap=CAP_B, ms=MS_B, reach=RATE)
        mark(b, r["nominal"], yy, col, ms=MS_B)
        batch_name(b, yy, names[j])
        if row is ROWS[0] and j == 0:
            if r["high"] >= RATE:
                raise SystemExit(f"{NAME}: the first batch line reaches the declared rate: no room for {REACH_LABEL!r}")
            S.direct_label(b, RATE, yy, REACH_LABEL, dx=REACH_DX, dy=-cap_half(FS["small"]), ha="left", va="baseline",
                           size=FS["small"], color=col)
        if unresolved and m == 1:
            mant, exp = f"{r['nominal']:.1e}".split("e")
            value = f"${mant}\\times10^{{{int(exp)}}}$"
            dx = MS_B / 2.0 + MEW / 2.0 + VALUE_DX
            base = yy - cap_half(FS["small"])
            S.direct_label(b, r["nominal"], yy, value, dx=dx, dy=base - yy - math_rise(value, FS["small"]),
                           ha="left", va="baseline", size=FS["small"], color=col)
            width, rise = math_ink(value, FS["small"])
            end = drawn_pt(b, 0, r["nominal"], yy) + dx + width
            if end + VALUE_CLEAR > drawn_pt(b, 0, RATE, yy):
                period = sum(RATE_DASH) * S.LW["thin"]
                rate_foot = max(rate_foot, period * -(-(base + rise + VALUE_CLEAR - YLIM[0]) // period) + YLIM[0])
                print(f"{NAME}: {row['name']}: value {mant}e{int(exp)} beside its mark, ink {base:.2f} to {base + rise:.2f} "
                      f"pt over the axis, right end {end:.1f} pt from zero (declared rate at "
                      f"{drawn_pt(b, 0, RATE, yy):.1f} pt): the rate's line ends {rate_foot:.1f} pt over the axis, "
                      f"{rate_foot - base - rise:.2f} pt over the value's ink")
        for c in ("low", "nominal", "high"):
            keyrow(f"{row['name']}: sustainable rate, batch {ln['batch']}, {c} (per second) = 1 / write time",
                   r[c], key=ln["key"])
    _m.__exit__(None, None, None)
REF[0] = None
b.plot([RATE, RATE], [rate_foot, ROW_TOP], color=S.C["gray"], lw=S.LW["thin"], ls=(0, RATE_DASH), zorder=1)
b.set_xticks([0, RATE, 2 * RATE, 3 * RATE], [f"{v:g}" for v in (0, RATE, 2 * RATE, 3 * RATE)])
b.spines["bottom"].set_bounds(0, XB[1])
b.set_xlabel("Rate (per second)", x=(XB[1] / 2.0 - XB[0]) / (XB[1] - XB[0]), ha="center",
             fontsize=FS["label"], va="bottom")
S.panel_title(b, "b", "Sustainable rate", pad=TITLE_PAD)

for ax in (a, b):
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FS["tick"], pad=1.0, length=2.0)

    ax.xaxis.set_label_coords(ax.xaxis.label.get_position()[0], -(BOTTOM - XLABEL_UP) / (H_PT - BOTTOM - TOP))
    for yb in tops[1:]:
        ax.axhline(yb, color=S.C["gray3"], lw=S.LW["grid"], zorder=1)


fig.canvas.draw()
forms_key(fig, X_LAB[0] + 0.6, (("band", "Corner range", b.get_xticklabels()[0]),
                                ("estimate", "Nominal estimate", b.xaxis.label)), TONE["accent"])

for what, v, src, ref in LATE:
    KEYS.append({"what": what, "value": v, "source": src, "ref": ref})
for k in KEYS:
    k.setdefault("ref", None)
S.save_fig(fig, NAME, rows=KEYS)
S.save_keys(NAME, KEYS)
