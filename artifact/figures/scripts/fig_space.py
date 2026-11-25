""
import math
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

NAME = "fig_space"
D = S.load(NAME)
S.setup()
BLUE, INK, INK2 = S.C["blue"], S.C["ink"], S.C["ink2"]
EST = S.tint(BLUE, 0.74)
EST_BAND = S.tint(BLUE, 0.62)


SHARE = dict(ring=0.053, ring_edge=0.0105, built=0.056, built_edge=0.007, band=0.028, cap=0.072)
MARK_MAX = 0.07


LEAST = dict(ring=2.0, ring_edge=0.5, built=2.0, built_edge=0.25, band=1.2, cap=2.4)
LW_SPAN, LW_DASH = 0.6, 0.6
OVER, REFUSED, REFUSED_EDGE = S.C["gray2"], S.C["gray3"], S.C["gray2"]

STATUS = (("estimated", EST, EST, "Estimated"), ("over_budget", OVER, OVER, "Timed out"),
          ("refused", REFUSED, REFUSED_EDGE, "Rejected"))
KEY_SWATCH, KEY_GAP = 4.0, 2.0
KEY_SEP = 5.5
HALO = [patheffects.withStroke(linewidth=2.4, foreground="white")]

XLIM, YLIM_E, YLIM_L = (1e-6, 1.5e2), (5e-4, 1.35e5), (3e3, 1.2e13)
GAP = 0.07
GAP_D = 0.07


W_BARS, W_DOTS = 0.285, 0.265
TICK_AIR_MIN = 3.5
H_E = 1.65
X_TITLE_B = -0.30
W_NET, W_D = 1.0, 1.27
AT_E, AT_T = 0.05, 0.02


S.MIN_FONT_PT = 5.5
FS_NOTE = 5.5
FS_KEY = 5.5
FS_HOT = S.FS["focal"]
FS_MARGIN = 5.5
MARGIN_AT = 3.5
MARGIN_DX = 2.6
MARGIN_KEY_TEXT = "Least lifetime margin"
FS_MINB = 5.5
FS_KIND = 5.5


SMALLER_TEXT = "Smaller core budgets"
SMALLER_DOT_GAP = 1.6
REQUIRED_TEXT = "Service life"

BELOW_TEXT = "Below the service life"
MARGIN_WORDS = "the service life"


HEAD_A_TEXT, HEAD_A_REFUSED = "{built:,} points built as estimated", ", {refused:,} rejected"
HEAD_A_PENDING = "Points built as estimated:"
REQUIRED_AT = (3.0, 3.0)

KEY_D_LINE, KEY_D_GAP = 8.0, 2.2

LW_CONN, LW_CONN_KEY = 0.45, 0.8


HEIGHT = 133.4
FS_NAME = 6.5


HEAD = (10.4, 3.0)
PAD_T = 19.2
HEAD_D = HEAD
FS_L2 = 5.5
TICK_BASE = 0.15
TITLE_PAD = 2.6
HEAD_A = HEAD
LINE_W = 8.2
YPAD_D = (0.5, 5.0)

KEY_D = (7.3, 6.9)
KEY_AXIS_MIN = 0.5
MINB_AT, MINB_REL = (-4.0, 10.0), 0.93
KEY_ROW_SEP, KEY_LINE = 7.0, 10.0
KEY_ROOM_MIN = 6.0
DOT_SHARE = 0.84
GUTTER_MIN = 8.0
FS_CHIP = 5.5
plt.rcParams.update({"axes.titlepad": TITLE_PAD, "axes.labelpad": 0.4, "xtick.major.pad": 1.2, "axes.labelsize": S.FS["tick"],
                     "xtick.labelsize": S.FS["small"], "ytick.labelsize": S.FS["small"],
                     "figure.constrained_layout.h_pad": 0.5 / 72, "figure.constrained_layout.hspace": 0.0})
GRAY, GRAY2 = S.C["gray"], S.C["gray2"]

UNDER_OUT = dict(ls="none", marker="o", ms=2.1, color=GRAY, mew=0)
UNDER_DEP = dict(ls="none", marker="o", ms=2.2, mfc="white", mec=GRAY, mew=0.5)
OVER = dict(ls="none", marker="o", ms=2.6, mfc="white", mec=BLUE, mew=0.65)


HOLLOW = dict(ls="none", marker="o", ms=2.6, mfc="white", mec=BLUE, mew=0.6)
BUILT = dict(ls="none", marker="D", ms=3.4, color=BLUE, mec="white", mew=0.5)
MARKS = {}


def sized(ax, kind, artists):
    ""
    MARKS.setdefault(ax, {}).setdefault(kind, []).extend(artists)


def mark_sizes(h):
    ""
    return {k: max(LEAST[k], v * h) for k, v in SHARE.items()}


def decade_ticks(ax, axis, exps):
    getattr(ax, f"set_{axis}ticks")([10.0 ** k for k in exps])
    getattr(ax, f"{axis}axis").set_minor_locator(NullLocator())


def span_mark(ax, lo, hi, number, words, *, vertical=False, at=0.05, hot=False, label=True):
    ""
    kw = dict(color=BLUE if hot else INK2, lw=LW_SPAN, solid_capstyle="butt", zorder=2, clip_on=False)
    if vertical:
        sized(ax, "cap", ax.plot([at, at], [lo, hi], marker="_", markersize=3.0, markeredgewidth=LW_SPAN,
                                 transform=ax.get_yaxis_transform(), **kw))
        if label:
            corner_label(ax, (at, 1.0), number, words, hot=hot, upper=True)
        return
    sized(ax, "cap", ax.plot([lo, hi], [at, at], marker="|", markersize=3.0, markeredgewidth=LW_SPAN,
                             transform=ax.get_xaxis_transform(), **kw))
    corner_label(ax, (1.0, at), number, words, hot=hot, upper=False)


def corner_label(ax, xy, number, words, *, hot=False, upper=False):
    ""
    color = BLUE if hot else INK2
    size_w, size_n = FS_NOTE, (FS_HOT if hot else S.FS["small"])
    pitch = LINE_W + (size_n - S.FS["small"]) * 1.07
    if upper:
        x, ha, y_w = 4.0, "left", -1.5 - 0.74 * size_w
    else:
        x, ha, y_w = -0.5, "right", 2.6 + pitch
    kw = dict(xy=xy, xycoords="axes fraction", textcoords="offset points", ha=ha, va="baseline", path_effects=HALO,
              zorder=7, annotation_clip=False, color=color)
    ax.annotate(words.capitalize(), xytext=(x, y_w), fontsize=size_w, **kw)
    ax.annotate(number, xytext=(x, y_w - pitch), fontsize=size_n, fontweight="bold" if hot else "normal", **kw)


def bands(ax, xs, ys, **kw):
    ""
    cols = {}
    for x, v in zip(xs, ys):
        cols.setdefault(round(math.log10(x), 2), []).append((x, v))
    for pts in cols.values():
        x = math.exp(sum(math.log(p[0]) for p in pts) / len(pts))
        lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
        sized(ax, "band", ax.plot([x, x], [lo, hi], color=EST_BAND, lw=1.6, solid_capstyle="round", **kw))


def inside(lim, *vals):
    return all(lim[0] < v < lim[1] for v in vals)


fig = plt.figure(figsize=S.figsize("text", HEIGHT))
nets = D["networks"]


gs = fig.add_gridspec(2, 5 + len(nets), width_ratios=[W_BARS, W_DOTS, GAP] + [W_NET] * len(nets) + [GAP_D, W_D],
                      height_ratios=[H_E, 1.0])
keys = []


a = fig.add_subplot(gs[:, 0])
chips = D["chips"]
y = list(range(len(chips)))[::-1]
start = [0] * len(chips)
refused_bars = []
for field, color, _, _ in STATUS:
    w = [c[field] for c in chips]
    with S.marks(fig, "a", series="a"):
        bars = a.barh(y, w, left=start, height=0.55, color=color, edgecolor="none", linewidth=0.4)
    if field == "refused":
        refused_bars = list(bars)
    start = [l + v for l, v in zip(start, w)]
    keys += [{"what": f"(a) {c['label']}: points {field.replace('_', ' ')}", "value": c[field], "source": c["source"],
              "ref": "a"} for c in chips]
a.set_yticks(y, [c["label"] for c in chips], fontsize=FS_CHIP)
a.tick_params(axis="y", length=0, pad=3)
a.set_ylim(-0.55, len(chips) - 0.45)


over_names = blended_transform_factory(fig.transFigure, a.transAxes)
key_a, x_item = [], 0.0
shown = [st for st in STATUS if any(c[st[0]] for c in chips)]
for k, (field, face, edge, text) in enumerate(shown):
    line = 0
    base, dx = HEAD_A[line], (x_item if line == 0 else 0.0)
    sw, = a.plot([0.0], [1.0], ls="none", marker="s", ms=KEY_SWATCH, mfc=face, mec=edge, mew=0.6, clip_on=False,
                 transform=over_names + matplotlib.transforms.ScaledTranslation(dx / 72.0, (base + 2.1) / 72.0,
                                                                                fig.dpi_scale_trans))
    tx = a.annotate(text, xy=(0.0, 1.0), xycoords=over_names, xytext=(dx + KEY_SWATCH / 2.0 + KEY_GAP, base),
                    textcoords="offset points", ha="left", va="baseline", fontsize=FS_NOTE, color=INK2,
                    annotation_clip=False)
    sw.set_in_layout(False); tx.set_in_layout(False)
    key_a.append((sw, tx))
    x_item += KEY_SWATCH + KEY_GAP + S.text_width_pt(fig, text, FS_NOTE) + KEY_SEP
a.set_xlim(*S.frame_range(0, max(c["total"] for c in chips) * 1.02,
                          [{"what": "(a) points", "source": c["source"]} for c in chips[:1]], step=2000))
a.set_xticks([0, 2000, 4000])
a.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.0f}"))
label_a = fig.text(0, 0, "Points (thousands)", ha="right", va="baseline", fontsize=S.FS["tick"], color=INK)
label_a.set_in_layout(False)
a.spines["left"].set_visible(False)
title_a = fig.text(0, 0, f"(a) {D['chips_total']['display']} explored points", ha="left", va="baseline",
                   fontsize=S.FS["title"], fontweight="bold", color=INK)
title_a.set_in_layout(False)
S.relabel(fig, "a", title_a, "(a) Explored points")

keys.append({"what": "(a) total of the explored points", "value": D["chips_total"]["display"], "key": "clfivepoints",
             "ref": "a"})


d = fig.add_subplot(gs[:, 1], sharey=a)
n_s = D["samples_per_chip"]["value"]
tot = D["samples_total"]
dots_a, rings_a = [], []
for yi, c in zip(y, chips):
    k = c["samples_equal"]
    with S.marks(fig, "a", series="a"):
        dots_a += d.plot(range(k), [yi] * k, ls="none", marker="o", ms=1.6, color=BLUE, mew=0)
        rings_a += d.plot(range(k, c["samples"]), [yi] * (c["samples"] - k), ls="none", marker="x", ms=1.4,
                          color=S.C["gray"], mew=0.45)
    keys.append({"what": f"(a) {c['label']}: sampled points built and equal to the estimate, of {c['samples']}",
                 "value": k, "source": c["samples_source"], "ref": "a"})
d.set_xlim(-0.9, n_s - 0.1)
d.tick_params(axis="y", length=0, labelleft=False)
d.set_xticks([])
for s in d.spines.values():
    s.set_visible(False)


def head_a_words(tot):
    ""
    built, refused, sampled = tot["equal"], tot["refused_before_build"], tot["sampled"]
    if built + refused != sampled:
        raise SystemExit(f"FAIL {NAME}: (a): {built} points built as estimated and {refused} refused are not the "
                         f"{sampled} sampled points")
    return HEAD_A_TEXT.format(built=built) + (HEAD_A_REFUSED.format(refused=refused) if refused else "")


head_d = [d.annotate(head_a_words(tot), xy=(n_s - 1, 1.0),
                     xycoords=d.get_xaxis_transform(),
                     xytext=(1.1, HEAD_A[1]), textcoords="offset points", ha="right", va="baseline",
                     fontsize=FS_NOTE, color=INK, annotation_clip=False)]
head_d[0].set_in_layout(False)
S.relabel(fig, "a", head_d[0], f"{HEAD_A_PENDING} {S.PENDING_TOKEN}")

keys.append({"what": "(a) sampled points built and equal to the estimate, of the sampled points",
             "value": tot["display"], "key": tot["key"], "ref": "a"})
keys.append({"what": "(a) sampled points with a named refusal before a build (gray hollow dots) ("
                     "drawn as gray crosses)",
             "value": tot["refused_before_build"], "key": tot["key"], "ref": "a"})


floor = D["floor"]["hours"]
top, bot = [], []
margins = []
WIDEST = max(((j, axis) for j in range(len(nets)) for axis in ("time", "energy")),
             key=lambda t: nets[t[0]]["span"][t[1] + "_factor"]["value"])
for j, n in enumerate(nets):
    b = fig.add_subplot(gs[0, 3 + j], sharey=top[0] if top else None)
    c = fig.add_subplot(gs[1, 3 + j], sharex=b, sharey=bot[0] if bot else None)
    top.append(b); bot.append(c)
    sp = n["span"]
    if not (inside(XLIM, *n["time_s"], *(p["time_s"] for p in n["checked"]))
            and inside(YLIM_E, *n["energy_mj"], *(p["energy_mj"] for p in n["checked"]))
            and inside(YLIM_L, floor, *sp["lifetime_h"])):
        raise SystemExit(f"FAIL {NAME}: {n['label']}: a plotted value lies outside the fixed axis limits")
    small = n["smaller_chip"]


    idx = [i for i, s in enumerate(small) if s]
    mb_, mc_ = S.marks(fig, f"b/{j}", series="b"), S.marks(fig, f"c/{j}", series="c")
    with mb_:
        bands(b, [n["time_s"][i] for i in idx], [n["energy_mj"][i] for i in idx], zorder=3)
    il = [i for i in idx if n["lifetime_h"][i] is not None]
    with mc_:
        bands(c, [n["time_s"][i] for i in il], [n["lifetime_h"][i] for i in il], zorder=3)
    idx = [i for i, s in enumerate(small) if not s]
    with mb_:
        sized(b, "ring", b.plot([n["time_s"][i] for i in idx], [n["energy_mj"][i] for i in idx], zorder=4, **HOLLOW))
    il = [i for i in idx if n["lifetime_h"][i] is not None]
    with mc_:
        sized(c, "ring", c.plot([n["time_s"][i] for i in il], [n["lifetime_h"][i] for i in il], zorder=4, **HOLLOW))
    if n["checked"]:
        with S.marks(fig, f"b/{j}/built", series="b"):
            sized(b, "built", b.plot([p["time_s"] for p in n["checked"]], [p["energy_mj"] for p in n["checked"]],
                                     zorder=6, **BUILT))
    b.set_xscale("log"); b.set_yscale("log"); c.set_yscale("log")
    c.axhline(floor, color=S.C["gray"], lw=LW_DASH, ls=(0, (3, 2)), zorder=1)
    b.tick_params(labelbottom=False)
    if j:
        b.tick_params(labelleft=False); c.tick_params(labelleft=False)


    acc = f"{n['accuracy']['pct']:.1f}%"
    kw = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline",
              annotation_clip=False)
    name, _, kind = n["label"].partition(", ")
    lead = f"{n['configurations']} points, accuracy"
    head_b = [
        b.annotate(name, xytext=(0, HEAD[0]), fontsize=FS_NAME, fontweight="bold", color=INK, **kw),
        b.annotate(lead, xytext=(0, HEAD[1]), fontsize=FS_L2, color=INK2, **kw),
        b.annotate(acc, xytext=(S.text_width_pt(fig, lead, FS_L2) + 2.2, HEAD[1]), fontsize=FS_L2,
                   fontweight="bold", color=INK, **kw)]
    if kind:
        head_b.append(b.annotate(f", {kind}", xytext=(S.text_width_pt(fig, name, FS_NAME, "bold"), HEAD[0]),
                                 fontsize=FS_KIND, color=INK, **kw))
    for h in head_b:
        h.set_in_layout(False)
    S.add_marks(fig, f"b/{j}", [head_b[1]], series="b")
    S.add_marks(fig, f"b/{j}/acc", [head_b[2]], series="b")
    keys += [
        {"what": f"(b) {n['label']}: checked accuracy carried", "value": acc, "key": n["accuracy"]["key"],
         "ref": f"b/{j}/acc"},
        {"what": f"(b) {n['label']}: points, energy (mJ) against time per inference (s)",
         "value": n["configurations"], "source": n["source"], "ref": f"b/{j}"},
        {"what": f"(c) {n['label']}: lifetime (h) of the points that write a cell per inference",
         "value": sp["lifetime_configurations"], "source": n["source"], "ref": f"c/{j}"},
    ]
    if n["checked"]:
        cs = n["checked_summary"]
        batches = ", ".join(str(k) for k in cs["batches"][:-1]) + f" and {cs['batches'][-1]}"
        keys.append({"what": f"(b) {n['label']}: built and checked cost points (diamonds): {cs['deployments']} of "
                             f"the {cs['of_deployments']} deployments of {cs['key']}, each at batch {batches}",
                     "value": cs["cost_points"], "key": cs["key"], "ref": f"b/{j}/built"})


    for axis, vertical, at in (("time", False, AT_T), ("energy", True, AT_E)):
        fac = sp[axis + "_factor"]
        with mb_:
            span_mark(b, *fac["of"], fac["display"], f"{axis} range", vertical=vertical, at=at,
                      hot=(j, axis) == WIDEST)
        keys.append({"what": f"(b) {n['label']}: range of {'time per inference' if axis == 'time' else 'energy'} "
                             f"across the points, greatest over least (two significant digits)",
                     "value": fac["display"], "source": n["source"], "ref": f"b/{j}"})

    mar = sp["floor_margin_factor"]


    if tuple(mar["of"]) != (floor, sp["lifetime_h"][0]):
        raise SystemExit(f"{NAME}: (c) {n['label']}: the margin {mar['of']} is not the required life {floor} to the "
                         f"least lifetime {sp['lifetime_h'][0]}, which the key {MARGIN_KEY_TEXT!r} names")
    with mc_:
        span_mark(c, floor, sp["lifetime_h"][0], "", "", vertical=True, at=AT_E, label=False)
    kwm = dict(xy=(AT_E, floor), xycoords=c.get_yaxis_transform(), textcoords="offset points", ha="left",
               va="baseline", color=INK2, path_effects=HALO, zorder=7)
    with mc_:
        margins.append(c.annotate(f"{mar['display']} {MARGIN_WORDS}", xytext=(MARGIN_DX, MARGIN_AT),
                                  fontsize=FS_MARGIN, **kwm))
    keys.append({"what": f"(c) {n['label']}: the shortest lifetime over the required service life (a factor, two significant digits)",
                 "value": mar["display"], "source": n["source"] + "; ledger:lifefloor", "ref": f"c/{j}"})

top[0].set_xlim(*XLIM); top[0].set_ylim(*YLIM_E)
bot[0].set_ylim(*YLIM_L)
for b, c in zip(top, bot):
    b.set_xlim(*XLIM)
    decade_ticks(c, "x", (-4, -2, 0, 2))
    decade_ticks(b, "y", (0, 2, 4))
    c.set_yticks([1e4, 1e8, 1e12], ["$10^{4}$", "", "$10^{12}$"]); c.yaxis.set_minor_locator(NullLocator())
    S.grid(b, "both"); S.grid(c, "both")
top[0].set_ylabel("Energy (mJ)")
bot[0].set_ylabel("Lifetime (h)")
bot[1].set_xlabel("Time per inference (s)")
fig.align_ylabels([top[0], bot[0]])
title_b = S.panel_title(top[0], "b", "Cost at a measured accuracy", pad=PAD_T, x=X_TITLE_B)
title_c = S.panel_title(bot[0], "c", "Lifetime")
LIFE_TEXT = f"Required service life, {D['floor']['years']} years"
keys.append({"what": "(c) required service life (h)", "value": D["floor"]["display"], "key": "lifefloor"})


P = D["floor_panel"]
dax = fig.add_subplot(gs[:, -1])
Ks = P["batches"]
k_lo, k_hi = Ks[0], Ks[-1]
dax.set_xscale("log"); dax.set_yscale("log")
YLIM_D = (10.0 ** math.floor(math.log10(P["lifetime_h"][0]) - YPAD_D[0]), P["lifetime_h"][1] * YPAD_D[1])
YLIM_D = S.frame_range(*YLIM_D, [{"what": "(d) lifetime", "source": P["source"]}], scale="log")
if not inside(YLIM_D, floor, *P["lifetime_h"]):
    raise SystemExit(f"FAIL {NAME}: (d): a plotted value lies outside the axis limits")

CONN_UNDER, CONN_OVER = S.C["gray3"], EST_BAND
_d_marks = S.marks(fig, "d", series="d").__enter__()
for l0 in P["connectors"]["lifetime_h_at_least_batch"]:
    l1 = l0 * k_hi / k_lo
    kx = min(max(floor / l0 * k_lo, k_lo), k_hi)
    if kx > k_lo:
        dax.plot([k_lo, kx], [l0, l0 * kx / k_lo], color=CONN_UNDER, lw=LW_CONN, solid_capstyle="butt", zorder=2)
    if kx < k_hi:
        dax.plot([kx, k_hi], [l0 * kx / k_lo, l1], color=CONN_OVER, lw=LW_CONN, solid_capstyle="butt", zorder=2)
ser = P["series"]
pts_of = lambda r: list(zip(ser[r]["batch"], ser[r]["lifetime_h"]))
n_under = sum(v < floor for r in ser for _, v in pts_of(r))
if (n_under != P["under_the_floor"] or sum(len(pts_of(r)) for r in ser) != P["points"]
        or any(len(pts_of(r)) != P["verdicts"][r] for r in ser)
        or any(v < floor for _, v in pts_of("accepted")) or any(v >= floor for _, v in pts_of("ruled_out"))):
    raise SystemExit(f"FAIL {NAME}: (d): the drawn points do not add up to the data's counts")


VERDICT_FORM = (("ruled_out", UNDER_OUT, 3.5), ("corner_dependent", UNDER_DEP, 4), ("accepted", OVER, 5))
for r, style, z in VERDICT_FORM:
    dax.plot([q[0] for q in pts_of(r)], [q[1] for q in pts_of(r)], zorder=z, **style)
_d_marks.__exit__(None, None, None)
dax.axhline(floor, color=GRAY, lw=LW_DASH, ls=(0, (3, 2)), zorder=1)

required_d = dax.annotate(REQUIRED_TEXT, xy=(0.0, floor), xycoords=dax.get_yaxis_transform(), xytext=REQUIRED_AT,
                          textcoords="offset points", ha="left", va="baseline", fontsize=FS_NOTE, color=INK2,
                          path_effects=HALO, zorder=8)

ex = P["example"]
_d_marks.__enter__()
dax.plot(ex["batch"], ex["lifetime_h"], color=INK2, lw=0.9, zorder=6, solid_capstyle="round")
dax.plot([ex["least_batch"]], [ex["lifetime_h"][0] * ex["least_batch"] / k_lo], ls="none", marker="o", ms=3.4,
         color=INK2, mew=0, zorder=7)

y_x = ex["lifetime_h"][0] * ex["least_batch"] / k_lo
dax.annotate(f"Minimum batch {ex['least_batch']}", xy=(ex["least_batch"], y_x), xytext=MINB_AT,
             textcoords="offset points", ha="right", va="bottom", fontsize=FS_MINB, color=INK,
             path_effects=HALO, zorder=8,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0.5, shrinkB=1.8,
                             relpos=(MINB_REL, 0.0)))
_d_marks.__exit__(None, None, None)
dax.set_xlim(k_lo / 1.5, k_hi * 1.9); dax.set_ylim(*YLIM_D)
dax.set_xticks(Ks, [str(k) for k in Ks]); dax.xaxis.set_minor_locator(NullLocator())

decade_ticks(dax, "y", [k for k in range(math.ceil(math.log10(YLIM_D[0])), math.floor(math.log10(YLIM_D[1])) + 1, 2)
                        if k <= math.log10(YLIM_D[1]) - 0.7])
S.grid(dax, "y")
dax.set_xlabel("Inference batch")
dax.set_ylabel("Lifetime (h)", labelpad=1.0)
title_d = fig.text(0, 0, f"(d) {P['chip_label']}", ha="left", va="baseline", fontsize=S.FS["title"],
                   fontweight="bold", color=INK)
title_d.set_in_layout(False)
kwd = dict(xycoords="axes fraction", textcoords="offset points", path_effects=HALO, zorder=8)


kwh = dict(ha="left", va="baseline")
head_n = [fig.text(0, 0, f"{S.thousands(P['under_the_floor'])} of {S.thousands(P['points'])}",
                   fontsize=S.FS["small"], color=INK, **kwh),
          fig.text(0, 0, BELOW_TEXT, fontsize=FS_L2, color=INK2, **kwh)]
for h in head_n:
    h.set_in_layout(False)
S.add_marks(fig, "d", head_n, series="d")


key_d = []


for r_, sides in (("accepted", {True}), ("ruled_out", {False}), ("corner_dependent", {True, False})):
    if {v >= floor for _, v in pts_of(r_)} != sides:
        raise SystemExit(f"FAIL {NAME}: (d): the key's line sample of {r_} does not hold for its points")
corner_pt = (matplotlib.transforms.Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans
             + matplotlib.transforms.ScaledTranslation(1, 0, dax.transAxes))
for row, (style, text, color, tones) in enumerate((
        (UNDER_DEP, ser["corner_dependent"]["label"], GRAY, (CONN_UNDER, CONN_OVER)),
        (UNDER_OUT, ser["ruled_out"]["label"], GRAY, (CONN_UNDER, CONN_UNDER)),
        (OVER, ser["accepted"]["label"], BLUE, (CONN_OVER, CONN_OVER)))):
    yk = KEY_D[0] + KEY_D[1] * row
    key_d.append(dax.annotate(text, xy=(1, 0), xytext=(-3.0, yk), ha="right", va="center_baseline", fontsize=FS_NOTE,
                              color=color, **kwd))
    wk = S.text_width_pt(fig, text, FS_NOTE)
    xm = -3.0 - wk - KEY_D_GAP - KEY_D_LINE / 2.0
    for k_, tone in enumerate(tones):
        seg = Line2D([xm + (k_ - 1) * KEY_D_LINE / 2.0, xm + k_ * KEY_D_LINE / 2.0], [yk + 0.2] * 2,
                     transform=corner_pt, color=tone, lw=LW_CONN_KEY, solid_capstyle="butt", clip_on=False, zorder=7.5)
        seg.set_in_layout(False)
        dax.add_artist(seg)
    mk = dax.plot([xm], [yk + 0.2], transform=corner_pt, clip_on=False, zorder=8, **style)
keys += [
    {"what": f"(d) {P['chip_label']}: explored points with a lifetime reading (h, nominal corner) per batch",
     "value": P["points"], "source": P["source"], "ref": "d"},
    {"what": f"(d) {P['chip_label']}: points below the required service life at the nominal corner (under the "
             f"dashed line)",
     "value": P["under_the_floor"], "source": P["source"] + "; ledger:lifefloor", "ref": "d"},
    {"what": f"(d) {P['chip_label']}: points ruled out, below the required service life at every corner (gray, "
             f"filled)", "value": P["verdicts"]["ruled_out"], "source": P["source"], "ref": "d"},
    {"what": f"(d) {P['chip_label']}: points corner dependent, below the required service life at some corner "
             f"(gray, hollow)", "value": P["verdicts"]["corner_dependent"], "source": P["source"], "ref": "d"},
    {"what": f"(d) {P['chip_label']}: points accepted, over the required service life at every corner (blue, "
             f"hollow)", "value": P["verdicts"]["accepted"], "source": P["source"], "ref": "d"},
    {"what": f"(d) {P['chip_label']}: example configuration ({ex['network']}, budget {ex['budget']}), lifetime (h) "
             f"at batch {', '.join(str(k) for k in ex['batch'])}", "value": ex["lifetime_h"], "source": ex["source"],
     "ref": "d"},
    {"what": "(d) the example's minimum batch that meets the required service life (toolchain:eq:kstar)",
     "value": ex["least_batch"],
     "source": ex["source"], "ref": "d"},
    {"what": "(d) connectors: one per configuration, lifetime in proportion to the batch (pale blue over the required "
             "service life, gray under it, at the nominal corner; sampled behind the key's marks)",
     "value": sum(P["connectors"]["configurations"]), "source": P["source"] + "; " + P["connectors"]["rule"],
     "ref": "d"},
]


fig.canvas.draw()
fig.set_layout_engine("none")
r = fig.canvas.get_renderer()
px_per_pt = fig.dpi / 72.0
inv = fig.transFigure.inverted()


down = ((a.get_xticklabels()[0].get_window_extent(r).y0 - bot[1].xaxis.label.get_window_extent(r).y0)
        / fig.bbox.height - TICK_BASE / HEIGHT)
for ax_ in (a, d):
    pos = ax_.get_position()
    ax_.set_position([pos.x0, pos.y0 - down, pos.width, pos.height + down])
fig.canvas.draw()
r = fig.canvas.get_renderer()


def baseline_px(t):
    ""
    _, _, desc = r.get_text_width_height_descent(t.get_text(), t.get_fontproperties(), False)
    return t.get_window_extent(r).y0 + desc


for ax_, kinds in MARKS.items():
    m = mark_sizes(ax_.get_window_extent(r).height / px_per_pt)
    for art in kinds.get("ring", []):
        art.set_markersize(m["ring"]); art.set_markeredgewidth(m["ring_edge"])
    for art in kinds.get("built", []):
        art.set_markersize(m["built"]); art.set_markeredgewidth(m["built_edge"])
    for art in kinds.get("band", []):
        art.set_linewidth(m["band"])
    for art in kinds.get("cap", []):
        art.set_markersize(m["cap"])
pitch_a = d.get_window_extent(r).width / px_per_pt / (d.get_xlim()[1] - d.get_xlim()[0])
for art in dots_a:
    art.set_markersize(DOT_SHARE * pitch_a)
for art in rings_a:
    art.set_markersize(DOT_SHARE * pitch_a); art.set_markeredgewidth(0.5)
base_title = baseline_px(title_b)
x0 = min(t.get_window_extent(r).x0 for t in a.get_yticklabels())
title_a.set_position(inv.transform((x0, base_title)))
title_d.set_position(inv.transform((dax.yaxis.label.get_window_extent(r).x0, base_title)))

xd_px, yd_px = dax.get_window_extent(r).x0, dax.get_window_extent(r).y1
for h, base in zip(head_n, HEAD_D):
    h.set_position(inv.transform((xd_px, yd_px + base * px_per_pt)))
for sw, tx in key_a:
    xf = inv.transform((x0 + KEY_SWATCH / 2.0 * px_per_pt, 0))[0]
    sw.set_xdata([xf]); tx.xy = (xf, tx.xy[1])
label_a.set_position(inv.transform((a.get_xticklabels()[0].get_window_extent(r).x0 - 4.0 * px_per_pt,
                                    bot[1].xaxis.label.get_window_extent(r).y0)))
label_a.set_va("bottom")


def key_row(items, x_right_px, base_px, *, size=FS_KEY, sep=KEY_ROW_SEP, gap=2.4):
    ""
    x, texts = x_right_px, []
    mid = base_px + 0.35 * size * px_per_pt
    for (w_s, draw), name in reversed(items):
        w = S.text_width_pt(fig, name, size) * px_per_pt
        t = fig.text(*inv.transform((x - w, base_px)), name, ha="left", va="baseline", fontsize=size, color=INK2)
        t.set_in_layout(False)
        texts.append(t)
        x1 = x - w - gap * px_per_pt
        draw(x1 - w_s * px_per_pt, x1, mid)
        x = x1 - (w_s + sep) * px_per_pt
    return texts, x + sep * px_per_pt


def sample_mark(**style):
    ""
    def draw(xa, xb, ym):
        fig.add_artist(Line2D([inv.transform(((xa + xb) / 2.0, ym))[0]], [inv.transform((0, ym))[1]],
                              transform=fig.transFigure, **style))
    return draw


def sample_line(**style):
    ""
    def draw(xa, xb, ym):
        fig.add_artist(Line2D([inv.transform((xa, ym))[0], inv.transform((xb, ym))[0]], [inv.transform((0, ym))[1]] * 2,
                              transform=fig.transFigure, **style))
    return draw


def sample_band(width, height, color):
    ""
    def draw(xa, xb, ym):
        xl, xr = (inv.transform((x_, ym))[0] for x_ in (xa + width / 2.0 * px_per_pt, xb - width / 2.0 * px_per_pt))
        ya, yb = (inv.transform((0, ym + s_ * height / 2.0 * px_per_pt))[1] for s_ in (-1, 1))
        fig.add_artist(Line2D([xl, xl], [ya, yb], transform=fig.transFigure, color=color, lw=width,
                              solid_capstyle="round"))
        fig.add_artist(Line2D([xr], [inv.transform((0, ym))[1]], transform=fig.transFigure, ls="none", marker="o",
                              ms=width, color=color, mew=0))
    return draw


mb = mark_sizes(top[0].get_window_extent(r).height / px_per_pt)
x_end = top[-1].get_window_extent(r).x1
ring_b = dict(HOLLOW, ms=mb["ring"], mew=mb["ring_edge"])
built_b = dict(BUILT, ms=mb["built"], mew=mb["built_edge"])
key_b, x_key_b = key_row([((1.42 * mb["built"], sample_mark(**built_b)), "Built"),
                          ((mb["ring"] + mb["ring_edge"], sample_mark(**ring_b)), "Estimated"),
                          ((2 * mb["band"] + SMALLER_DOT_GAP, sample_band(mb["band"], 4.6, EST_BAND)), SMALLER_TEXT)],
                         x_end, base_title)
span_style = dict(color=INK2, lw=LW_SPAN, marker="|", markersize=mb["cap"], markeredgewidth=LW_SPAN,
                  solid_capstyle="butt")
key_c, x_key_c = key_row([((KEY_LINE, sample_line(**span_style)), MARGIN_KEY_TEXT),
                          ((KEY_LINE, sample_line(color=GRAY, lw=LW_DASH, ls=(0, (3, 2)))), LIFE_TEXT)],
                         x_end, baseline_px(title_c))
for bar in refused_bars:
    if bar.get_window_extent(r).width > 1.0 * px_per_pt:
        bar.set_edgecolor(REFUSED_EDGE)


fig.canvas.draw()
r = fig.canvas.get_renderer()
ext = lambda arts: [x.get_window_extent(r) for x in arts]
ink_a = ext([title_a, label_a, d] + head_d + [t for _, t in key_a] + a.get_xticklabels())
ink_b = ext([title_b, title_c, top[0].yaxis.label, bot[0].yaxis.label] + top[0].get_yticklabels()
            + bot[0].get_yticklabels())
ink_b_right = ext(key_b + key_c + [top[-1], bot[-1]] + bot[-1].get_xticklabels())
ink_d = ext([title_d, dax.yaxis.label] + head_n + dax.get_yticklabels())
gut_ab = (min(e.x0 for e in ink_b if e.width > 0) - max(e.x1 for e in ink_a)) / px_per_pt
gut_bd = (min(e.x0 for e in ink_d if e.width > 0) - max(e.x1 for e in ink_b_right)) / px_per_pt
room_b = (x_key_b - title_b.get_window_extent(r).x1) / px_per_pt
room_c = (x_key_c - title_c.get_window_extent(r).x1) / px_per_pt
h_b, h_c = (ax_.get_window_extent(r).height / px_per_pt for ax_ in (top[0], bot[0]))
print(f"gutters {NAME}: (a) to (b), (c) {gut_ab:.1f} pt; (b), (c) to (d) {gut_bd:.1f} pt (least {GUTTER_MIN} pt); "
      f"plots: energy {h_b:.1f} pt, lifetime {h_c:.1f} pt high, a column "
      f"{top[0].get_window_extent(r).width / px_per_pt:.1f} pt wide, (a) row pitch "
      f"{a.get_window_extent(r).height / px_per_pt / (len(chips) + 0.1):.2f} pt, dot pitch {pitch_a:.2f} pt, (a) "
      f"{(max(e.x1 for e in ink_a) - min(e.x0 for e in ink_a)) / px_per_pt:.1f} pt wide, (d) "
      f"{dax.get_window_extent(r).width / px_per_pt:.1f} x {dax.get_window_extent(r).height / px_per_pt:.1f} pt")
ms_b, ms_c = mark_sizes(h_b), mark_sizes(h_c)
print(f"marks {NAME}: ring outer {ms_b['ring'] + ms_b['ring_edge']:.2f} pt in (b), {ms_c['ring'] + ms_c['ring_edge']:.2f} "
      f"pt in (c) ({100 * (ms_b['ring'] + ms_b['ring_edge']) / h_b:.1f}% and {100 * (ms_c['ring'] + ms_c['ring_edge']) / h_c:.1f}% "
      f"of the plot's height); diamond side {ms_b['built']:.2f} pt; band {ms_b['band']:.2f} and {ms_c['band']:.2f} pt; "
      f"title to key {room_b:.1f} pt in the row of (b), {room_c:.1f} pt in the row of (c) (least {KEY_ROOM_MIN} pt)")
if min(gut_ab, gut_bd) < GUTTER_MIN:
    print(f"FAIL {NAME}: a gutter under {GUTTER_MIN} pt")
if min(room_b, room_c) < KEY_ROOM_MIN:
    print(f"FAIL {NAME}: a key under {KEY_ROOM_MIN} pt from its row's title")


ring_b, diag_b = ms_b["ring"] + ms_b["ring_edge"], 1.4142 * (ms_b["built"] - ms_b["built_edge"])
tick_ext = sorted((e.x0, e.x1) for e in ext([t for t in a.get_xticklabels() if t.get_text()]))
tick_air = min(b0 - a1 for (_, a1), (b0, _) in zip(tick_ext, tick_ext[1:])) / px_per_pt
halo = HALO[0]._gc["linewidth"] / 2.0
floor_px = [c.transData.transform((1.0, floor))[1] for c in bot]
margin_in = min((c.get_window_extent(r).x1 - m.get_window_extent(r).x1) / px_per_pt for c, m in zip(bot, margins))
margin_floor = min((baseline_px(m) - 0.24 * FS_MARGIN * px_per_pt - f) / px_per_pt for m, f in zip(margins, floor_px))
margin_data = min((c.transData.transform((1.0, n["span"]["lifetime_h"][0]))[1] - mark_sizes(h_c)["ring"] / 2.0 * px_per_pt
                   - m.get_window_extent(r).y1) / px_per_pt for c, m, n in zip(bot, margins, nets))
axis_top = dax.spines["bottom"].get_window_extent(r).y1
key_axis = min((baseline_px(t) - 0.24 * FS_NOTE * px_per_pt - axis_top) / px_per_pt - halo for t in key_d)
print(f"pass 10 {NAME}: (b) ring outer {ring_b:.2f} pt ({100 * ring_b / h_b:.1f}%), diamond diagonal {diag_b:.2f} pt "
      f"({100 * diag_b / h_b:.1f}% of the plot's height, at most {100 * MARK_MAX:.0f}%); (a) bars "
      f"{a.get_window_extent(r).width / px_per_pt:.1f} pt wide, tick labels {tick_air:.1f} pt apart (least "
      f"{TICK_AIR_MIN} pt); (c) margin label {margin_in:.1f} pt inside its plot's end, its descenders "
      f"{margin_floor:.1f} pt over the dashed line, its line box {margin_data:.1f} pt under the lowest mark; (d) key "
      f"halo {key_axis:.1f} pt over the x axis line (least {KEY_AXIS_MIN} pt)")
if max(ring_b, diag_b) > MARK_MAX * h_b:
    print(f"FAIL {NAME}: a mark of (b) over {100 * MARK_MAX:.0f}% of the plot's height")
if tick_air < TICK_AIR_MIN:
    print(f"FAIL {NAME}: (a): two tick labels of the bars under {TICK_AIR_MIN} pt apart")
if margin_in < 0 or margin_floor < halo or margin_data < 0:
    print(f"FAIL {NAME}: (c): a margin label leaves its plot or meets the dashed line or a mark")
if key_axis < KEY_AXIS_MIN:
    print(f"FAIL {NAME}: (d): a key name's halo under {KEY_AXIS_MIN} pt from the x axis line")

S.pending_key_at(fig, 1.0, 0.0, ha="right", ax=dax, anchor=title_d, words="Release version")
S.save_fig(fig, NAME, rows=keys)
S.save_keys(NAME, keys)
