""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.MIN_FONT_PT = 5.5
S.INK_CLEAR_PT = 1.2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

NAME = "fig_area"

FS = {"focal": 8.0, "label": 6.8, "tick": 6.5, "small": 6.0}
FS_DIE = 5.5
FS_A = 5.5
LINE_EM, DESC_EM, AIR_EM = 1.362, 0.293, 0.31


TITLE_H, PLOT_H, FOOT = 11.0, 74.0, 21.5
HEIGHT_PT = TITLE_H + PLOT_H + FOOT
EDGE = 0.6
YLAB_W, YLAB_GAP, TICK_PAD, TICK_LEN = 8.6, 1.0, 1.0, 2.2
A_TICK_W, C_TICK_W = 14.5, 22.5
A_PLOT_W = 168.0
AB_GAP, B_NAME_W, B_BAR_W = 9.0, 44.0, 77.0
BC_GAP, C_MARGIN = 11.0, 53.0
TITLE_PAD, XLABEL_UP = 1.0, 0.4


BAND_H = LINE_EM * 6.0 + 1.5


TYPE_ORDER = ("T1", "T2", "T3", "T4")
PAD_DEC = 0.36

BAND_FACTOR = 2.0
MARK_PT = {"o": 3.5, "s": 3.1, "^": 4.0, "D": 3.1}
MARK_AIR = 2.0
NOLEAD_AIR = 3.2

QUIET_PT = 2.4


FOCAL_X, FOCAL_PITCH, FOCAL_LINE, KEY_PITCH, BAND_AIR = 2.0, 9.4, 9.4, 6.8, 2.0
KEY_LINE = 6.8


STACK_AIR = 1.3

BAR_H, PAIR_GAP = 2.9, 0.7
GROUPS = (("array", "Array"), ("converters", "Converters"), ("periphery_logic_routing", "Logic and periphery"))

EXEC_PT, PLAN_PT = 3.0, 1.9
LIN_DEC = 0.38
EQUAL = ("equal", "checked")


STEP_LW, STEP_DASH, STEP_CASE = 1.0, (3.0, 1.7), 0.5
KEY_FORMS = (("checked", "Equal outputs"), ("pending", "Pending"), ("planned", "Planned only"))
TYPE_KEY_COLS, TYPE_KEY_SEP = 1, 5.0
C_YPAD = (1.25, 1.35)
BRACKET_DX, BRACKET_TICK = 6.0, 3.2


OWN_TEXT, OTHER_TEXT, MODEL_TEXT = "Own cell or converter area", "Area from other papers", "Unfabricated design"


A_KEY_COL_W = 62.0875
BAND_TEXT = f"Within {BAND_FACTOR:g}\u00d7"


PRICED, REFUSED = "priced", "named refusal"
REFUSAL_TEXT = "Rejected"
DERIVED_TEXT = "Published area, computed"


REFUSAL_LW, REFUSAL_CAP = 0.9, 3.6
REFUSAL_STROKE = 3.6
REFUSAL_BAR_W = 1.7
DRAWN_ROLES = ("upper bound", "range low", "range high")
REFUSAL_GRID = (15, 0.25)


DERIVED_FORM = "halo"
DERIVED_GAP, DERIVED_LW = 0.9, 0.45
FOOT_KEY_GAP = 6.0
DERIVED_REACH = {}

REFUSED_EDGE, REFUSED_HATCH_LW, REFUSED_HATCH, REFUSED_LABEL_GAP = 0.5, 0.35, 1.5, 2.2


LABEL_SEED = int(os.environ.get("FIG_AREA_LABEL_SEED", "7"))
LABEL_TRIES = int(os.environ.get("FIG_AREA_LABEL_TRIES", "2500"))


MISS, MISS_NAMED = 100000.0, 90000.0
ALONE_PT = 16.0
DEEP_WIDTH, DEEP_NODES, DEEP_APART = 5, 20000, 3.0

def type_of(t: str) -> str:
    return t[:2]


def status_of(r) -> str:
    s = r.get("status")
    if s == "reference":
        return "reference"
    if s in EQUAL:
        return "checked"
    if s == "planned":
        return "planned"
    return "pending"


def factor_text(v: float) -> str:
    ""
    return f"{v:.1f}" if v < 9.95 else S.thousands(round(v))


def to_pt(ax, x, y):
    ""
    px, py = ax.transData.transform((x, y))
    return (px - ax.bbox.x0) * 72.0 / ax.figure.dpi, (py - ax.bbox.y0) * 72.0 / ax.figure.dpi


def overlap(a, b, slack=0.0):
    return min(a[2], b[2]) - max(a[0], b[0]) > slack and min(a[3], b[3]) - max(a[1], b[1]) > slack


def crosses(box, p, q):
    ""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for d, lo, hi in ((dx, x0 - p[0], x1 - p[0]), (dy, y0 - p[1], y1 - p[1])):
        if abs(d) < 1e-12:
            if lo > 0 or hi < 0:
                return False
            continue
        a, b = lo / d, hi / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def seg_cross(p, q, r, s):
    ""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0


class Placer:
    ""

    def __init__(self, ax, width, height, room=(0.8, 0.8, 0.8, 0.8)):
        self.ax, self.w, self.h, self.room = ax, width, height, room
        self.texts, self.lines, self.marks, self.leaders = [], [], [], []
        self.owner = {}
        self.air = 0.0
        self.air_bare = 0.0
        self.slide = False
        self.blame = False
        self.stack_air = None
        self.ink_y, self.ink_ext = {}, {}

    def block(self, box, ink_y=None):
        ""
        self.texts.append(box)
        if ink_y is not None:
            self.ink_y[tuple(box)] = ink_y

    def meets_text(self, box, ink_y=None):
        ""
        for k in self.texts:
            if self.stack_air is None or ink_y is None or tuple(k) not in self.ink_y:
                if overlap((box[0] - 2.2, box[1], box[2] + 2.2, box[3]), k, 0.3):
                    return True
            elif min(box[2] + 2.2, k[2]) - max(box[0] - 2.2, k[0]) > 0.3:
                k0, k1 = self.ink_y[tuple(k)]
                if max(ink_y[0], k0) - min(ink_y[1], k1) < self.stack_air:
                    return True
        return False

    def add_line(self, pts):
        self.lines += list(zip(pts, pts[1:]))

    def add_mark(self, x, y, radius, owner=None):
        ""
        self.marks.append((x, y, radius))
        if owner is not None:
            self.owner[(x, y, radius)] = owner

    def own_part(self, mark, own):
        if own is None:
            return False
        o = self.owner.get(mark, mark[:2])
        return abs(o[0] - own[0]) < 1e-6 and abs(o[1] - own[1]) < 1e-6

    def free(self, box, ink, own=None, leader=None, ink_y=None, off=()):
        l, b, r, t = self.room
        if box[0] < l or box[2] > self.w - r or ink[1] < b or ink[3] > self.h - t:
            return False
        if any(crosses(ink, p, q) for p, q in off):
            return False
        if self.meets_text(box, ink_y):
            return False
        wide = (ink[0] - 0.6, ink[1] - 0.6, ink[2] + 0.6, ink[3] + 0.5)
        if any(crosses(ink, p, q) for p, q in self.lines) or any(crosses(wide, p, q) for p, q in self.leaders):
            return False
        for mx, my, mr in self.marks:
            if self.own_part((mx, my, mr), own):
                continue
            air = self.air if (leader is not None or own is None) else max(self.air, self.air_bare)
            if overlap(ink, (mx - mr - air, my - mr - air, mx + mr + air, my + mr + air)):
                return False
        if leader is not None:
            p, q = leader
            if any(crosses((k[0] - 0.6, k[1] + 0.6, k[2] + 0.6, k[3] - 0.9), p, q) for k in self.texts):
                return False
            if any(seg_cross(p, q, a, c) for a, c in self.leaders):
                return False
            for mx, my, mr in self.marks:
                if self.own_part((mx, my, mr), own):
                    continue
                if crosses((mx - mr, my - mr, mx + mr, my + mr), p, q):
                    return False
        return True

    def text(self, box, text, fs, **kw):
        px = self.ax.figure.dpi / 72.0
        inv = self.ax.transAxes.inverted()
        fx, fy = inv.transform((self.ax.bbox.x0 + box[0] * px, self.ax.bbox.y0 + box[1] * px))
        return self.ax.text(fx, fy, text, transform=self.ax.transAxes, ha="left", va="bottom", fontsize=fs,
                            clip_on=False, zorder=7, **kw)

    def find(self, x, y, w, fs, radius, angles, reaches, off=(), rng=None, skip=0.0, take=None):
        ""
        hits = []
        h, d, air = LINE_EM * fs, DESC_EM * fs, AIR_EM * fs
        ink_h = h - d - air
        first = None


        shifts = (0.0, -0.25 * w, 0.25 * w, 1.5 - 0.5 * w, 0.5 * w - 1.5) if self.slide else (0.0,)
        for reach in reaches:
            for ang, shift in ((a, sh) for a in angles
                               for sh in (shifts if abs(math.cos(math.radians(a))) <= 0.3 else (0.0,))):
                c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                ax_, ay_ = x + (radius + reach) * c, y + (radius + reach) * s
                ix0 = ax_ - w if c < -0.3 else (ax_ if c > 0.3 else ax_ - w / 2.0 + shift)
                if abs(c) <= 0.3:
                    lim = (self.room[0] + 0.6, self.w - self.room[2] - w - 0.6)
                    if lim[0] <= lim[1] and ix0 + 1.0 <= x <= ix0 + w - 1.0:
                        ix0 = min(max(ix0, lim[0], x - w + 1.0), lim[1], x - 1.0) if lim[0] <= x - 1.0 else ix0
                iy0 = ay_ - ink_h if s < -0.5 else (ay_ if s > 0.5 else ay_ - ink_h / 2.0)
                ink = (ix0 - 0.5, iy0 - 0.5, ix0 + w + 0.5, iy0 + ink_h + 0.5)
                box = (ix0, iy0 - d, ix0 + w, iy0 - d + h)
                lead = None
                if reach > 3.5:
                    ex = min(max(x, ix0), ix0 + w)
                    ey = min(max(y, iy0), iy0 + ink_h)
                    gap = math.hypot(ex - x, ey - y)
                    if gap > 1e-6:
                        ux, uy = (ex - x) / gap, (ey - y) / gap
                        lead = ((x + ux * (radius + 0.6), y + uy * (radius + 0.6)), (ex - ux * 0.9, ey - uy * 0.9))
                ext = self.ink_ext.get((x, y, w))
                ink_y = (iy0 + ext[0], iy0 + ext[1]) if ext else None
                if self.free(box, ink, own=(x, y), leader=lead, ink_y=ink_y, off=off):
                    if ink_y is not None:
                        self.ink_y[tuple(box)] = ink_y
                    if take:
                        if all(abs(box[0] - h[0][0]) >= DEEP_APART or abs(box[1] - h[0][1]) >= DEEP_APART
                               for h in hits):
                            hits.append((box, lead, reach))
                            if len(hits) >= take:
                                return hits
                        continue
                    if first is None:
                        first = (box, lead, reach)
                    if rng is None or rng.random() >= skip:
                        return box, lead, reach
        return hits if take else first

    def solve(self, items, tries=2500, seed=7, miss=None):
        ""
        import random
        rng = random.Random(seed)
        base_t, base_l = list(self.texts), list(self.leaders)
        best, best_score = None, None
        order = list(range(len(items)))


        blamed = [0.0] * len(items)
        for k in range(tries):
            if k:
                rng.shuffle(order)
                if self.blame and k % 4:
                    order.sort(key=lambda i: -blamed[i] + 2.0 * rng.random())
            self.texts, self.leaders = list(base_t), list(base_l)
            got, cost = [None] * len(items), 0.0
            for i in order:
                hit = self.find(*items[i], rng=rng if k else None, skip=0.3 if k % 2 else 0.0)
                if hit is None:
                    cost += miss[i] if miss else 1000.0
                    blamed[i] += 1.0
                    continue
                box, lead, reach = hit
                self.texts.append(box)
                if lead is not None:
                    self.leaders.append(lead)
                got[i] = (box, lead)
                cost += reach * reach
            if best_score is None or cost < best_score:
                best, best_score = got, cost

        def book(skip=()):
            self.texts = base_t + [g[0] for k, g in enumerate(best) if g and k not in skip]
            self.leaders = base_l + [g[1] for k, g in enumerate(best) if g and g[1] and k not in skip]


        if self.blame and any(g is None for g in best):
            by_blame = sorted(range(len(items)), key=lambda i: -blamed[i])
            chosen, tried = {}, [0]

            def go(d):
                if d == len(by_blame):
                    return True
                self.texts = base_t + [chosen[j][0] for j in by_blame[:d]]
                self.leaders = base_l + [chosen[j][1] for j in by_blame[:d] if chosen[j][1]]
                for hit in self.find(*items[by_blame[d]], take=DEEP_WIDTH):
                    tried[0] += 1
                    if tried[0] > DEEP_NODES:
                        return False
                    chosen[by_blame[d]] = hit[:2]
                    if go(d + 1):
                        return True
                    self.texts = base_t + [chosen[j][0] for j in by_blame[:d]]
                    self.leaders = base_l + [chosen[j][1] for j in by_blame[:d] if chosen[j][1]]
                return False
            if go(0):
                best = [chosen[i] for i in range(len(items))]
            print(f"pass 13 {NAME}: the deep step of the label search tried {tried[0]} places, "
                  f"{sum(g is None for g in best)} label(s) without a place")
        book()
        return best

    def leader(self, lead, color=None):
        px = self.ax.figure.dpi / 72.0
        inv = self.ax.transAxes.inverted()
        pts = [inv.transform((self.ax.bbox.x0 + a * px, self.ax.bbox.y0 + b * px)) for a, b in lead]
        self.ax.add_line(Line2D([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], transform=self.ax.transAxes,
                                color=color or S.C["gray"], linewidth=0.45, zorder=3, clip_on=False))


def key_entry(ax, pw, x, y, text, fs, draw_mark, *, color=None, mark_w=7.0):
    ""
    w = S.text_width_pt(ax.figure, text, fs)
    ym = y + (DESC_EM + 0.36) * fs
    draw_mark(x + mark_w / 2.0 - 0.5, ym)
    ax.text((x + mark_w) / pw, y / PLOT_H, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=fs,
            color=color or S.C["ink2"], zorder=7)
    return (x - 1.0, y, x + mark_w + w, y + LINE_EM * fs)


def panel_a(fig, ax, d, pw, keys):
    ""
    pts, st = d["parity"], d["stats"]
    tone = dict(zip(TYPE_ORDER, S.TONES4))
    form = dict(zip(TYPE_ORDER, S.TONES4_MARKERS))
    gray, quiet = S.C["gray"], S.C["gray2"]
    vals = [v for p in pts for v in (p["published_mm2"], p["estimate_mm2"])]
    vals += [p["published_mm2"] * r for p in pts if p["strict_ratio"] for r in p["strict_ratio"]]
    refusals = d["named_refusals"]
    vals += [v for e in refusals for v in [e["published_mm2"]] + [r["estimate_mm2"] for r in e["readings"]
                                                                  if r["role"] in DRAWN_ROLES]]
    lo, hi = min(vals) / 10 ** PAD_DEC, max(vals) * 10 ** PAD_DEC
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    decades = [10.0 ** k for k in range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(decades))
        axis.set_major_formatter(FuncFormatter(lambda v, pos=None: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=FS["tick"], pad=TICK_PAD, length=TICK_LEN)


    ax.add_patch(Polygon([(lo, lo / BAND_FACTOR), (hi, hi / BAND_FACTOR), (hi, hi * BAND_FACTOR),
                          (lo, lo * BAND_FACTOR)], closed=True, facecolor=S.C["gray3"], edgecolor="none", zorder=0))
    ax.plot([lo, hi], [lo, hi], color=gray, linewidth=S.LW["grid"] + 0.2, zorder=1)
    place = Placer(ax, pw, PLOT_H)
    place.air, place.air_bare = MARK_AIR, NOLEAD_AIR
    place.stack_air = STACK_AIR
    place.slide = place.blame = True
    _pa = S.marks(fig, "a").__enter__()

    def ink_of(word, size, weight="normal"):
        ""
        e = TextPath((0, 0), word, size=size, prop=FontProperties(weight=weight)).get_extents()
        return e.y0, e.y1
    place.add_line([to_pt(ax, lo, lo), to_pt(ax, hi, hi)])


    def draw(ax_, x, y, p, *, transform=None, size=1.0, strict=True):
        kw = dict(transform=transform) if transform is not None else {}
        t = type_of(p["type"])
        if not p["silicon"]:
            ax_.plot([x], [y], marker="o", markersize=QUIET_PT * size, color=gray, markeredgewidth=0, zorder=4,
                     linestyle="none", **kw)
            return QUIET_PT / 2.0
        if p["scope_mismatch"]:
            ax_.plot([x], [y], marker="x", markersize=QUIET_PT * 1.25 * size, color=gray, markeredgewidth=0.8,
                     zorder=4, linestyle="none", **kw)
            return QUIET_PT * 0.7
        ms = MARK_PT[form[t]] * size
        if p["own_unit"]:
            ax_.plot([x], [y], marker=form[t], markersize=ms - 0.6, markerfacecolor="none",
                     markeredgecolor=tone[t], markeredgewidth=1.0, zorder=5, linestyle="none", **kw)
        else:
            ax_.plot([x], [y], marker=form[t], markersize=ms, color=tone[t], markeredgecolor="white",
                     markeredgewidth=0.4, zorder=5, linestyle="none", **kw)
        return ms / 2.0 + 0.3

    def derived(x_pt, y_pt, rad, z=4.6):
        ""
        dark = S.C["ink2"]
        if DERIVED_FORM == "halo":
            ax.plot([x_pt / pw], [y_pt / PLOT_H], marker="o", markersize=2.0 * (rad + DERIVED_GAP),
                    markerfacecolor="none", markeredgecolor=dark, markeredgewidth=DERIVED_LW, transform=ax.transAxes,
                    linestyle="none", zorder=z, clip_on=False)
        elif DERIVED_FORM == "prime":
            x0, y0 = x_pt + rad * 0.72 + 0.5, y_pt + rad * 0.72 + 0.2
            ax.add_line(Line2D([x0 / pw, (x0 + 0.9) / pw], [y0 / PLOT_H, (y0 + 2.2) / PLOT_H],
                               transform=ax.transAxes, color=dark, linewidth=0.8, zorder=6.5, solid_capstyle="butt"))
        elif DERIVED_FORM == "under":
            ax.add_line(Line2D([(x_pt - rad - 0.4) / pw, (x_pt + rad + 0.4) / pw],
                               [(y_pt - rad - 1.1) / PLOT_H] * 2, transform=ax.transAxes, color=dark, linewidth=0.7,
                               zorder=6.5, solid_capstyle="butt"))

    def whisker(x, y0, y1, color):
        ax.plot([x, x], [y0, y1], color=color, linewidth=0.75, zorder=6, solid_capstyle="butt")
        for y in (y0, y1):
            ax.plot([x], [y], marker="_", markersize=3.0, color=color, markeredgewidth=0.75, zorder=6)

    def tick(x, y, width=REFUSAL_CAP):
        ""
        ax.plot([x], [y], marker="_", markersize=width, color=gray, markeredgewidth=REFUSAL_LW, zorder=6,
                linestyle="none")

    def stroke(x, y, dy0, dy1, lw=REFUSAL_LW):
        ""
        px, py = to_pt(ax, x, y)
        ax.add_line(Line2D([px / pw, px / pw], [(py + dy0) / PLOT_H, (py + dy1) / PLOT_H], transform=ax.transAxes,
                           color=gray, linewidth=lw, zorder=6, solid_capstyle="butt"))

    marks = []
    for p in pts:
        x, y = p["published_mm2"], p["estimate_mm2"]
        r = draw(ax, x, y, p)
        if p["own_unit"] and p["strict_ratio"]:
            if p["strict_status"] == REFUSED:
                for rr in sorted(set(p["strict_ratio"])):
                    tick(x, x * rr)
                keys.append({"what": f"(a) {p['label']}: the reading with the type's default unit areas, a named "
                                     "refusal (a gray tick, in no statistic), over published",
                             "value": " to ".join(f"{rr:.3g}" for rr in sorted(set(p["strict_ratio"]))),
                             "source": p["strict_source"]})
            else:
                whisker(x, x * p["strict_ratio"][0], x * p["strict_ratio"][1], tone[type_of(p["type"])])
                keys.append({"what": f"(a) {p['label']}: estimate over published with the area from other "
                                     "publications (the whisker)",
                             "value": f"{p['strict_ratio'][0]:.3g} to {p['strict_ratio'][1]:.3g}",
                             "source": p["strict_source"]})
        px, py = to_pt(ax, x, y)
        if p["mark_kind"] == "derived":
            derived(px, py, r)
            r += DERIVED_REACH.get(DERIVED_FORM, 0.0)
            keys.append({"what": f"(a) {p['label']}: the derived mark (mark_kind 'derived': its published area is "
                                 "computed from printed figures)", "value": p["mark_kind"],
                         "source": p["published_source"]})
        marks.append((p, px, py, r))
        place.add_mark(px, py, r)
        if p["own_unit"] and p["strict_ratio"]:
            for rr in p["strict_ratio"]:
                place.add_mark(px, to_pt(ax, x, x * rr)[1], 1.6, owner=(px, py))

            place.add_line([(px, to_pt(ax, x, x * rr)[1]) for rr in p["strict_ratio"]])
        keys.append({"what": f"(a) {p['label']}: published area (mm2), {p['scope']}", "value": f"{x:g}",
                     "source": p["published_source"]})
        keys.append({"what": f"(a) {p['label']}: held-out estimate (mm2)"
                             + (", own cell or converter area kept" if p["own_unit"] else ""),
                     "value": f"{y:.4g}", "source": p["estimate_source"]})


    refused_marks = []
    for e in refusals:
        x = e["published_mm2"]
        by_role = {}
        for rd in e["readings"]:
            if rd["role"] in DRAWN_ROLES:
                by_role[rd["role"]] = rd
        spans = []
        if "range low" in by_role and "range high" in by_role:
            y0, y1 = by_role["range low"]["estimate_mm2"], by_role["range high"]["estimate_mm2"]
            px, p0 = to_pt(ax, x, y0)
            p1 = to_pt(ax, x, y1)[1]
            ax.add_patch(Rectangle(((px - REFUSAL_BAR_W / 2.0) / pw, p0 / PLOT_H), REFUSAL_BAR_W / pw,
                                   (p1 - p0) / PLOT_H, transform=ax.transAxes, facecolor=gray, edgecolor="none",
                                   zorder=6))
            spans += [y0, y1]
            keys.append({"what": f"(a) {e['label']}: a named refusal ({e['reason']}), a gray bar between its "
                                 "readings (mm2), in no statistic; over published: "
                                 f"{by_role['range low']['ratio_display']['text']} to "
                                 f"{by_role['range high']['ratio_display']['text']}",
                         "value": f"{by_role['range low']['mm2_display']['text']} to "
                                  f"{by_role['range high']['mm2_display']['text']}", "source": e["source"]})
        if "upper bound" in by_role:
            y1 = by_role["upper bound"]["estimate_mm2"]
            tick(x, y1)
            stroke(x, y1, -REFUSAL_STROKE, 0.0)
            spans.append(y1)
            keys.append({"what": f"(a) {e['label']}: a named refusal ({e['reason']}), a gray cap with a downward "
                                 "stroke at its upper-bound reading (mm2), in no statistic; over published: "
                                 f"{by_role['upper bound']['ratio_display']['text']}",
                         "value": by_role["upper bound"]["mm2_display"]["text"], "source": e["source"]})
        left = [rd["role"] for rd in e["readings"] if rd["role"] not in DRAWN_ROLES]
        if left:
            keys.append({"what": f"(a) {e['label']}: readings not drawn (the text carries them)",
                         "value": ", ".join(left), "source": e["source"]})
        if not spans:
            print(f"NOTE {NAME}: the named refusal {e['label']} has no drawn reading")
            continue
        px = to_pt(ax, x, spans[0])[0]
        ys = [to_pt(ax, x, v)[1] for v in spans]
        lo_pt, hi_pt = min(ys) - (REFUSAL_STROKE if "upper bound" in by_role and len(spans) == 1 else 0.0), max(ys)
        py, rad = (lo_pt + hi_pt) / 2.0, max((hi_pt - lo_pt) / 2.0, REFUSAL_CAP / 2.0)
        place.add_mark(px, py, rad)
        refused_marks.append((e, px, py, rad))
        keys.append({"what": f"(a) {e['label']}: published area (mm2), {e['scope']}", "value": f"{x:g}",
                     "source": e["source"]})


    fs = FS["focal"]
    focal = d["displays"]["focal_label"]
    text = focal["text"]
    head, _, tail = text.partition(", ")
    runs = [(head + (", " if tail else ""), dict(fontsize=FS["label"], color=S.C["ink"]))]
    if tail:
        runs.append((tail, dict(fontsize=fs, fontweight="bold", color=S.C["blue"])))
    fsk_b = FS["small"]
    S.descriptor(ax, [runs], x=0.0, gap=BAND_H - LINE_EM * fsk_b + 0.6 + DESC_EM * fsk_b - DESC_EM * fs - 0.2)
    base = PLOT_H + TITLE_PAD + DESC_EM * S.FS["title"] - FOCAL_PITCH
    inked = []
    lines = [text]
    base_last = base + KEY_PITCH
    keys.append({"what": "(a) focal label: fabricated chips of the statistics and their worst factor",
                 "value": text, "source": focal["source"]})

    _pa.__exit__(None, None, None)


    fsk = FS_A
    own_tone = S.C["ink2"]

    def ring_key(xm, ym):
        ax.plot([xm / pw], [ym / PLOT_H], marker="o", markersize=MARK_PT["o"] - 0.6, markerfacecolor="none",
                markeredgecolor=own_tone, markeredgewidth=1.0, transform=ax.transAxes, linestyle="none", zorder=7)


    priced_strict = [p for p in pts if p["own_unit"] and p["strict_ratio"] and p["strict_status"] != REFUSED]
    ranged = any(min(p["strict_ratio"]) != max(p["strict_ratio"]) for p in priced_strict)

    def whisker_key(xm, ym):
        ends = (ym - 2.4, ym + 2.4) if ranged else (ym,)
        if ranged:
            ax.plot([xm / pw, xm / pw], [ends[0] / PLOT_H, ends[1] / PLOT_H], color=own_tone, linewidth=0.75,
                    transform=ax.transAxes, zorder=7, solid_capstyle="butt")
        for yy in ends:
            ax.plot([xm / pw], [yy / PLOT_H], marker="_", markersize=3.0, color=own_tone, markeredgewidth=0.75,
                    transform=ax.transAxes, zorder=7)
    if priced_strict:
        keys.append({"what": "(a) key: the sample of the priced readings' entry, in the form of their marks (a "
                             "whisker with two caps where a priced strict_ratio is a range, one cap where none is)",
                     "value": "whisker" if ranged else "single cap",
                     "source": "; ".join(p["strict_source"] for p in priced_strict)})

    def derived_key(xm, ym):
        ax.plot([xm / pw], [ym / PLOT_H], marker="o", markersize=MARK_PT["o"] - 0.9, color=S.C["gray2"],
                markeredgewidth=0, transform=ax.transAxes, linestyle="none", zorder=7)
        derived(xm, ym, (MARK_PT["o"] - 0.9) / 2.0, z=7.1)

    has_derived = any(p["mark_kind"] == "derived" for p in pts)
    refusal_sources = [e["source"] for e in refusals] + [p["strict_source"] for p in pts
                                                         if p["strict_status"] == REFUSED]
    entries = ((OTHER_TEXT, whisker_key), (OWN_TEXT, ring_key)) + (((DERIVED_TEXT, derived_key),) if has_derived
                                                                   else ())


    y = base_last - KEY_PITCH - DESC_EM * fsk
    for word, mark in entries:
        e0, e1 = ink_of(word, fsk)
        place.block(key_entry(ax, pw, 3.0, y, word, fsk, mark), ink_y=(y + DESC_EM * fsk + e0, y + DESC_EM * fsk + e1))
        inked.append((3.0 + 7.0, y + DESC_EM * fsk, word, fsk, "normal"))
        y -= KEY_LINE

    ppx, ppy = pw / math.log10(hi / lo), PLOT_H / math.log10(hi / lo)
    clear = min(((yb + vy) - ((xb + vx) / ppx + math.log10(BAND_FACTOR)) * ppy) / math.hypot(1.0, ppy / ppx)
                for xb, yb, word, size, weight in inked
                for vx, vy in TextPath((0, 0), word, size=size, prop=FontProperties(weight=weight)).vertices)
    print(f"pass 23 {NAME}: (a) focal label {fs:g} pt bold over the plot (C-3), the forms' key from "
          f"{base:.1f} pt over the foot; the key's ink to the band's edge {clear:.1f} pt (least {BAND_AIR:g} pt)")
    if clear < BAND_AIR:
        print(f"FAIL {NAME}: (a): the ink of the focal label or of a key under it is {clear:.1f} pt from the band")


    types = []
    for p in pts:
        t = type_of(p["type"])
        if p["headline"] and (t, p["type_label"]) not in types:
            types.append((t, p["type_label"]))
    types.sort(key=lambda tl: TYPE_ORDER.index(tl[0]))


    rows_k = [types[k:k + TYPE_KEY_COLS] for k in range(0, len(types), TYPE_KEY_COLS)]
    col_w = [max(S.text_width_pt(fig, row[c][1], fsk) for row in rows_k if c < len(row)) + 7.0
             for c in range(TYPE_KEY_COLS)]


    col_w[-1] = max(col_w[-1], S.text_width_pt(fig, MODEL_TEXT, fsk) + 7.0, A_KEY_COL_W)


    foot_rows = 0
    x_key = pw - 1.5 - sum(col_w) - TYPE_KEY_SEP * (len(col_w) - 1)
    if has_derived:
        keys.append({"what": "(a) key: the derived mark's entry (points with mark_kind 'derived')",
                     "value": DERIVED_TEXT, "source": "; ".join(p["published_source"] for p in pts
                                                              if p["mark_kind"] == "derived")})
    for j, row in enumerate(rows_k):
        yk = 0.6 + (len(rows_k) - 1 - j + foot_rows) * (LINE_EM * fsk - 0.4)
        for c, (t, lab) in enumerate(row):
            xk = x_key + sum(col_w[:c]) + TYPE_KEY_SEP * c
            place.block(key_entry(ax, pw, xk, yk, lab, fsk, lambda xm, ym, t=t: ax.plot(
                [xm / pw], [ym / PLOT_H], marker=form[t], markersize=MARK_PT[form[t]] - 0.3, color=tone[t],
                markeredgewidth=0, transform=ax.transAxes, linestyle="none", zorder=7), color=S.C["ink"]))


    def quiet_label(m):
        p, px, py, r = m
        place.ink_ext[(px, py, S.text_width_pt(fig, p["label"], FS_A))] = ink_of(p["label"], FS_A)
        hit = (place.find(px, py, S.text_width_pt(fig, p["label"], FS_A), FS_A, r,
                          (0, -45, -90, 45, -135, 90, 180, 135), (1.0, 3.0, 5.0))
               or place.find(px, py, S.text_width_pt(fig, p["label"], FS_A), FS_A, r,
                             (0, -45, -90, 45, -135, 90, 180, 135), (7.0, 9.5, 12.0)))
        if hit:
            if hit[1]:
                place.leader(hit[1])
            place.block(hit[0])
            place.text(hit[0], p["label"], FS_A, color=gray)
        else:
            print(f"NOTE {NAME}: no free place for the quiet label of {p['label']}")


    head_marks = [m for m in marks if m[0]["headline"]]
    band_edges = ([to_pt(ax, lo, lo * BAND_FACTOR), to_pt(ax, hi / BAND_FACTOR, hi)],
                  [to_pt(ax, lo * BAND_FACTOR, lo), to_pt(ax, hi, hi / BAND_FACTOR)])
    quiet = [m for m in marks if not m[0]["headline"]]
    alone = [m for m in quiet if all(math.hypot(o[1] - m[1], o[2] - m[2]) >= ALONE_PT for o in head_marks)]
    n_lines = len(place.lines)
    for e in band_edges:
        place.add_line(e)
    for m in alone:
        quiet_label(m)
    del place.lines[n_lines:]


    ang = math.degrees(math.atan2(PLOT_H, pw))
    xs_h = sorted(px for p, px, py, r in marks if p["headline"])
    off = 2.9
    bw = S.text_width_pt(fig, BAND_TEXT, fsk) / 2.0
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def band_spot(xm):
        cx, cy = xm - off * sa, xm * PLOT_H / pw + off * ca
        return cx, cy, (cx - bw * ca - 2, cy - bw * sa - 4, cx + bw * ca + 2, cy + bw * sa + 4)


    spots = [a_ + (b_ - a_) * f for _, a_, b_ in sorted(((b_ - a_, a_, b_) for a_, b_ in zip(xs_h, xs_h[1:])),
                                                        reverse=True) for f in (0.4, 0.5, 0.3, 0.6, 0.2, 0.7, 0.8)]
    def band_cost(xm):
        box = band_spot(xm)[2]
        blocks = place.texts + [(mx - mr, my - mr, mx + mr, my + mr) for mx, my, mr in place.marks]
        return max([min(min(box[2], k[2]) - max(box[0], k[0]), min(box[3], k[3]) - max(box[1], k[1]))
                    for k in blocks if overlap(box, k)] or [0.0])
    cx, cy, bbox_ = band_spot(min(spots, key=lambda xm: (round(band_cost(xm), 1), spots.index(xm))))
    band_name = ax.text(cx / pw, cy / PLOT_H, BAND_TEXT, transform=ax.transAxes, ha="center", va="center",
                        rotation=ang, rotation_mode="anchor", fontsize=fsk, color=S.C["ink2"], zorder=2)

    bb = band_name.get_window_extent(fig.canvas.get_renderer())
    k_pt = 72.0 / fig.dpi
    place.block(((bb.x0 - ax.bbox.x0) * k_pt - 0.6, (bb.y0 - ax.bbox.y0) * k_pt - 0.6,
                 (bb.x1 - ax.bbox.x0) * k_pt + 0.6, (bb.y1 - ax.bbox.y0) * k_pt + 0.6))
    for k in (-1, 0, 1):
        sx, sy = cx + k * (2.0 * bw / 3.0) * ca, cy + k * (2.0 * bw / 3.0) * sa
        place.block((sx - bw / 3.0 * ca - 2.0, sy - bw / 3.0 * sa - 3.5, sx + bw / 3.0 * ca + 2.0,
                     sy + bw / 3.0 * sa + 3.5))


    def crowd(m):
        return -sum(1 for o in marks if math.hypot(o[1] - m[1], o[2] - m[2]) < 16.0)
    reaches = (1.0, 3.0, 5.0, 7.0, 9.5, 12.0, 15.0, 18.0, 21.0, 25.0, 29.0, 34.0, 40.0, 46.0, 52.0, 60.0, 68.0)
    head = sorted([m for m in marks if m[0]["headline"]], key=crowd)
    items = []
    for p, px, py, r in head:
        up = p["ratio"] >= 1.0
        angles = ((135, 160, 110, 180, 90, 200, 65, 225, 40, 250, 20, 0, 270, 300, 330, 235, 150, 120) if up
                  else (-45, -20, -70, 0, -90, 20, -115, 45, -140, 70, 180, 200, 90, 135, -30, -60, 160, 110))
        items.append((px, py, S.text_width_pt(fig, p["label"], fsk), fsk, r, angles, reaches))
        place.ink_ext[items[-1][:3]] = ink_of(p["label"], fsk)
    named_in_b = {s["label"] for s in d["stacks_all"] if s["silicon"]}
    got = place.solve(items, miss=[MISS_NAMED if m[0]["label"] in named_in_b else MISS for m in head],
                      tries=LABEL_TRIES, seed=LABEL_SEED)
    missing = [m[0]["label"] for m, g in zip(head, got) if g is None]
    for (p, px, py, r), g in zip(head, got):
        if g:
            if g[1]:
                place.leader(g[1])
            place.text(g[0], p["label"], fsk, color=S.C["ink"])


    diag = PLOT_H / pw
    place.stack_air = S.INK_CLEAR_PT
    for e, px, py, r in refused_marks:
        up = py >= px * diag
        angles = ((135, 160, 110, 180, 90, 200, 65, 225, 40, 250, 20, 0, 270, 300, 330, 235, 150, 120) if up
                  else (-45, -20, -70, 0, -90, 20, -115, 45, -140, 70, 180, 200, 90, 135, -30, -60, 160, 110))
        w_e = S.text_width_pt(fig, e["label"], fsk)
        place.ink_ext[(px, py, w_e)] = ink_of(e["label"], fsk)


        angles = angles + tuple(a for a in range(0, 360, REFUSAL_GRID[0]) if a not in angles and a - 360 not in angles)
        fine = tuple(k * REFUSAL_GRID[1] for k in range(1, int(reaches[-1] / REFUSAL_GRID[1]) + 1))
        hit = (place.find(px, py, w_e, fsk, r, angles, fine, [tuple(seg) for seg in band_edges])
               or place.find(px, py, w_e, fsk, r, angles, fine))
        if hit is None:
            missing.append(e["label"])
            continue
        place.block(hit[0])
        if hit[1]:
            place.leaders.append(hit[1])
            place.leader(hit[1])
        place.text(hit[0], e["label"], fsk, color=gray)

    for e in band_edges:
        place.add_line(e)
    for m in quiet:
        if m not in alone:
            quiet_label(m)
    if missing:
        print(f"NOTE {NAME}: no free place for the label of {', '.join(missing)}")

    ax.set_ylabel("Estimate (mm²)", fontsize=FS["label"])
    return dict(tone=tone, form=form, draw=draw, key_x=x_key + sum(col_w[:-1]) + TYPE_KEY_SEP * (len(col_w) - 1),
                key_w=col_w[-1], refusal_sources=refusal_sources)


def group_tones():
    ""
    blue = S.C["blue"]
    pub = (S.C["ink2"], S.C["gray"], S.C["gray2"])
    est = (S.shade(blue, 0.30), S.tint(blue, 0.28), S.tint(blue, 0.62))
    return pub, est


def refused_segment(ax, pw, x, y0, v, top):
    ""
    gray = S.C["gray"]
    box = Rectangle((x, y0), v, BAR_H, facecolor="white", edgecolor="none", zorder=3)
    ax.add_patch(box)
    x0, x1 = x / top * pw, (x + v) / top * pw
    k = -BAR_H
    while x0 + k < x1:
        ln = Line2D([(x0 + k) / pw, (x0 + k + BAR_H) / pw], [y0 / PLOT_H, (y0 + BAR_H) / PLOT_H],
                    transform=ax.transAxes, color=gray, linewidth=REFUSED_HATCH_LW, zorder=3.1,
                    solid_capstyle="butt")
        ax.add_line(ln)
        ln.set_clip_path(box)
        k += REFUSED_HATCH
    ins = REFUSED_EDGE / 2.0
    ax.add_patch(Rectangle(((x0 + ins) / pw, (y0 + ins) / PLOT_H), (x1 - x0 - 2 * ins) / pw,
                           (BAR_H - 2 * ins) / PLOT_H, transform=ax.transAxes, facecolor="none", edgecolor=gray,
                           linewidth=REFUSED_EDGE, zorder=3.2))


def panel_b(fig, ax, d, pw, look, keys, x_left, W):
    ""


    chips = [s for s in d["stacks_all"] if s["silicon"]]
    pub_t, est_t = group_tones()
    n = len(chips)
    fsk = FS["small"]
    head = LINE_EM * fsk + 1.5
    pitch = (PLOT_H - head) / n
    top = float(d["stacks_axis"]["axis_end_multiple"])
    longest = max(sum(v["estimate_um2"] for v in s["groups"].values() if v["status"] == PRICED)
                  / s["published_total_um2"] for s in chips)
    if longest > top:
        print(f"FAIL {NAME}: (b): an estimate bar of {longest:.2f} published totals passes the axis end at {top:g}")
    ax.set_xlim(0, top)
    ax.set_ylim(0, PLOT_H)
    keys.append({"what": "(b) the axis end, in published totals (the whole multiple that covers the longest "
                         "estimate bar)", "value": f"{top:g}", "source": d["stacks_axis"]["source"]})
    ax.axvline(1.0, ymin=0, ymax=(PLOT_H - head + 1.0) / PLOT_H, color=S.C["gray"], linewidth=S.LW["grid"] + 0.1,
               zorder=1)
    name_w = x_left["bars"] - x_left["names"]
    for i, s in enumerate(chips):
        yc = PLOT_H - head - (i + 0.5) * pitch
        for kind, tones, y0 in (("published_um2", pub_t, yc + PAIR_GAP / 2.0),
                                ("estimate_um2", est_t, yc - PAIR_GAP / 2.0 - BAR_H)):
            ref_b = "b/pub" if kind == "published_um2" else "b/est"
            _pb = S.marks(fig, ref_b).__enter__()
            x = 0.0
            refused = []
            for (g, gname), tone in zip(GROUPS, tones):
                if g not in s["groups"]:
                    continue
                if kind == "estimate_um2" and s["groups"][g]["status"] != PRICED:
                    refused.append((g, gname))
                    continue
                v = s["groups"][g][kind] / s["published_total_um2"]
                ax.add_patch(Rectangle((x, y0), v, BAR_H, facecolor=tone, edgecolor="white", linewidth=0.35,
                                       zorder=3))
                x += v
                keys.append({"what": f"(b) {s['label']}: {gname.lower()}, "
                                     f"{'published' if kind == 'published_um2' else 'held-out estimate'} (um2), drawn "
                                     "as a share of the published total", "value": f"{s['groups'][g][kind]:.6g}",
                             "source": s["source"], "ref": ref_b})
            if refused:


                um2 = sum(s["groups"][g]["published_um2"] for g, _ in refused)
                v = um2 / s["published_total_um2"]
                refused_segment(ax, pw, x, y0, v, top)
                wl = S.text_width_pt(fig, REFUSAL_TEXT, FS_A)
                xl = (x + v) / top * pw + REFUSED_LABEL_GAP
                ax.text(xl / pw, (y0 + BAR_H / 2.0) / PLOT_H, REFUSAL_TEXT, transform=ax.transAxes, ha="left",
                        va="center_baseline", fontsize=FS_A, color=S.C["gray"], clip_on=False, zorder=6)
                if x_left["bars"] + xl + wl > x_left["end"]:
                    print(f"FAIL {NAME}: (b): the name of {s['label']}'s refused part ends "
                          f"{x_left['bars'] + xl + wl - x_left['end']:.1f} pt past its panel")
                keys.append({"what": f"(b) {s['label']}: {', '.join(n_.lower() for _, n_ in refused)}, a named "
                                     "refusal (no estimate): one gray outlined segment, its length the groups' "
                                     "published area (um2) as a share of the published total",
                             "value": f"{um2:.6g}", "source": s["source"], "ref": "b/est"})
                keys.append({"what": f"(b) {s['label']}: the status of its refused groups and of the stack",
                             "value": f"{s['stack_status']}: {s['groups'][refused[0][0]]['reason']}",
                             "source": s["source"], "ref": "b/est"})
            _pb.__exit__(None, None, None)
        keys.append({"what": f"(b) {s['label']}: published total (um2), {s['scope']}",
                     "value": f"{s['published_total_um2']:.6g}", "source": s["source"], "ref": "b/pub"})

        t = type_of(s["type"])
        ax.plot([-(name_w - 3.0) / pw], [yc / PLOT_H], marker=look["form"][t],
                markersize=MARK_PT[look["form"][t]] - 0.5, color=look["tone"][t], markeredgewidth=0,
                transform=ax.transAxes, clip_on=False, linestyle="none", zorder=5)
        ax.text(-(name_w - 7.0) / pw, yc / PLOT_H, s["label"], transform=ax.transAxes, ha="left", va="center_baseline",
                fontsize=fsk, color=S.C["ink"])

    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_locator(NullLocator())
    ticks = {0.0: "0", 1.0: "Published total"}
    ticks.update({float(k): f"{k}×" for k in range(2, int(top + 1e-9) + 1)})
    ax.xaxis.set_major_locator(FixedLocator(sorted(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: ticks.get(round(v, 6), "")))
    ax.tick_params(axis="x", labelsize=FS["tick"], pad=TICK_PAD, length=TICK_LEN)


    x = x_left["names"] + 0.5
    y = TITLE_H * 0 + FOOT + PLOT_H - LINE_EM * fsk + 0.6
    for (g, gname), tp, te in zip(GROUPS, pub_t, est_t):
        ym = y + (DESC_EM + 0.36) * fsk - 1.7
        for k, tone in enumerate((tp, te)):
            fig.add_artist(Rectangle(((x + 2.6 * k) / W, ym / HEIGHT_PT), 2.6 / W, 3.4 / HEIGHT_PT, facecolor=tone,
                                     edgecolor="none", transform=fig.transFigure, zorder=5))
        fig.text((x + 6.4) / W, y / HEIGHT_PT, gname, ha="left", va="bottom", fontsize=fsk, color=S.C["ink2"])
        x += 6.4 + S.text_width_pt(fig, gname, fsk) + 3.6
    if x - 3.6 > x_left["end"]:
        print(f"FAIL {NAME}: the tones' key of (b) ends {x - 3.6 - x_left['end']:.1f} pt past its panel")


    words = (("Published", pub_t), ("Estimate", est_t))
    total = sum(11.0 + 2.0 + S.text_width_pt(fig, w, fsk) for w, _ in words) + 8.0
    x = x_left["bars"] + pw / 2.0 - total / 2.0 - (x_left["bars"] - x_left["names"]) / 4.0
    yk = XLABEL_UP + DESC_EM * (FS["label"] - fsk)
    for word, tones in words:
        ym = yk + (DESC_EM + 0.36) * fsk - BAR_H / 2.0
        for k, (tone, share) in enumerate(zip(tones, (0.45, 0.2, 0.35))):
            x0 = x + 11.0 * sum((0.45, 0.2, 0.35)[:k])
            fig.add_artist(Rectangle((x0 / W, ym / HEIGHT_PT), 11.0 * share / W, BAR_H / HEIGHT_PT, facecolor=tone,
                                     edgecolor="none", transform=fig.transFigure, zorder=5))
        fig.text((x + 13.0) / W, yk / HEIGHT_PT, word, ha="left", va="bottom", fontsize=fsk, color=S.C["ink2"])
        x += 13.0 + S.text_width_pt(fig, word, fsk) + 8.0


def panel_c(fig, ax, d, pw, keys, forms):
    ""
    nets = d["ladder"]
    blue, gray = S.C["blue"], S.C["gray"]
    order = sorted(range(len(nets)), key=lambda i: -nets[i]["rows"][0]["area_mm2"])
    shades = [S.shade(blue, 0.42), S.tint(blue, 0.40), blue, S.tint(blue, 0.18)]
    tone = {i: shades[k % len(shades)] for k, i in enumerate(order)}
    rows_all = [r for n in nets for r in n["rows"]]
    p_hi = max(r["passes"] for r in rows_all)
    lo = min(r["area_low_mm2"] for r in rows_all) / C_YPAD[0]
    hi = max(r["area_mm2"] for r in rows_all) * C_YPAD[1]

    probe = [{"what": "(c) passes and area", "source": r["area_source"]} for r in rows_all[:1]]
    hidden = S.withheld(probe)
    ax.set_xscale("symlog", linthresh=1.0, linscale=LIN_DEC)
    ax.set_xlim(*S.frame_range(-0.12, p_hi * 1.15, probe, scale="symlog"))
    ax.set_yscale("log")
    lo, hi = S.frame_range(lo, hi, probe, scale="log")
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([0] + [10 ** k for k in range(1, int(math.log10(max(p_hi, 10))) + 1)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: S.thousands(v)))
    decades = [10.0 ** k for k in range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: S.thousands(v) if v >= 1 else f"{v:g}"))
    ax.tick_params(axis="both", labelsize=FS["tick"], pad=TICK_PAD, length=TICK_LEN)
    place = Placer(ax, pw, PLOT_H, room=(0.8, 0.5, -60.0, 0.5))
    fsk = FS["small"]

    ends, trade, steps = [], None, []
    band_edges_c = []
    _pc = S.marks(fig, "c", series="c")
    for i, n in enumerate(nets):
        _pc.__enter__()
        rows = n["rows"]
        ref = rows[0]
        home = [r for r in rows if r["geometry"] == ref["geometry"]]
        ran = [r for r in rows if status_of(r) != "planned"]
        c = tone[i]
        ax.add_patch(Polygon([(r["passes"], r["area_mm2"]) for r in home]
                             + [(r["passes"], r["area_low_mm2"]) for r in reversed(home)], closed=True,
                             facecolor=S.tint(c, 0.80), edgecolor="none", zorder=1))


        if not hidden:
            low = [to_pt(ax, home[0]["passes"], home[0]["area_mm2"])] + \
                [to_pt(ax, r["passes"], r["area_low_mm2"]) for r in home] + \
                [to_pt(ax, home[-1]["passes"], home[-1]["area_mm2"])]
            band_edges_c += list(zip(low, low[1:]))
        ax.plot([r["passes"] for r in home], [r["area_mm2"] for r in home], color=c, linewidth=1.1, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
        if not hidden:
            place.add_line([to_pt(ax, r["passes"], r["area_mm2"]) for r in home])
        for r0, r1 in zip(ran, ran[1:]):
            if r1["geometry"] != r0["geometry"]:
                on, off = STEP_DASH
                wide = STEP_LW + 2.0 * STEP_CASE
                ax.plot([r0["passes"], r1["passes"]], [r0["area_mm2"], r1["area_mm2"]], color=S.C["white"],
                        zorder=4.0, linewidth=wide, solid_capstyle="butt", dash_capstyle="butt",
                        linestyle=(STEP_CASE / wide, ((on + 2.0 * STEP_CASE) / wide, (off - 2.0 * STEP_CASE) / wide)))
                ax.plot([r0["passes"], r1["passes"]], [r0["area_mm2"], r1["area_mm2"]], color=S.C["ink"], zorder=4.1,
                        linewidth=STEP_LW, dash_capstyle="butt", linestyle=(0, (on / STEP_LW, off / STEP_LW)))
                if not hidden:
                    place.add_line([to_pt(ax, r0["passes"], r0["area_mm2"]), to_pt(ax, r1["passes"], r1["area_mm2"])])
                steps.append((n, r1))
        for r in rows:
            form = status_of(r)
            forms.add(form)
            x, y = r["passes"], r["area_mm2"]
            if form == "planned":
                ax.plot([x], [y], marker="o", markersize=PLAN_PT, markerfacecolor="white", markeredgecolor=c,
                        markeredgewidth=0.6, zorder=5, clip_on=False)
            elif form == "pending":
                ax.plot([x], [y], marker="o", markersize=EXEC_PT + 1.0, markerfacecolor="white",
                        markeredgecolor=gray, markeredgewidth=0.9, zorder=6, clip_on=False)
            else:
                ax.plot([x], [y], marker="o", markersize=EXEC_PT + 0.4, color=c, markeredgecolor="white",
                        markeredgewidth=0.35, zorder=6, clip_on=False)
            radius = (PLAN_PT + 0.6 if form == "planned" else EXEC_PT + 0.75) / 2.0
            if not hidden:
                place.add_mark(*to_pt(ax, x, y), radius)
            keys.append({"what": f"(c) {n['label']} {r['budget']} ({r['status']}): scheduled passes",
                         "value": r["passes"], "source": r["passes_source"]})
            keys.append({"what": f"(c) {n['label']} {r['budget']} ({r['status']}): estimated chip area (mm2) at "
                                 f"{S.thousands(r['cores'])} cores; the band's low edge {r['area_low_mm2']:.5g}",
                         "value": f"{r['area_mm2']:.5g}", "source": r["area_source"]})
        _pc.__exit__(None, None, None)
        last = max(home, key=lambda r: r["passes"])
        ends.append((i, n["label"].partition(", ")[0], to_pt(ax, last["passes"], last["area_mm2"])))
        done = [r for r in home if status_of(r) == "checked"]
        if done and (trade is None or ref["area_mm2"] > trade[1]["area_mm2"]):
            trade = (i, ref, min(done, key=lambda r: r["area_mm2"]), n)


    for die in d["dies"]:
        if not lo < die["mm2"] < hi:
            continue
        y_pt = to_pt(ax, 1.0, die["mm2"])[1]
        text = f"{die['label']} die"
        fsd = FS_DIE
        w = S.text_width_pt(fig, text, fsd)
        h, dsc, air = LINE_EM * fsd, DESC_EM * fsd, AIR_EM * fsd
        for y0 in (y_pt - 0.8 - h + air, y_pt + 1.0 - dsc):
            box = (2.0, y0, 2.0 + w, y0 + h)
            ink = (2.0, y0 + dsc - 0.3, 2.0 + w, y0 + h - air + 0.3)
            if place.free(box, ink):
                ax.axhline(die["mm2"], color=S.C["gray2"], linewidth=0.55, zorder=0)
                place.block(box)
                place.text(box, text, fsd, color=gray)
                keys.append({"what": f"(c) {die['label']}: published die area (mm2), a reference line",
                             "value": f"{die['mm2']:g}", "source": die["source"]})
                break
        else:
            print(f"NOTE {NAME}: the die line of {die['label']} is dropped (its name meets another)")


    step = LINE_EM * fsk - 0.2
    ends.sort(key=lambda e: e[2][1])
    ys = []
    for k, (_, _, (ex, ey)) in enumerate(ends):
        ys.append(max(ey, (ys[-1] + step) if ys else 2.2))
    over = ys[-1] - max(e[2][1] for e in ends)
    ys = [y - min(over / 2.0, ys[0] - 2.2) for y in ys] if over > 0 else ys
    x_lab = pw + 2.5
    for (i, name, (ex, ey)), yl in zip(ends, ys):
        lead = Line2D([(ex + 2.6) / pw, (x_lab - 1.2) / pw], [ey / PLOT_H, yl / PLOT_H], transform=ax.transAxes,
                      color=tone[i], linewidth=0.6, zorder=3, clip_on=False)
        ax.add_line(lead)
        t_name = ax.text(x_lab / pw, (yl - (DESC_EM + 0.36) * fsk) / PLOT_H, name, transform=ax.transAxes, ha="left",
                         va="bottom", fontsize=fsk, color=S.C["ink"], clip_on=False, zorder=7)


        S.anchored(fig, "c", t_name, leaders=[lead], column=(ax, x_lab / pw, 1.0, "axes", "left"), order=i,
                   pitch=step)
        y0 = yl - (DESC_EM + 0.36) * fsk
        place.block((x_lab, y0, x_lab + S.text_width_pt(fig, name, fsk), y0 + LINE_EM * fsk))
        place.add_line([(ex + 2.6, ey), (x_lab - 1.2, yl)])


    _pc.__enter__()
    if trade:
        i, ref, row, n = trade
        ratio = ref["area_mm2"] / row["area_mm2"]
        tx, ty = to_pt(ax, row["passes"], row["area_mm2"])
        text = f"{factor_text(ratio).removesuffix('.0')}× less area"


        ty0 = to_pt(ax, ref["passes"], ref["area_mm2"])[1]
        bx, dark = tx + BRACKET_DX, S.shade(blue, 0.42)
        ax.add_line(Line2D([(bx - BRACKET_TICK) / pw, bx / pw, bx / pw, (bx - BRACKET_TICK) / pw],
                           [ty0 / PLOT_H, ty0 / PLOT_H, ty / PLOT_H, ty / PLOT_H], transform=ax.transAxes,
                           color=dark, linewidth=0.7, zorder=6, clip_on=False, solid_joinstyle="miter"))
        ym = (ty0 + ty) / 2.0
        ax.text((bx + 2.5) / pw, (ym - (DESC_EM + 0.36) * FS["label"]) / PLOT_H, text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=FS["label"], color=dark, clip_on=False, zorder=7)
        wt = S.text_width_pt(fig, text, FS["label"])
        place.block((bx - BRACKET_TICK, ym - (DESC_EM + 0.36) * FS["label"], bx + 2.5 + wt,
                     ym - (DESC_EM + 0.36) * FS["label"] + LINE_EM * FS["label"]))
        place.add_line([(bx, ty0), (bx, ty)])
        keys.append({"what": f"(c) value label: {n['label']}, resident area over the area at "
                             f"{S.thousands(row['cores'])} cores and {row['passes']} passes ({row['budget']}, equal "
                             "outputs)", "value": text, "source": f"{ref['area_source']} ; {row['area_source']}"})

    for n, r1 in steps:
        sx, sy = to_pt(ax, r1["passes"], r1["area_mm2"])
        hit = place.find(sx, sy, S.text_width_pt(fig, str(r1["budget"]), fsk), fsk, (EXEC_PT + 0.4) / 2.0,
                         (-90, -60, -120, 0, 180, -30, -150, 90, 60, 120), (1.0, 2.0, 3.0, 4.5, 6.0, 8.0),
                         off=band_edges_c)
        if hit:
            place.block(hit[0])
            if hit[1]:
                place.leader(hit[1])
            place.text(hit[0], str(r1["budget"]), fsk, color=S.C["ink"])
            keys.append({"what": f"(c) {n['label']}: the budget's name at the end of the dashed step",
                         "value": str(r1["budget"]), "source": r1["area_source"]})
        elif hidden:
            keys.append({"what": f"(c) {n['label']}: the budget's name at the end of the dashed step",
                         "value": str(r1["budget"]), "source": r1["area_source"]})
        else:
            print(f"NOTE {NAME}: no free place for the name of budget {r1['budget']} of {n['label']}")
    _pc.__exit__(None, None, None)

    ylab = "Estimated chip area (mm²)"
    ax.set_ylabel(ylab, fontsize=FS["label"] if S.text_width_pt(fig, ylab, FS["label"]) <= PLOT_H else FS["small"])


def title(fig, W, x, text):
    fig.text(x / W, (FOOT + PLOT_H + TITLE_PAD) / HEIGHT_PT, text, ha="left", va="bottom", fontsize=S.FS["title"],
             fontweight="bold", color=S.C["ink"])


def main():
    global PLOT_H
    d = S.load(NAME)
    S.setup()
    plt.rcParams["figure.constrained_layout.use"] = False
    W = S.width_pt("text")
    fig = plt.figure(figsize=S.figsize("text", HEIGHT_PT))
    fig.canvas.draw()
    a_left = EDGE + YLAB_W + YLAB_GAP + A_TICK_W + TICK_PAD + TICK_LEN
    b_names = a_left + A_PLOT_W + AB_GAP
    b_bars = b_names + B_NAME_W
    b_end = b_bars + B_BAR_W
    c_left = b_end + BC_GAP + YLAB_W + YLAB_GAP + C_TICK_W + TICK_PAD + TICK_LEN
    c_w = W - EDGE - C_MARGIN - c_left
    box = lambda x, w, h=PLOT_H: [x / W, FOOT / HEIGHT_PT, w / W, h / HEIGHT_PT]
    full = PLOT_H
    ax_a = fig.add_axes(box(a_left, A_PLOT_W, full - BAND_H))
    ax_b = fig.add_axes(box(b_bars, B_BAR_W, full))
    ax_c = fig.add_axes(box(c_left, c_w, full))
    keys, forms = [], set()

    PLOT_H = full - BAND_H
    look = panel_a(fig, ax_a, d, A_PLOT_W, keys)
    PLOT_H = full
    for r in keys:
        r["ref"] = None if r["what"].startswith("(a) key") else "a"
    panel_b(fig, ax_b, d, B_BAR_W, look, keys, dict(names=b_names, bars=b_bars, end=b_end + BC_GAP - 1.5), W)
    for r in keys:
        r.setdefault("ref", None)
    k_c = len(keys)
    panel_c(fig, ax_c, d, c_w, keys, forms)
    for r in keys[k_c:]:
        r["ref"] = None if "published die area" in r["what"] else "c"
    ax_a.yaxis.set_label_coords(-(A_TICK_W + TICK_PAD + TICK_LEN + YLAB_GAP) / A_PLOT_W, 0.5)
    ax_c.yaxis.set_label_coords(-(C_TICK_W + TICK_PAD + TICK_LEN + YLAB_GAP) / c_w, 0.5)

    title(fig, W, a_left, "(a) Estimate against published area")
    room_b = c_left - C_TICK_W - YLAB_W - b_names
    for t in ("Published and estimated components", "Area by component"):
        if S.text_width_pt(fig, f"(b) {t}", S.FS["title"], "bold") <= room_b:
            break
    title(fig, W, b_names, f"(b) {t}")
    title(fig, W, c_left, "(c) Chip area against passes")


    fsl, fsk = FS["label"], FS["small"]
    y_txt = XLABEL_UP / HEIGHT_PT
    y_key = (XLABEL_UP + DESC_EM * (fsl - fsk)) / HEIGHT_PT
    y_mark = (XLABEL_UP + (DESC_EM + 0.36) * fsl) / HEIGHT_PT

    word = MODEL_TEXT
    wk = max(S.text_width_pt(fig, word, FS_A), look["key_w"] - 7.0)
    x_col = a_left + look["key_x"]
    fig.text((x_col + 7.0) / W, (XLABEL_UP + DESC_EM * (fsl - FS_A)) / HEIGHT_PT, word, ha="left", va="bottom",
             fontsize=FS_A, color=S.C["ink2"])
    fig.add_artist(Line2D([(x_col + 3.0) / W], [y_mark], transform=fig.transFigure, linestyle="none", marker="o",
                          markersize=QUIET_PT, color=S.C["gray"], markeredgewidth=0))
    x_name = a_left + (A_PLOT_W - wk - 12.0) / 2.0
    if look["refusal_sources"]:


        x0 = EDGE + 0.5
        w_ref = S.text_width_pt(fig, REFUSAL_TEXT, FS_A)
        fig.text((x0 + 7.0) / W, (XLABEL_UP + DESC_EM * (fsl - FS_A)) / HEIGHT_PT, REFUSAL_TEXT, ha="left",
                 va="bottom", fontsize=FS_A, color=S.C["ink2"])


        xc = x0 + 0.6 + (REFUSAL_CAP + 0.6) / 2.0
        fig.add_artist(Line2D([(x0 + 0.6) / W, (x0 + 0.6 + REFUSAL_CAP + 0.6) / W], [y_mark + 1.4 / HEIGHT_PT] * 2,
                              transform=fig.transFigure, color=S.C["gray"], linewidth=REFUSAL_LW,
                              solid_capstyle="butt"))
        fig.add_artist(Line2D([xc / W, xc / W], [y_mark + 1.4 / HEIGHT_PT, y_mark - 1.6 / HEIGHT_PT],
                              transform=fig.transFigure, color=S.C["gray"], linewidth=REFUSAL_LW,
                              solid_capstyle="butt"))
        x_name = (x0 + 7.0 + w_ref + x_col) / 2.0
        w_name = S.text_width_pt(fig, "Published area (mm²)", fsl)
        gap = min(x_name - w_name / 2.0 - (x0 + 7.0 + w_ref), x_col - (x_name + w_name / 2.0))
        if gap < FOOT_KEY_GAP:
            print(f"FAIL {NAME}: the axis name of (a) stands {gap:.1f} pt from a key of the foot line "
                  f"(least {FOOT_KEY_GAP:g} pt)")
        keys.append({"what": "(a) key: the one entry of the gray strokes (a named refusal's readings: status or "
                             "strict_status 'named refusal')", "value": REFUSAL_TEXT,
                     "source": "; ".join(look["refusal_sources"])})
    fig.text(x_name / W, y_txt, "Published area (mm²)", ha="center", va="bottom", fontsize=fsl)
    x = W - EDGE - 1.0
    marks = {"checked": dict(markersize=EXEC_PT + 0.4, color=S.C["blue"], markeredgewidth=0),
             "pending": dict(markersize=EXEC_PT + 1.0, markerfacecolor="white", markeredgecolor=S.C["gray"],
                             markeredgewidth=0.9),
             "planned": dict(markersize=PLAN_PT, markerfacecolor="white", markeredgecolor=S.C["blue"],
                             markeredgewidth=0.6)}
    for form, text in reversed(KEY_FORMS):
        if form not in forms:
            continue
        w = S.text_width_pt(fig, text, fsk)
        fig.text(x / W, y_key, text, ha="right", va="bottom", fontsize=fsk, color=S.C["ink2"])
        fig.add_artist(Line2D([(x - w - 4.6) / W], [y_mark], transform=fig.transFigure, linestyle="none", marker="o",
                              **marks[form]))
        x -= w + 4.6 + 3.5 + 6.0
    wl = S.text_width_pt(fig, "Scheduled passes", fsl)
    xc = min(c_left + c_w / 2.0, x - 2.0 - wl / 2.0)
    fig.text(xc / W, y_txt, "Scheduled passes", ha="center", va="bottom", fontsize=fsl)

    for r in keys:
        r.setdefault("ref", None)


    S.pending_key_at(fig, 0.03, 0.1, ha="left", ax=ax_c, words="Release version", size=fsk)
    S.save_fig(fig, NAME, rows=keys)
    S.save_keys(NAME, keys)


if __name__ == "__main__":
    main()
