""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S

S.FS["title"] = 7.5

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from sciviz.connect.trunk import trunk_route
from sciviz.inkcenter import center_on_ink

NAME = "fig_places"
D = S.load(NAME)


S.MIN_FONT_PT = 5.5


S.INK_CLEAR_PT = 0.6


S.MIN_ARM_PT = 5.0
FS_MIN = 6.0
FS_FOCAL = 8.0
TITLE_PAD = 2.3
S.setup()
BLK, CEN = D["block"], D["census"]
CNT = CEN["counts"]


W, H = S.TEXT_PT, 117.4
TOP = 10.3
WA, GAP = 268.0, 10.0
XB, WB = WA + GAP, W - WA - GAP
HA = H - TOP
B2 = 2.2
PLACE_COLOR = {"cores": S.ROLE["cores"], "operand": S.ROLE["operand"], "host": S.ROLE["host"]}

fig = plt.figure(figsize=S.figsize("text", H))
fig.set_layout_engine("none")
REN = fig.canvas.get_renderer()


def pt_axes(x, y, w, h, *, points=False, **kw):
    ""
    ax = fig.add_axes([x / W, y / H, w / W, h / H], **kw)
    if points:
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_axis_off()
    return ax


def text_w(s, size, weight="normal"):
    ""
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(REN).width * 72.0 / fig.dpi
    t.remove()
    return w


def run_text(ax, x, y, parts, *, gap=3.0, va="baseline", ha="left"):
    ""
    widths = [text_w(s, kw.get("size", S.FS["label"]), kw.get("weight", "normal")) for s, kw in parts]
    if ha == "right":
        x -= sum(widths) + gap * (len(parts) - 1)
    for (s, kw), w in zip(parts, widths):
        kw = dict(kw)
        size, weight = kw.pop("size", S.FS["label"]), kw.pop("weight", "normal")
        ax.text(x, y, s, fontsize=size, fontweight=weight, ha="left", va=va, **kw)
        x += w + gap
    return x - gap


axa = pt_axes(0, 0, WA, HA, points=True)
TITLE_A = f"Placement of one {BLK['instance']['value']} encoder block"
_title_a = S.panel_title(axa, "a", TITLE_A, pad=TITLE_PAD, y=1.0)
S.relabel(fig, "a", _title_a, "(a) Placement of one encoder block")
if text_w(f"(a) {TITLE_A}", S.FS["title"], "bold") > XB - 6.0:
    print(f"FAIL {NAME}: the title of (a) runs into the title of (b)")


LANE_H, LANE_GAP, LEAD, FORK = 22.4, 1.2, 8.2, 3.2
B2A = 2.2
LANE = {"host": (19.8, 19.8 + LANE_H)}
LANE["cores"] = (LANE["host"][1] + LANE_GAP, LANE["host"][1] + LANE_GAP + LANE_H)
Y_NODE = {"cores": LANE["cores"][0] + 7.2, "host": LANE["host"][0] + 13.4}
RULER = {"passes": LANE["cores"][1] + 4.6}
RULER["segment"] = RULER["passes"] + LEAD
BOX_H, CELL_H, R_ADD = 10.0, 7.0, 3.6
NAMED_CELL_H = 8.6
Y_JOIN = RULER["segment"] + 5.0
Y_MODEL = Y_JOIN + FORK + BOX_H / 2
GROUP_Y = (Y_MODEL - 5.8, Y_MODEL + 11.4)
Y_GROUP_NAME = Y_MODEL + 5.5


Y_ARC = GROUP_Y[1] + 3.1
HEAD_MODEL = "-|>,head_width=0.16,head_length=0.34"
HEAD_PT = 0.34 * plt.rcParams["font.size"]
HOST_LABEL_GAP = 0.6
SIDE, RAD = 9.0, 3.6
Y_FORK = Y_NODE["host"] + RAD + 0.4 + FORK
R_FORK = 2.0
FS_NODE, FS_RULE, FS_MODEL = S.FS["small"], S.FS["small"], FS_MIN
SEG_LINE = dict(color=S.C["gray"], linewidth=0.9, solid_capstyle="butt")
PASS_LINE = dict(color=S.C["ink2"], linewidth=0.7, linestyle=(0, (2.2, 1.6)), dash_capstyle="butt")
MODEL_LINE = dict(color=S.C["ink2"], linewidth=0.6, solid_capstyle="butt", solid_joinstyle="round")

axa.add_patch(Rectangle((0, LANE["cores"][0]), WA, LANE["cores"][1] - LANE["cores"][0], zorder=0,
                        facecolor=S.tint(S.C["blue"], 0.91), edgecolor="none"))
axa.add_patch(Rectangle((0, LANE["host"][0]), WA, LANE["host"][1] - LANE["host"][0], zorder=0,
                        facecolor=S.tint(S.C["gray"], 0.86), edgecolor="none"))
DEP = BLK["deployment"]
S.add_marks(fig, "a", [axa.text(5, sum(LANE["cores"]) / 2, f"{DEP['cores']['value']}\ncores", fontsize=S.FS["label"],
                                   fontweight="bold", color=S.C["blue"], ha="left", va="center", linespacing=1.05)],
            series="a")
axa.text(5, sum(LANE["host"]) / 2, "Host\nCPU", fontsize=S.FS["label"], fontweight="bold", color=S.C["ink2"],
         ha="left", va="center", linespacing=1.05)
axa.text(5, RULER["segment"], "Segment", fontsize=FS_RULE, color=S.C["ink2"], ha="left", va="center")
axa.text(5, RULER["passes"], "Passes", fontsize=FS_RULE, color=S.C["ink2"], ha="left", va="center")
_marks_a = S.marks(fig, "a", series="a").__enter__()

ops, SEGS, MDL = BLK["ops"], BLK["segments"], BLK["model"]
seg_of = {o: k for k, sg in enumerate(SEGS) for o in sg["ops"]}


added = set(MDL["inserted"])
group_of = {o: g for g, grp in enumerate(MDL["groups"]) for o in grp["ops"]}
res_of = {r["add"]: r for r in MDL["residuals"]}


def form(i):
    return "none" if i in added else "cell" if i in group_of else "plus" if i in res_of else "box"


def named(i):
    ""
    return form(i) == "box" or (form(i) == "cell" and ops[i]["place"] == "host")


model_ops = [i for i in range(len(ops)) if form(i) != "none"]
item_of = {i: ("group", group_of[i]) if form(i) == "cell" else ("op", i) for i in model_ops}

junction_before = set()
for r in MDL["residuals"]:
    s = r["start"]["op"]
    if s not in model_ops or r["add"] not in model_ops or form(r["add"]) != "plus":
        print(f"FAIL {NAME}: a residual connection of the data names no model operation")
    junction_before.add(s if r["start"]["side"] == "before" else model_ops[model_ops.index(s) + 1])


marks, drawn_segs = [], set()
for i, o in enumerate(ops):
    if o["place"] == "host":
        marks.append({"op": i, "lane": "host"})
    elif seg_of[i] not in drawn_segs:
        drawn_segs.add(seg_of[i])
        for q, in_pass in enumerate(SEGS[seg_of[i]]["pass_ops"]):
            marks += [{"op": j, "lane": "cores", "seg": seg_of[i], "pass": q} for j in in_pass]
groups = {}
for k, m in enumerate(marks):
    groups.setdefault(m["op"], []).append(k)
for i, ks in groups.items():
    if ks != list(range(ks[0], ks[-1] + 1)):
        print(f"FAIL {NAME}: the passes of {ops[i]['label']} are not drawn side by side")


lw = {i: (0.0 if form(i) == "box" and ops[i]["place"] != "host" else text_w(ops[i]["label"], FS_NODE))
      for i in groups}
PADX = 1.2
bw = {i: text_w(ops[i]["label"], FS_MODEL) + 2 * PADX for i in groups if named(i)}


CELL_HALF, CELL_GAP, GPAD, ITEM_GAP, JUNC = 3.5, 2.0, 2.2, 2.2, 2.8
CELL_PAIR = 7.0
SHIFT_MAX = 5.5


def shift_cap(i):
    ""
    return max(0.0, min(SHIFT_MAX, lw[i] / 2 - RAD - 2.0)) if lw[i] else 0.0


def half_w(i):
    ""
    f = form(i)
    return bw[i] / 2 if named(i) else R_ADD if f == "plus" else CELL_HALF + (len(groups[i]) - 1) * CELL_PAIR / 2.0


MIN_STEP, PAIR, PAD, PAD_LINE = 10.5, 12.0, 3.5, 3.0
xs, right = [], {"host": None, "cores": None}
last_host, last_item, item_right, op_right = None, None, None, None
give = 0.0
for k, m in enumerate(marks):
    i, ks = m["op"], groups[m["op"]]
    if k > ks[0]:
        xs.append(xs[-1] + PAIR)
        continue
    x = xs[-1] + MIN_STEP if xs else 0.0
    half = (len(ks) - 1) * PAIR / 2.0
    if right[m["lane"]] is not None:

        relief = give + shift_cap(i) if m["lane"] == "host" else 0.0
        x = max(x, right[m["lane"]] + PAD + lw[i] / 2 - half - relief)
    if m["lane"] == "host":
        if right["cores"] is not None:
            x = max(x, right["cores"] + PAD_LINE)
    elif last_host is not None:
        x = max(x, last_host + PAD_LINE + lw[i] / 2 - half)
    if form(i) != "none":
        if item_of[i] == last_item:
            x = max(x, op_right + CELL_GAP + half_w(i) - half)
        elif last_item is not None:
            lead = ITEM_GAP + (JUNC if i in junction_before else 0.0) + (GPAD if form(i) == "cell" else 0.0)
            x = max(x, item_right + lead + half_w(i) - half)
        op_right = x + half + half_w(i)
        item_right, last_item = op_right + (GPAD if form(i) == "cell" else 0.0), item_of[i]
    if m["lane"] == "host":
        last_host = x
    xs.append(x)

    if m["lane"] == "host":
        free = x - lw[i] / 2 - (right["host"] + PAD) if right["host"] is not None else SHIFT_MAX
        give = min(shift_cap(i), max(0.0, free))
    right[m["lane"]] = x + half + (lw[i] / 2 if lw[i] else RAD + 1.0 - PAD)


HEAD_LANE = "-|>,head_width=0.19,head_length=0.34"
X_LO, X_TAIL = 35.5, RAD + 0.4 + 3.0 * HEAD_PT
x_lo, x_hi = X_LO, WA - X_TAIL
scale = (x_hi - x_lo) / (xs[-1] - xs[0])
print(f"     {NAME}: the chain of (a) needs {xs[-1] - xs[0]:.1f} pt, {x_hi - x_lo:.1f} pt at hand")
if scale < 1.0:
    print(f"FAIL {NAME}: the chain of (a) needs {xs[-1] - xs[0]:.1f} pt, {x_hi - x_lo:.1f} pt at hand")
xs = [x_lo + (x - xs[0]) * scale for x in xs]
ys = [Y_NODE[m["lane"]] for m in marks]
centre = {i: sum(xs[k] for k in ks) / len(ks) for i, ks in groups.items()}

shift, prev, prev_free = {i: 0.0 for i in groups}, None, SHIFT_MAX
for i in [m["op"] for m in marks if m["lane"] == "host"]:
    if prev is not None:
        end = centre[prev] + shift[prev] + (lw[prev] / 2 if lw[prev] else RAD + 1.0 - PAD)
        over = end + PAD - (centre[i] - lw[i] / 2) if lw[i] else 0.0
        free = -over
        if over > 0:
            back = min(shift_cap(prev), max(0.0, prev_free))
            s_prev = min(over / 2, back)
            s_here = min(over - s_prev, shift_cap(i))
            s_prev = min(over - s_here, back)
            if s_prev + s_here < over - 0.05:
                print(f"FAIL {NAME}: the host labels {ops[prev]['label']} and {ops[i]['label']} touch")
            shift[prev] -= s_prev
            shift[i] += s_here
            free = 0.0
        prev_free = free
    prev = i


host_runs, run = [], []
for k, m in enumerate(marks):
    if m["lane"] == "host":
        run.append(k)
    elif run:
        host_runs.append(run)
        run = []
if run:
    host_runs.append(run)
bounds = [sum(xs[k] for k in r) / len(r) for r in host_runs]
if marks[0]["lane"] != "host" or marks[-1]["lane"] != "host" or len(bounds) != len(SEGS) + 1:
    print(f"FAIL {NAME}: the block's chain does not open and close on a host operation")


PATH_CODE = {"M": [MPath.MOVETO], "L": [MPath.LINETO], "Q": [MPath.CURVE3, MPath.CURVE3]}
for r, xb in zip(host_runs, bounds):
    route = trunk_route([(xs[k], Y_MODEL) for k in r], [(xs[k], Y_NODE["host"]) for k in r], join=Y_JOIN,
                        fork=Y_FORK, trunk=xb)
    for stroke in route.commands(R_FORK):
        verts = [pt for cmd in stroke for pt in cmd[1:]]
        codes = [c for cmd in stroke for c in PATH_CODE[cmd[0]]]
        axa.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=SEG_LINE["color"], zorder=1,
                                linewidth=SEG_LINE["linewidth"], capstyle="butt", joinstyle="round"))
for k, sg in enumerate(SEGS):
    xc = (bounds[k] + bounds[k + 1]) / 2
    axa.text(xc, RULER["segment"], f"{sg['order']}", fontsize=FS_RULE, fontweight="bold", color=S.C["ink"],
             ha="center", va="center")
    axa.text(xc, RULER["passes"], f"{sg['passes']}", fontsize=FS_RULE, color=S.C["ink"], ha="center",
             va="center")


Y_LABEL = Y_NODE["cores"] + SIDE / 2 + 1.2
pass_lines = 0
for k in range(1, len(marks)):
    a, b = marks[k - 1], marks[k]
    if a["lane"] == b["lane"] == "cores" and a["seg"] == b["seg"] and a["pass"] != b["pass"]:
        xp = (xs[k - 1] + xs[k]) / 2
        under = any(abs(xp - centre[i]) < lw[i] / 2 + 1.5 for i in groups if ops[i]["place"] != "host")
        axa.plot([xp, xp], [LANE["cores"][0], Y_LABEL - 1.6 if under else LANE["cores"][1]], zorder=1, **PASS_LINE)
        pass_lines += 1
if pass_lines != sum(sg["passes"] - 1 for sg in SEGS):
    print(f"FAIL {NAME}: {pass_lines} pass boundaries drawn, the data holds {sum(sg['passes'] - 1 for sg in SEGS)}")

axa.plot(xs, ys, color=S.C["ink2"], linewidth=S.LW["thin"], zorder=2, solid_joinstyle="round")
axa.annotate("", xy=(xs[-1] + X_TAIL, ys[-1]), xytext=(xs[-1] + RAD + 0.4, ys[-1]), zorder=2, annotation_clip=False,
             arrowprops=dict(arrowstyle=HEAD_LANE, color=S.C["ink2"],
                             linewidth=S.LW["thin"], shrinkA=0, shrinkB=0))
for m, x, y in zip(marks, xs, ys):
    c = PLACE_COLOR[ops[m["op"]]["place"]]
    if m["lane"] == "cores":
        axa.add_patch(FancyBboxPatch((x - SIDE / 2 + 1.5, y - SIDE / 2 + 1.5), SIDE - 3, SIDE - 3, zorder=3,
                                     boxstyle="round,pad=1.5,rounding_size=1.9", facecolor=c,
                                     edgecolor=S.C["white"], linewidth=0.8))
    else:
        axa.plot([x], [y], marker="o", markersize=2 * RAD + 0.8, markerfacecolor=c, markeredgecolor=S.C["white"],
                 markeredgewidth=0.8, zorder=3)
for i in groups:
    if not lw[i]:
        continue
    if ops[i]["place"] == "host":
        axa.text(centre[i] + shift[i], Y_NODE["host"] - RAD - HOST_LABEL_GAP, ops[i]["label"], fontsize=FS_NODE, ha="center",
                 va="top", color=S.C["ink2"])
    else:
        axa.text(centre[i], Y_LABEL, ops[i]["label"], fontsize=FS_NODE, ha="center", va="bottom", color=S.C["ink"])


ext = {}
for i in model_ops:
    ks = groups[i]
    ext[i] = (centre[i] - half_w(i), centre[i] + half_w(i))
item_ext = {}
for i in model_ops:
    pad = GPAD if form(i) == "cell" else 0.0
    a, b = item_ext.get(item_of[i], (ext[i][0] - pad, ext[i][1] + pad))
    item_ext[item_of[i]] = (min(a, ext[i][0] - pad), max(b, ext[i][1] + pad))
X_IN = item_ext[item_of[model_ops[0]]][0] - JUNC - 5.0


X_OUT = ext[model_ops[-1]][1] + 3.0 * HEAD_PT
axa.plot([X_IN, ext[model_ops[-1]][1]], [Y_MODEL, Y_MODEL], zorder=2, **MODEL_LINE)
axa.annotate("", xy=(X_OUT, Y_MODEL), xytext=(ext[model_ops[-1]][1], Y_MODEL), zorder=2, annotation_clip=False,
             arrowprops=dict(arrowstyle=HEAD_MODEL, color=S.C["ink2"],
                             linewidth=MODEL_LINE["linewidth"], shrinkA=0, shrinkB=0))
for g, grp in enumerate(MDL["groups"]):
    x0, x1 = item_ext[("group", g)]
    axa.add_patch(FancyBboxPatch((x0 + 1.6, GROUP_Y[0] + 1.6), x1 - x0 - 3.2, GROUP_Y[1] - GROUP_Y[0] - 3.2,
                                 boxstyle="round,pad=1.6,rounding_size=2.4", zorder=0.6,
                                 facecolor=S.tint(S.C["gray"], 0.93), edgecolor=S.C["gray2"], linewidth=0.5))
    axa.text(x0 + GPAD, Y_GROUP_NAME, grp["name"], fontsize=FS_MODEL, color=S.C["ink2"], ha="left", va="baseline")
    if text_w(grp["name"], FS_MODEL) > x1 - x0 - 2 * GPAD:
        print(f"FAIL {NAME}: the group name {grp['name']} is wider than its frame")
for i in model_ops:
    c, (x0, x1) = PLACE_COLOR[ops[i]["place"]], ext[i]
    if form(i) == "plus":
        axa.add_patch(Circle((centre[i], Y_MODEL), R_ADD, zorder=3, facecolor=S.C["white"], edgecolor=S.C["ink2"],
                             linewidth=0.7))
        for dx, dy in ((R_ADD - 1.5, 0.0), (0.0, R_ADD - 1.5)):
            axa.plot([centre[i] - dx, centre[i] + dx], [Y_MODEL - dy, Y_MODEL + dy], zorder=4, color=S.C["ink2"],
                     linewidth=0.7, solid_capstyle="butt")
        continue
    h = BOX_H if form(i) == "box" else NAMED_CELL_H if named(i) else CELL_H
    axa.add_patch(FancyBboxPatch((x0 + 1.4, Y_MODEL - h / 2 + 1.4), x1 - x0 - 2.8, h - 2.8, zorder=3,
                                 boxstyle="round,pad=1.4,rounding_size=1.8", facecolor=S.tint(c, 0.62),
                                 edgecolor=c, linewidth=0.6))
    if named(i):
        center_on_ink(axa.text(centre[i], Y_MODEL, ops[i]["label"], fontsize=FS_MODEL, color=S.C["ink"], ha="center",
                               va="center", zorder=4), vertical=False)
R_C = R_FORK


if Y_ARC - R_C - (Y_MODEL + R_ADD) - HEAD_PT < S.MIN_ARM_PT:
    print(f"FAIL {NAME}: a skip line's shaft is under {S.MIN_ARM_PT} pt")
for r in MDL["residuals"]:
    s, a = r["start"]["op"], r["add"]
    if r["start"]["side"] == "before":
        xj = item_ext[item_of[s]][0] - JUNC
    else:
        xj = (item_ext[item_of[s]][1] + item_ext[item_of[model_ops[model_ops.index(s) + 1]]][0]) / 2
    xa = centre[a]
    verts = [(xj, Y_MODEL), (xj, Y_ARC - R_C), (xj, Y_ARC), (xj + R_C, Y_ARC), (xa - R_C, Y_ARC), (xa, Y_ARC),
             (xa, Y_ARC - R_C), (xa, Y_ARC - R_C - 0.6)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO]
    axa.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=S.C["ink2"], zorder=2,
                            linewidth=MODEL_LINE["linewidth"], capstyle="butt", joinstyle="round"))
    axa.annotate("", xy=(xa, Y_MODEL + R_ADD), xytext=(xa, Y_ARC - R_C), zorder=2,
                 arrowprops=dict(arrowstyle=HEAD_MODEL, color=S.C["ink2"],
                                 linewidth=MODEL_LINE["linewidth"], shrinkA=0, shrinkB=0))
    axa.add_patch(Circle((xj, Y_MODEL), 1.0, zorder=3, facecolor=S.C["ink2"], edgecolor="none"))


MS_ = BLK["mac_share"]
B1A = B2A + LEAD
FS_SHARE = FS_MIN
BX0, BH, BAR_MID = 5.0, 5.4, B1A + 0.36 * FS_FOCAL
KEYS = (("Segment boundary", SEG_LINE), ("Pass boundary", PASS_LINE))
XK = WA - 1.5 - 4.5 - max(text_w(name, FS_MIN) for name, _ in KEYS)
for (name, style), yk in zip(KEYS, (B1A, B2)):
    axa.plot([XK + 1.0, XK + 1.0], [yk - 1.4, yk + 5.6], **style)
    axa.text(XK + 4.5, yk, name, fontsize=FS_MIN, color=S.C["ink2"], ha="left", va="baseline")
FOCAL = f"{MS_['on_cores']['display']} of multiply-accumulates on cores"
BW = XK - 6.0 - text_w(FOCAL, FS_FOCAL, "bold") - 5.0 - BX0
if BW < 30.0:
    print(f"FAIL {NAME}: {BW:.1f} pt left for the bar of the share row")
BX1, BY = BX0 + BW, BAR_MID - BH / 2
split = BX0 + (BX1 - BX0) * MS_["weights"]["value"] / (MS_["weights"]["value"] + MS_["operand"]["value"])
axa.add_patch(Rectangle((BX0, BY), split - BX0, BH, facecolor=S.ROLE["cores"], edgecolor="none", zorder=2))
axa.add_patch(Rectangle((split, BY), BX1 - split, BH, facecolor=S.ROLE["operand"], edgecolor="none", zorder=2))
axa.text(BX1 + 5.0, B1A, FOCAL, fontsize=FS_FOCAL, fontweight="bold", color=S.C["blue"], ha="left",
         va="baseline")
x = BX0
for name, k, c in (("Cores", "weights", S.C["blue"]), ("Dynamic-weight cores", "operand", S.C["teal"]),
                   ("Host CPU", "host", S.C["ink2"])):
    x = run_text(axa, x, B2, [(name, dict(color=c, size=FS_SHARE)),
                              (MS_[k]["display"], dict(color=c, weight="bold", size=FS_SHARE))], gap=2.6) + 8.0
if x - 8.0 > XK - 5.0:
    print(f"FAIL {NAME}: the share labels of (a) run into the key of the line styles")

_marks_a.__exit__(None, None, None)


axb = pt_axes(XB, 0, WB, HA, points=True)

TITLE_B = f"{CNT['censusmembers']['display']} ImageNet classifiers, {CEN['condition']['label']}"
_title_b = S.panel_title(axb, "b", TITLE_B, pad=TITLE_PAD, y=1.0)
S.relabel(fig, "b", _title_b, "(b) ImageNet classifiers")
W_TITLE_B = text_w(f"(b) {TITLE_B}", S.FS["title"], "bold")
print(f"     {NAME}: the title of (b) is {W_TITLE_B:.1f} pt wide in a panel of {WB:.1f} pt")
if W_TITLE_B > WB - 4.0:
    print(f"FAIL {NAME}: the title of (b) ends {WB - W_TITLE_B:.1f} pt from the panel's right edge (least 4 pt)")

host_rows, core_rows, undec_rows = CEN["host_share"], CEN["on_cores"], CEN["undecidable"]
FS_ROW, FS_HEAD = 6.0, S.FS["small"]
Y_TICK = 1.6
Y_AX = 8.6
Y_TOP = HA - 8.4
PITCH = (Y_TOP - Y_AX) / len(host_rows)
LEAD_MIN = 1.13
print(f"     {NAME}: the rows of (b) stand {PITCH:.2f} pt apart ({PITCH / FS_ROW:.3f} of the {FS_ROW} pt names)")
if PITCH < LEAD_MIN * FS_ROW:
    print(f"FAIL {NAME}: the rows of (b) stand {PITCH:.2f} pt apart, under {LEAD_MIN} of the {FS_ROW} pt names")
Y_HEAD = Y_TOP + 4.45
row_y = lambda i: Y_TOP - (i + 0.5) * PITCH


HEADS = {"cores": "Fully on cores", "undecidable": "Undecided",
         "host": "With a host share of multiply-accumulates"}
X_NAME = max(text_w(HEADS[k], FS_HEAD, "bold") for k in ("cores", "undecidable"))
X_T0, T_PITCH, T_H, T_W = X_NAME + 4.0, 3.0, 5.0, 1.05
X_END = X_T0 + (max(r["count"] for r in core_rows) - 1) * T_PITCH + T_W / 2
CONV = [("Converted", dict(color=S.C["ink2"])),
        (CNT["censusconvert"]["display"], dict(weight="bold", size=FS_FOCAL, color=S.C["ink"]))]
W_CONV = sum(text_w(s, kw.get("size", S.FS["label"]), kw.get("weight", "normal")) for s, kw in CONV) + 3.0

X_R0 = max(X_END, W_CONV) + 8.0
X_NR = X_R0 + max(text_w(r["family"], FS_ROW) for r in host_rows)
X_CNT = X_NR + 3.5
X_D = X_CNT + max(text_w(f"{r['count']}", FS_ROW) for r in host_rows) + 5.5
x_max = 10.0 * math.ceil(max(r["max_percent"] for r in host_rows) / 10.0)
W_D = WB - X_D - text_w(f"{x_max:g}%", FS_MIN) / 2 - 2.0
print(f"     {NAME}: the host-share axis of (b) is {W_D:.1f} pt wide")
_marks_b = S.marks(fig, "b", series="b").__enter__()


SWATCH, SWATCH_GAP = 4.0, 2.6


def head(x_name, x_count, y, name, count, color=S.C["ink"], ha="right", swatch=None):
    ""
    if ha == "left":
        x_count = x_name + text_w(name, FS_HEAD, "bold") + 3.5
    axb.text(x_name, y, name, fontsize=FS_HEAD, fontweight="bold", ha=ha, va="center", color=color)
    axb.text(x_count, y, count, fontsize=FS_HEAD, fontweight="bold", ha="left", va="center", color=color)
    end = x_count + text_w(count, FS_HEAD, "bold")
    if swatch is not None:
        axb.add_patch(Rectangle((end + SWATCH_GAP, y - SWATCH / 2 - 0.3), SWATCH, SWATCH, facecolor=swatch,
                                edgecolor="none", zorder=2))
        end += SWATCH_GAP + SWATCH
    return end


def name_block(x, ys, lines, *, ha, color):
    ""
    for y, line in zip(ys, lines):
        axb.text(x, y, line, fontsize=FS_ROW, ha=ha, va="center", color=color)


def tally(y, n, color):
    axb.plot([X_T0 + k * T_PITCH for k in range(n)], [y] * n, linestyle="none", marker="|", markersize=T_H,
             markeredgewidth=T_W, markeredgecolor=color)


SEG_COLOR = {"cores": S.ROLE["cores"], "host": S.tint(S.C["blue"], 0.6), "undecidable": S.ROLE["host"]}
head(X_NAME, X_T0 - T_W / 2, Y_HEAD, HEADS["cores"], CNT["censusallcores"]["display"], swatch=SEG_COLOR["cores"])
name_block(X_NAME, [row_y(i) for i in range(len(core_rows))], [r["family"] for r in core_rows], ha="right",
           color=S.C["ink"])
for i, r in enumerate(core_rows):
    tally(row_y(i), r["count"], S.ROLE["cores"])
UN_HEAD, UN_ROW = 0.8 * PITCH, Y_HEAD - row_y(0)
y_un = row_y(len(core_rows) - 0.5) - UN_HEAD
head(X_NAME, X_T0 - T_W / 2, y_un, HEADS["undecidable"], CNT["censusundecidable"]["display"], S.C["ink2"],
     swatch=SEG_COLOR["undecidable"])
name_block(X_NAME, [y_un - UN_ROW - k * PITCH for k in range(len(undec_rows))], [r["family"] for r in undec_rows],
           ha="right", color=S.C["ink"])
for k, r in enumerate(undec_rows):
    tally(y_un - UN_ROW - k * PITCH, r["count"], S.ROLE["host"])
y_low = y_un - UN_ROW - (len(undec_rows) - 1) * PITCH - PITCH / 2


B1 = y_low - 8.0
SB_TOP, SB_H = B1 - 3.2, 4.4
if SB_TOP - SB_H < B2 + 0.75 * S.FS["label"] + 1.5:
    print(f"FAIL {NAME}: the left block of (b) does not fit over its last line "
          f"({B2 + 0.75 * S.FS['label'] + 1.5 - SB_TOP + SB_H:.1f} pt short)")
n_all = float(CNT["censusmembers"]["value"])
SEG = [(float(CNT["censusallcores"]["value"]), SEG_COLOR["cores"]),
       (float(CNT["censushostshare"]["value"]), SEG_COLOR["host"]),
       (float(CNT["censusundecidable"]["value"]), SEG_COLOR["undecidable"])]
x = 0.0
for n, c in SEG:
    w = X_END * n / n_all
    axb.add_patch(Rectangle((x, SB_TOP - SB_H), w, SB_H, facecolor=c, edgecolor="none", zorder=2))
    if x > 0:
        axb.plot([x, x], [SB_TOP - SB_H, SB_TOP], color=S.C["white"], linewidth=0.8, zorder=3,
                 solid_capstyle="butt")
    x += w
run_text(axb, 0.0, B1, CONV)
x = run_text(axb, 0.0, B2, [("Differing outputs", dict(color=S.C["ink2"])),
                            (CNT["censussilent"]["display"], dict(weight="bold", color=S.C["ink"]))])
S.zero_mark(axb, x + 6.0, B2 + 2.7, size=6.2)


if head(X_R0, None, Y_HEAD, HEADS["host"], CNT["censushostshare"]["display"], ha="left",
        swatch=SEG_COLOR["host"]) > WB:
    print(f"FAIL {NAME}: the header of the host-share plot runs off the panel")
xd = lambda v: X_D + W_D * v / x_max
R_H, R_MIN, M_H, M_W = 3.0, 5.0, 5.0, 1.1
name_block(X_NR, [row_y(i) for i in range(len(host_rows))], [r["family"] for r in host_rows], ha="right",
           color=S.C["ink"])
name_block(X_CNT, [row_y(i) for i in range(len(host_rows))], [f"{r['count']}" for r in host_rows], ha="left",
           color=S.C["ink2"])
for i, r in enumerate(host_rows):
    y = row_y(i)
    axb.plot([X_D, X_D + W_D], [y, y], color=S.C["gray3"], linewidth=S.LW["grid"], zorder=1)
    x0, x1 = xd(r["min_percent"]), xd(r["max_percent"])
    if x1 - x0 < R_MIN:
        x0 = max(X_D, (x0 + x1 - R_MIN) / 2)
        x1 = x0 + R_MIN
    axb.add_patch(Rectangle((x0, y - R_H / 2), x1 - x0, R_H, facecolor=S.tint(S.ROLE["host"], 0.45),
                            edgecolor="none", zorder=2))
    xm = xd(r["median_percent"])
    axb.plot([xm, xm], [y - M_H / 2, y + M_H / 2], color=S.C["ink2"], linewidth=M_W, zorder=3,
             solid_capstyle="butt")
_marks_b.__exit__(None, None, None)
TICK_L = 1.6
axb.plot([X_D, X_D + W_D], [Y_AX, Y_AX], color=S.C["ink2"], linewidth=S.LW["axis"], zorder=2,
         solid_capstyle="projecting")
for v in (0.0, x_max / 2, x_max):
    axb.plot([xd(v), xd(v)], [Y_AX, Y_AX - TICK_L], color=S.C["ink2"], linewidth=S.LW["axis"], zorder=2)
    axb.text(xd(v), Y_TICK, f"{v:g}%", fontsize=FS_MIN, ha="center", va="baseline", color=S.C["ink"])


rows = [{"what": f"place of operation {o['order']} ({o['label']})", "value": o["place"], "source": o["source"]}
        for o in ops]
rows.append({"what": "block instance", "value": BLK["instance"]["value"], "source": BLK["instance"]["source"]})
for sg in SEGS:
    rows.append({"what": f"segment {sg['order']} of the block ({', '.join(ops[i]['label'] for i in sg['ops'])}): "
                         f"its passes (ruler row, squares per operation, dashed boundaries)",
                 "value": sg["passes"], "source": sg["source"]})
rows.append({"what": "model row (layout, no measured value): the groups, the residual connections "
                     "(start side, start operation, addition) and the operations the toolchain adds",
             "value": "; ".join(
                 [f"group {g['name']}: operations {g['ops']}" for g in MDL["groups"]]
                 + [f"residual connection from {r['start']['side']} operation {r['start']['op']} to the addition "
                    f"{r['add']}" for r in MDL["residuals"]]
                 + [f"added by the toolchain: operations {MDL['inserted']}"]),
             "source": MDL["source"]})
rows.append({"what": "cores of the deployment the passes belong to (the label of the cores lane)",
             "value": DEP["cores"]["value"], "key": DEP["cores"]["key"], "source": DEP["cores"]["source"]})
rows.append({"what": "share of multiply-accumulates on cores", "value": MS_["on_cores"]["display"],
             "key": MS_["on_cores"]["key"]})
for k, what in (("weights", "share on cores that store weights"), ("operand", "share on operand cores"),
                ("host", "share on the host CPU")):
    rows.append({"what": what, "value": MS_[k]["display"], "source": MS_[k]["source"],
                 **{f: MS_[k][f] for f in ("key", "derivation") if f in MS_[k]}})
for r in rows:
    r["ref"] = "a"
k_b = len(rows)
for k in ("censusmembers", "censusconvert", "censussilent", "censusallcores", "censushostshare",
          "censusundecidable"):
    rows.append({"what": k, "value": CNT[k]["display"], "key": CNT[k]["key"]})
for r in core_rows:
    rows += [{"what": f"fully on cores: {m['member']} ({r['family']})", "value": 1, "source": m["source"]}
             for m in r["members"]]
for r in undec_rows:
    rows += [{"what": f"undecided: {m['member']} ({r['family']})", "value": 1, "source": m["source"]}
             for m in r["members"]]
for r in host_rows:
    rows += [{"what": f"host share of multiply-accumulates (%): {m['member']} ({r['family']})",
              "value": m["percent"], "source": m["source"]} for m in r["members"]]
    rows.append({"what": f"range bar and median tick (%): {r['family']} ({r['count']} members)",
                 "value": [r["min_percent"], r["median_percent"], r["max_percent"]],
                 "source": "the family's member rows above (smallest, median, largest)"})
COND = CEN["condition"]
rows.append({"what": f"census condition in the title of (b): {COND['label']}",
             "value": "; ".join(f"{h['family']} {h['members']} ({', '.join(h['kinds'])})"
                                for h in COND["activation_products_on_host"]) + " on the host",
             "source": COND["activation_products_on_host"][0]["sources"][0] + " and the like lines; ledger key "
                       + COND["ledger"]["key"] + f" ({COND['ledger']['display']})"})
for r in rows[k_b:]:
    r["ref"] = "b"
rows += S.caption_keys(NAME)
S.save_fig(fig, NAME, rows=rows)
S.save_keys(NAME, rows)
