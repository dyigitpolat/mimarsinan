""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.MIN_FONT_PT = 6.0
S.FS["title"] = 7.5
S.INK_CLEAR_PT = 1.2
S.MIN_ARM_PT = 3.0

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from sciviz.inkcenter import center_on_ink

NAME = "fig_replay"
D = S.load(NAME)
EX, COUNT = D["example"], D["count"]
S.setup()
FS = {"title": S.FS["title"], "focal": 8.0, "label": 7.0, "tick": 6.5, "small": 6.0}

LAYERS = EX["layers"]
WIN, SPIKES = EX["windows"], EX["spikes"]
NS = EX["steps"]
PASSES = EX["passes"]
STORED = EX["stored_steps"]
DELIV = EX["deliveries"]
OUT_LAYER = EX["output_layer"]
FIRST, LATER = PASSES[0], [n for p in PASSES[1:] for n in p]
N_CHIP = EX["cores_chip"]
N_REF = EX["cores_single_pass"]

assert len(PASSES) == 2 and len(LATER) == N_CHIP and LAYERS == FIRST + LATER and len(FIRST) == 1, PASSES

_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def word(n: int, cap: bool = False) -> str:
    ""
    w = _WORDS[n] if 0 <= n < len(_WORDS) else str(n)
    return w.capitalize() if cap else w


W = S.COLUMN_PT
LEFT = 1.0
STEP, PITCH = 15.5, 12.0
BAND = 8.8
TILE = 10.8
CHIP_PAD = 1.6
CELL = 9.2
BUF_BAND = CELL + 2.0
DOT_R = 2.3
LW_DELIV = 0.8
STORE_HEAD = (1.55, 2.7)
BLOCK = NS * STEP
R_MARGIN = 16.0
X2 = W - R_MARGIN - BLOCK
X2_TILE = X2 - 4.0 - TILE
X1_TILE = LEFT + CHIP_PAD
X1 = X1_TILE + TILE + 4.0

GAP_TEXT = 3.0
ROW_PHASE = 4.9
ROW_BAR = ROW_PHASE + 0.5 * 0.714 * FS["label"] + GAP_TEXT + 0.8
ROW_TICK_B = ROW_BAR + 0.8 + 1.7 + 0.5 * 0.714 * FS["tick"]
TICK_DROP = TILE / 2 + CHIP_PAD + 2.1 + 0.5 * 0.714 * FS["tick"]
Y_CORES = [ROW_TICK_B + TICK_DROP + PITCH * k for k in range(N_CHIP - 1, -1, -1)]
Y_BUF = Y_CORES[0] + PITCH
ROW_TITLE_B = Y_BUF + BUF_BAND / 2 + GAP_TEXT + 0.6 + 0.5 * 0.714 * FS["title"]
ROW_FOCAL = ROW_TITLE_B + 0.5 * 0.714 * (FS["title"] + FS["focal"]) + 4.4
ROW_TICK_A = ROW_FOCAL + 0.5 * 0.714 * (FS["focal"] + FS["tick"]) + GAP_TEXT + 0.3
Y_A = [ROW_TICK_A + TICK_DROP + PITCH * k for k in range(len(LAYERS) - 1, -1, -1)]
ROW_TITLE_A = Y_A[0] + BAND / 2 + 1.4 + 0.5 * 0.714 * FS["small"]
H = ROW_TITLE_A + 0.5 * 0.714 * FS["title"] + 2.6

BLUE, GRAY, ORANGE, GREEN = S.C["blue"], S.C["gray"], S.C["orange"], S.C["green"]
INK, INK2 = S.C["ink"], S.C["ink2"]

fig = plt.figure(figsize=S.figsize("column", H))
fig.set_layout_engine("none")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def sx(x0, step):
    ""
    return x0 + (step + 0.5) * STEP


def rbox(x, y, w, h, *, fc, ec, lw=S.LW["edge"], r=2.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))


def text(x, y, s, *, size=FS["label"], color=INK, ha="center", weight="normal", z=6, boxed=False, **kw):
    ""
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="baseline", fontweight=weight, zorder=z, **kw)
    return center_on_ink(t, horizontal=boxed, ref=None if boxed else "H")


def arrow(p, q, *, color, rad=0.0, dashed=False, a=3.4, b=3.4, lw=LW_DELIV, head=4.4, z=4, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, linewidth=lw,
                                 linestyle=(0, (2.4, 1.5)) if dashed else "-", shrinkA=a, shrinkB=b,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, joinstyle="miter", alpha=alpha))


def short_arrow(p, q, *, color, a, b, head_len, head_w, lw=LW_DELIV, z=4):
    ""
    d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
    ux, uy = (q[0] - p[0]) / d, (q[1] - p[1]) / d
    start, tip = (p[0] + a * ux, p[1] + a * uy), (q[0] - b * ux, q[1] - b * uy)
    base = (tip[0] - 0.8 * head_len * ux, tip[1] - 0.8 * head_len * uy)
    ax.plot([start[0], base[0]], [start[1], base[1]], color=color, linewidth=lw, solid_capstyle="butt", zorder=z)
    ax.add_patch(FancyArrowPatch(start, tip, arrowstyle=f"-|>,head_length={head_len},head_width={head_w / 2}",
                                 mutation_scale=1.0, color=color, linewidth=0.0, shrinkA=0, shrinkB=0, zorder=z))
    return d - a - b - head_len, head_len


def guides(x0, ys):
    ""
    for k in range(NS):
        ax.plot([sx(x0, k)] * 2, [min(ys) - 0.5 * PITCH + 1.0, max(ys) + 0.5 * PITCH - 1.0],
                color=S.C["gray3"], linewidth=S.LW["grid"], zorder=1)


def ticks(x0, y):
    for k in range(NS):
        text(sx(x0, k), y, str(k), size=FS["tick"], color=INK2)
    text(x0 - 3.0, y, "Step", size=FS["tick"], color=INK2, ha="right")


def window(x0, y, layer, color, amount=0.80):
    a, b = WIN[layer]
    rbox(x0 + a * STEP + 0.8, y - BAND / 2, (b - a + 1) * STEP - 1.6, BAND, fc=S.tint(color, amount),
         ec=S.tint(color, 0.35), lw=0.5, r=2.2)


def spikes(x0, y, layer, color):
    for t in SPIKES[layer]:
        ax.add_patch(Circle((sx(x0, t), y), DOT_R, facecolor=color, edgecolor="none", zorder=5))


def tile(x, y, label, color, *, idle=False):
    if idle:
        rbox(x, y - TILE / 2, TILE, TILE, fc="none", ec=S.C["gray2"], ls=(0, (2.0, 1.6)), r=2.0)
        return
    rbox(x, y - TILE / 2, TILE, TILE, fc=S.tint(color, 0.80), ec=color, r=2.0)
    text(x + TILE / 2, y, label, size=FS["label"], color=INK, boxed=True)


def chip(x, ys):
    ""
    rbox(x - CHIP_PAD, min(ys) - TILE / 2 - CHIP_PAD, TILE + 2 * CHIP_PAD, max(ys) - min(ys) + TILE + 2 * CHIP_PAD,
         fc="none", ec=S.C["gray2"], lw=0.6, r=2.8, z=1)


def stored(x0, y, t):
    ""
    rbox(sx(x0, t) - CELL / 2, y - CELL / 2, CELL, CELL, fc="white", ec=BLUE, lw=0.9, r=1.6, z=5)
    text(sx(x0, t), y, str(t), size=FS["label"], color=BLUE, weight="bold", z=6, boxed=True)


def deliveries(x0, lane_y, color, *, source_gap):
    ""
    for d in DELIV:
        if d["src"] not in lane_y or d["dst"] not in lane_y:
            continue
        p = (sx(x0, d["sent"]), lane_y[d["src"]])
        q = (sx(x0, d["at"]), lane_y[d["dst"]])
        gap = source_gap.get(d["src"], DOT_R + 0.9)
        if d["skip"]:
            rad = (p[1] - q[1]) / (2.0 * (q[0] - p[0])) * 0.94
            arrow(p, q, color=S.tint(color, 0.14), rad=rad, dashed=True, a=gap, b=DOT_R + 0.7, lw=0.75, z=3)
        else:
            arrow(p, q, color=color, a=gap, b=DOT_R + 0.5)


text(LEFT, ROW_TITLE_A, f"(a) Single pass, {word(N_REF)} cores", size=FS["title"], weight="bold", ha="left")

_schematic = S.marks(fig, "schematic").__enter__()
lane_a = dict(zip(LAYERS, Y_A))
guides(X2, Y_A)
for name in LAYERS:
    window(X2, lane_a[name], name, GRAY)
    spikes(X2, lane_a[name], name, INK2)
    tile(X2_TILE, lane_a[name], name, GRAY)
chip(X2_TILE, Y_A)
deliveries(X2, lane_a, GRAY, source_gap={})
ticks(X2, ROW_TICK_A)


top = LAYERS[0]
text(sx(X2, SPIKES[top][0]), ROW_TITLE_A, "Spike", size=FS["small"], color=INK2)
text(X2 + (WIN[top][1] + 1) * STEP + 2.2, lane_a[top], "Window", size=FS["small"], color=INK2, ha="left",
     bbox=dict(facecolor="white", edgecolor="none", pad=0.8))


NODE_R = 7.0
ny = 0.5 * (Y_A[0] + Y_A[-1])
nx_first, nx_last = LEFT + NODE_R, X2_TILE - CHIP_PAD - 30.0 - NODE_R
nx = [nx_first + (nx_last - nx_first) * i / (len(LAYERS) - 1) for i in range(len(LAYERS))]
for i, name in enumerate(LAYERS):
    ax.add_patch(Circle((nx[i], ny), NODE_R, facecolor="white", edgecolor=INK2, linewidth=S.LW["edge"], zorder=3))
    text(nx[i], ny, name, size=FS["label"], boxed=True)
pos = dict(zip(LAYERS, nx))
for a_, b_ in sorted({(d["src"], d["dst"]) for d in DELIV if not d["skip"]}):
    arrow((pos[a_], ny), (pos[b_], ny), color=INK2, a=NODE_R + 1.0, b=NODE_R + 1.0, lw=0.8, head=5.0)
for a_, b_ in sorted({(d["src"], d["dst"]) for d in DELIV if d["skip"]}):
    arrow((pos[a_], ny + NODE_R * 0.72), (pos[b_], ny + NODE_R * 0.72), color=INK2, rad=-0.2, dashed=True,
          a=NODE_R * 0.75, b=NODE_R * 0.75, lw=0.8, head=5.0)

arrow((nx[-1] + NODE_R + 5.0, ny), (X2_TILE - CHIP_PAD - 4.0, ny), color=S.C["gray2"], a=0, b=0, lw=1.0, head=6.2)


_schematic.__exit__(None, None, None)
text(LEFT, ROW_TITLE_B, f"(b) {word(len(PASSES), cap=True)} passes, {word(N_CHIP)} cores", size=FS["title"],
     weight="bold", ha="left")
_schematic.__enter__()

rbox(X1, Y_BUF - BUF_BAND / 2, X2 + BLOCK - X1, BUF_BAND, fc=S.tint(GRAY, 0.86), ec="none", r=2.4, z=1)

text(0.5 * (sx(X1, max(STORED)) + sx(X2, min(STORED))), Y_BUF, "Inter-pass buffer",
     color=INK2)


guides(X1, Y_CORES)
lane_1 = dict(zip(FIRST, Y_CORES))
for k, y in enumerate(Y_CORES):
    name = FIRST[k] if k < len(FIRST) else None
    tile(X1_TILE, y, name, BLUE, idle=name is None)
    if name:
        window(X1, y, name, BLUE)
        spikes(X1, y, name, BLUE)
for t in STORED:
    stored(X1, Y_BUF, t)

    shaft, head = short_arrow((sx(X1, t), lane_1[FIRST[0]]), (sx(X1, t), Y_BUF), color=BLUE, a=DOT_R,
                              b=CELL / 2 + 0.45, head_len=STORE_HEAD[0], head_w=STORE_HEAD[1])
    if shaft < 1.95 * head:
        print(f"FAIL {NAME}: a storage arrow's shaft is {shaft:.2f} pt behind a {head:.2f} pt head, under two heads")

for k, y in enumerate(Y_CORES):
    if k >= len(FIRST):
        text(X1 + 0.8, y, "Unused core", size=FS["small"], color=GRAY, ha="left",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
ticks(X1, ROW_TICK_B)


guides(X2, Y_CORES)
lane_2 = dict(zip(LATER, Y_CORES))
for name in LATER:
    tile(X2_TILE, lane_2[name], name, BLUE)
    window(X2, lane_2[name], name, BLUE)
    spikes(X2, lane_2[name], name, BLUE)
for t in STORED:
    stored(X2, Y_BUF, t)
lane_2r = dict(lane_2)
for name in FIRST:
    lane_2r[name] = Y_BUF
deliveries(X2, lane_2r, BLUE, source_gap={n: CELL / 2 + 0.9 for n in FIRST})
ticks(X2, ROW_TICK_B)
for x in (X1_TILE, X2_TILE):
    chip(x, Y_CORES)


y_write = 0.5 * (Y_CORES[0] + Y_CORES[-1])
x_chip = X2_TILE - CHIP_PAD
arrow((x_chip - 21.0, y_write), (x_chip - 1.6, y_write), color=ORANGE, a=0, b=0, lw=0.8, head=4.6)
text(x_chip - 11.3, y_write + 5.4, "Write", size=FS["small"], color=ORANGE)


x_gap0, x_gap1 = X1 + BLOCK + 3.0, X2 - 3.0
for x0, x1, label, color in ((X1, X1 + BLOCK, f"Pass {PASSES.index(FIRST) + 1}, storage", BLUE),
                             (x_gap0, x_gap1, "Weight write", ORANGE),
                             (X2, X2 + BLOCK, f"Pass {len(PASSES)}, replay", BLUE)):
    ax.plot([x0 + 1.0, x1 - 1.0], [ROW_BAR] * 2, color=color, linewidth=1.6, solid_capstyle="round", zorder=3)
    text(0.5 * (x0 + x1), ROW_PHASE, label, color=color)


_schematic.__exit__(None, None, None)
_count = S.marks(fig, "count", tag_at=(ax, X2 + BLOCK - 20.0, ROW_FOCAL, "data")).__enter__()
xb = X2 + BLOCK + 8.5
ya, yb = lane_a[OUT_LAYER], lane_2[OUT_LAYER]
ax.plot([X2 + BLOCK + 2.0, xb, xb, X2 + BLOCK + 2.0], [ya, ya, yb, yb], color=GREEN, linewidth=1.1,
        solid_joinstyle="round", solid_capstyle="round", zorder=3)
S.zero_mark(ax, xb, ROW_FOCAL, size=8.6)


_count_label = text(xb - 7.2, ROW_FOCAL, f"{COUNT['display']['display']} random multi-pass programs differing",
                    size=FS["focal"], weight="bold", color=BLUE, ha="right")


S.relabel(fig, "count", _count_label, f"Random multi-pass programs differing: {S.PENDING_TOKEN}", as_tag=True)

_count.__exit__(None, None, None)

ROWS = [
    {"what": "random multi-pass programs that differ from their single-pass programs (focal label, caption)",
     "value": COUNT["display"]["display"], "key": COUNT["display"]["key"],
     "source": f"{COUNT['differ']['source']}; {COUNT['programs']['source']}", "ref": "count"},
    {"what": "the timeline's schedule (layers, windows, spike steps, passes, stored steps, core counts): a "
             "schematic of the mechanism, not a measurement",
     "value": {"windows": WIN, "spikes": SPIKES, "passes": PASSES, "stored_steps": STORED,
               "cores_single_pass": N_REF, "cores_chip": N_CHIP},
     "source": f"{EX['source']} ({EX['status']}); mechanism: {EX['mechanism_record']}", "ref": "schematic"},
] + [r for r in S.caption_keys(NAME) if r["key"] != COUNT["display"]["key"]]
S.pending_key_at(fig, xb - 3.0, ROW_FOCAL - 12.0, ha="right")
S.save_fig(fig, NAME, rows=ROWS)
S.save_keys(NAME, ROWS)
