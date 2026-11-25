""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import text_to_path
from sciviz.inkcenter import center_on_ink

NAME = "fig_invariants"


S.MIN_FONT_PT = 5.5
FS_FLOOR = 6.0


S.INK_CLEAR_PT = 1.0
S.FS["title"] = 7.5
S.setup()

STATEMENT = ("the statement of Theorem 1 by four invariants (Section IV); condition ids as in Table II")

INVARIANTS = [
    {"tag": "I1", "name": ["Stored values", "and latency"], "ids": [r"H1$_{\theta}$, H1$_{L}$, H10"],
     "ids_plain": "H1theta, H1L, H10", "check": "static and run"},
    {"tag": "I2", "name": ["Delivery"], "ids": [r"H5$_{c}$, H5$_{s}$,", "R3 to R5"],
     "ids_plain": "H5c, H5s (R1 and R2), R3 to R5", "check": "static and run"},
    {"tag": "I3", "name": ["Exact integer", "sum"], "ids": ["H6, H8"],
     "ids_plain": "H6, H8", "check": "static and run"},
    {"tag": "I4", "name": ["Update,", "event order, host"], "ids": ["H0, H7, H9, R6, Host"],
     "ids_plain": "H0, H7, H9, R6, Host", "check": "static and run"},
]


PRINTED = {"I4": ["Update", "function"]}
KEY = {"static": "Static check", "static and run": "Static and run checks"}
KEY_BUFFER = "Inter-pass buffer"
HEAD_NOTE = None


M_A = r"$\mathit{a}_{\mathit{mn}}(\mathit{t})$"
M_Q = r"$\mathit{q}_{\mathit{n}}(\mathit{t})$"
M_O = {"iota": r"$\mathit{o}^{\iota}_{\mathit{n}}(\mathit{t})$", "delta": r"$\mathit{o}^{\delta}_{\mathit{n}}(\mathit{t})$"}
M_U = r"$\mathit{u}_{\mathit{n}}$"
M_F = r"$\mathit{F}_{\sigma}$"
M_TH = r"$\vartheta_{\mathit{n}}$"
M_L = r"$\ell_{\mathit{n}}$"
M_M1 = r"$\mathit{m}_{1}$"
M_M2 = r"$\mathit{m}_{2}$"
M_SUM = r"$\Sigma$"


W = S.COLUMN_PT
FS_NAME, FS_IDS, FS_TAG, FS_MATH, FS_NOTE = 6.5, 6.0, 7.5, 6.5, 6.0
FS_SUB, SUB_DROP = 5.5, 1.3
SUB_KERN = (0.15, 0.2)
NOTE_AIR = 2.0
X_I = [8.0, 59.0, 103.5, 157.5]
X_PROD = 27.0
X_KNEE = 84.0
X_BR = 245.5
TRACK = 6.5
R_NODE = 5.6
R_SUM = 6.2
BOX_W, BOX_H = 26.0, 13.0
TILE_PAD = 10.5
CELL_HW = 6.5
LEAD = 7.0
GAP_TAG, GAP_IDS = 9.8, 8.1
STACK = GAP_TAG + LEAD + GAP_IDS
assert all(len(inv["name"]) + len(inv["ids"]) <= 3 for inv in INVARIANTS), "a label of the middle band over 3 lines"

ROW_FOOT = 7.3
Y_RAIL_B = 16.6
Y_NOTE_B = 23.2
Y_B = 39.5
BAND0 = Y_B + 13.0
BAND1 = BAND0 + 5.4 + STACK + 6.5
Y_A = BAND1 + 13.4
Y_RAIL_A = Y_A + 15.0
ROW_HEAD = Y_RAIL_A + 9.2
H = ROW_HEAD + 4.6

BLUE, GRAY, GRAY2, GRAY3 = S.C["blue"], S.C["gray"], S.C["gray2"], S.C["gray3"]
INK, INK2 = S.C["ink"], S.C["ink2"]

fig = plt.figure(figsize=S.figsize("column", H))
fig.set_layout_engine("none")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def rbox(x, y, w, h, *, fc, ec, lw=S.LW["edge"], r=2.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))


def text(x, y, s, *, size=FS_NAME, color=INK, ha="center", va="center_baseline", weight="normal", z=6,
         boxed=False, **kw):
    ""
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)
    return center_on_ink(t) if boxed else t


def letter(title, symbol):
    ""
    bb = title.get_window_extent(fig.canvas.get_renderer())
    x1 = bb.x1 * 72.0 / fig.dpi
    return text(x1 + 3.0, title.get_position()[1], symbol, size=title.get_fontsize() + 0.5, ha="left")


def lines(x, y, rows, *, size, pitch=LEAD, **kw):
    ""
    t = text(x, y, "\n".join(rows), size=size, va="center", **kw)
    r = fig.canvas.get_renderer()
    one = text(x, y, rows[0], size=size, va="center", **kw)
    h1 = one.get_window_extent(r).height * 72.0 / fig.dpi
    one.remove()
    t.set_linespacing(pitch / h1)
    t.set_y(y - 0.5 * pitch * (len(rows) - 1))
    return t


_SUB = re.compile(r"\$_\{(.+?)\}\$")


def metrics(s, size, ismath=False):
    ""
    return text_to_path.get_text_width_height_descent(s, FontProperties(size=size), ismath)


def sub_lines(x, y, rows, *, size, sub_size=FS_SUB, drop=SUB_DROP, pitch=LEAD, color=INK):
    ""
    cap = metrics("H", size)[1]
    for i, row in enumerate(rows):
        xr, yb = x, y - i * pitch - cap / 2.0
        for j, part in enumerate(_SUB.split(row)):
            if not part:
                continue
            sub = j % 2 == 1
            s_, fs = (f"${part}$", sub_size) if sub else (part, size)
            xr += SUB_KERN[0] if sub else 0.0
            ax.text(xr, yb - (drop if sub else 0.0), s_, fontsize=fs, color=color, ha="left", va="baseline", zorder=6)
            xr += metrics(s_, fs, sub)[0] + (SUB_KERN[1] if sub else 0.0)


def arrow(p, q, *, color=INK2, rad=0.0, dashed=False, a=0.0, b=0.0, lw=0.8, head=4.6, z=4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=head, color=color, linewidth=lw,
                                 linestyle=(0, (2.4, 1.5)) if dashed else "-", shrinkA=a, shrinkB=b,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, joinstyle="miter"))


def line(xs, ys, *, color=INK2, lw=0.8, z=3, **kw):
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=z, solid_capstyle="round", **kw)


def node(x, y, r, label, *, size=FS_MATH, ec=INK2, fc="white"):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=S.LW["edge"], zorder=5))
    text(x, y, label, size=size, z=6, boxed=True)


def tile(x0, x1, y, *, half=TILE_PAD):
    ""
    rbox(x0, y - half, x1 - x0, 2 * half, fc=S.tint(GRAY, 0.88), ec=GRAY2, r=2.6, z=1)


def stored_values(y0, y1):
    ""
    x = X_I[0]
    rbox(x - CELL_HW, y0, 2 * CELL_HW, y1 - y0, fc=S.tint(GRAY, 0.88), ec=GRAY, r=2.2, z=2)
    ym = 0.5 * (y0 + y1)
    text(x, ym + 5.0, M_TH, size=FS_MATH + 0.5, boxed=True)
    text(x, ym - 5.0, M_L, size=FS_MATH + 0.5, boxed=True)


def update_box(y):
    ""
    x = X_I[3]
    rbox(x - BOX_W / 2, y - BOX_H / 2, BOX_W, BOX_H, fc="white", ec=INK2, r=2.2, z=5)
    text(x - 6.0, y, M_F, size=FS_MATH + 0.5, boxed=True)
    rbox(x + 1.0, y - 4.6, 9.6, 9.2, fc=S.tint(GRAY, 0.86), ec=GRAY, lw=0.5, r=1.4, z=6)
    text(x + 5.8, y, M_U, size=FS_NAME - 0.5, z=7, boxed=True)


def buffer_cell(x, y, half=5.0):
    ""
    rbox(x - half, y - half, 2 * half, 2 * half, fc="white", ec=INK2, lw=0.9, r=0.36 * half, z=5)
    ax.add_patch(Circle((x, y), 0.4 * half, facecolor=INK2, edgecolor="none", zorder=6))


def delivery_bar(y):
    line([X_I[1]] * 2, [y - TRACK - 3.0, y + TRACK + 3.0], color=INK2, lw=1.5, z=5)


def output(y, program):
    x0 = X_I[3] + BOX_W / 2
    arrow((x0, y), (X_BR - 6.5, y))
    text(0.5 * (x0 + X_BR - 6.5) + 6.0, y + 6.0, M_O[program], size=FS_MATH + 0.5)


t_a = text(1.0, ROW_HEAD, "Single-pass program", size=S.FS["title"], weight="bold", ha="left")
letter(t_a, r"$\iota$")
if HEAD_NOTE:
    text(W - 1.0, ROW_HEAD, HEAD_NOTE, size=FS_NAME, color=INK2, ha="right")

stored_values(Y_A - 12.5, Y_RAIL_A + 2.5)
line([X_I[0] + CELL_HW, X_I[3]], [Y_RAIL_A] * 2, color=GRAY, lw=0.6, z=2)
for x, top in ((X_I[2], Y_A + R_SUM), (X_I[3], Y_A + BOX_H / 2)):
    arrow((x, Y_RAIL_A), (x, top + 0.6), color=GRAY, lw=0.6, head=4.0, z=3)

tile(X_I[2] - R_SUM - 4.0, X_I[3] + BOX_W / 2 + 4.0, Y_A)
for label, dy in ((M_M1, TRACK), (M_M2, -TRACK)):
    node(X_PROD, Y_A + dy, R_NODE, label)
    line([X_PROD + R_NODE, X_KNEE], [Y_A + dy] * 2)
    arrow((X_KNEE, Y_A + dy), (X_I[2], Y_A + dy * 0.25), b=R_SUM + 0.4, head=4.0)
delivery_bar(Y_A)
text(X_I[1] + 3.0, Y_A, M_A, ha="left", size=FS_MATH)
node(X_I[2], Y_A, R_SUM, M_SUM, size=7.5)
arrow((X_I[2] + R_SUM, Y_A), (X_I[3] - BOX_W / 2, Y_A))
xm = 0.5 * (X_I[2] + R_SUM + X_I[3] - BOX_W / 2)
text(xm, Y_A + 5.4, M_Q, size=FS_MATH)

x_note = min(xm, X_I[3] - BOX_W / 2 - 0.35 - NOTE_AIR - metrics("Single core", FS_NOTE)[0] / 2)
text(x_note, Y_A - 5.6, "Single core", size=FS_NOTE, color=INK2)
update_box(Y_A)
output(Y_A, "iota")


t_b = text(1.0, ROW_FOOT, "Deployment", size=S.FS["title"], weight="bold", ha="left")
letter(t_b, r"$\delta$")

stored_values(Y_RAIL_B - 2.5, Y_B + 12.5)
x_feed = (86.0, X_I[3])
line([X_I[0] + CELL_HW, x_feed[1]], [Y_RAIL_B] * 2, color=GRAY, lw=0.6, z=2)
arrow((x_feed[0], Y_RAIL_B), (x_feed[0], Y_B - TRACK - 5.4), color=GRAY, lw=0.6, head=4.0, z=3)
arrow((x_feed[1], Y_RAIL_B), (x_feed[1], Y_B - BOX_H / 2 - 0.6), color=GRAY, lw=0.6, head=4.0, z=3)

arrow((44.0, Y_RAIL_B), (72.0, Y_RAIL_B), color=GRAY, lw=0.6, head=4.0, z=3)
text(57.0, Y_RAIL_B, "Write", size=FS_NOTE, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=1.8))


node(X_PROD, Y_B + TRACK, R_NODE, M_M2)
line([X_PROD + R_NODE, X_I[1]], [Y_B + TRACK] * 2)

T = 11.0
rbox(19.0, Y_B - TRACK - T / 2, T + 1.0, T, fc=S.tint(GRAY, 0.80), ec=GRAY, r=2.2, z=3)
text(19.0 + (T + 1.0) / 2, Y_B - TRACK, M_M1, size=FS_MATH, boxed=True)
arrow((31.0, Y_B - TRACK), (38.0, Y_B - TRACK), head=3.8, a=0.4, b=0.6)
buffer_cell(43.0, Y_B - TRACK)
line([48.0, X_I[1]], [Y_B - TRACK] * 2)
text(33.5, Y_NOTE_B, "Earlier pass", size=FS_NOTE, color=INK2)

delivery_bar(Y_B)

X_PT = (80.0, 92.0)
for dy in (TRACK, -TRACK):
    arrow((X_I[1], Y_B + dy), (X_PT[0], Y_B + dy), b=0.3)
    rbox(X_PT[0], Y_B + dy - 5.4, X_PT[1] - X_PT[0], 10.8, fc=S.tint(GRAY, 0.88), ec=GRAY2, r=2.0, z=3)
    text(0.5 * sum(X_PT), Y_B + dy, M_SUM, size=7.0, boxed=True)
    arrow((X_PT[1], Y_B + dy), (X_I[2], Y_B), a=0.3, b=4.9, head=3.8)
text(X_I[1] + 3.0, Y_B, M_A, ha="left", size=FS_MATH)
tile(X_I[2] - 8.0, X_I[3] + BOX_W / 2 + 4.0, Y_B)
node(X_I[2], Y_B, 4.6, "+", size=7.5)
arrow((X_I[2] + 4.6, Y_B), (X_I[3] - BOX_W / 2, Y_B))
xm = 0.5 * (X_I[2] + 4.6 + X_I[3] - BOX_W / 2)
text(xm, Y_B + 5.4, M_Q, size=FS_MATH)
text(xm, Y_B - 5.6, "Split fan-in", size=FS_NOTE, color=INK2)
update_box(Y_B)
output(Y_B, "delta")


Y_TAG = BAND1 - 6.5
ENDS = [(Y_A - 12.5, Y_B + 12.5),
        (Y_A - TRACK - 3.0, Y_B + TRACK + 3.0),
        (Y_A - R_SUM, Y_B + 4.6),
        (Y_A - BOX_H / 2, Y_B + BOX_H / 2)]


def badge(x, y, check, r=4.0):
    filled = check == "static and run"
    ax.add_patch(Circle((x, y), r, facecolor=BLUE if filled else "white", edgecolor=BLUE, linewidth=0.9, zorder=7))
    for dy in (-1.15, 1.15):
        line([x - 1.9, x + 1.9], [y + dy] * 2, color="white" if filled else BLUE, lw=0.95, z=8)


for inv, x, (ya, yb) in zip(INVARIANTS, X_I, ENDS):
    line([x, x], [yb, ya], color=BLUE, lw=1.0, z=4)
    for y in (ya, yb):
        ax.add_patch(Circle((x, y), 1.5, facecolor=BLUE, edgecolor="none", zorder=7))
    badge(x, Y_TAG, inv["check"])
    text(x + 6.5, Y_TAG, inv["tag"], size=FS_TAG, color=BLUE, weight="bold", ha="left")
    y = Y_TAG - GAP_TAG
    name = PRINTED.get(inv["tag"], inv["name"])
    lines(x + 4.0, y, name, size=FS_NAME, ha="left")
    y -= LEAD * (len(name) - 1) + GAP_IDS
    sub_lines(x + 4.0, y, inv["ids"], size=FS_IDS, color=GRAY)


line([X_BR - 5.0, X_BR, X_BR, X_BR - 5.0], [Y_A, Y_A, Y_B, Y_B], color=BLUE, lw=1.3, z=4, solid_joinstyle="round")

text(X_BR - 4.5, Y_TAG, "Equal outputs", size=S.FS["focal"] - 0.5, weight="bold", color=BLUE, ha="right")


xk = W - 1.0
for check in [c for c in ("static and run", "static") if any(inv["check"] == c for inv in INVARIANTS)]:
    xk -= S.text_width_pt(fig, KEY[check], FS_NOTE)
    text(xk, ROW_FOOT, KEY[check], size=FS_NOTE, color=INK2, ha="left")
    xk -= 6.0
    badge(xk, ROW_FOOT + 0.3, check, r=3.4)
    xk -= 10.0
xk -= S.text_width_pt(fig, KEY_BUFFER, FS_NOTE)
text(xk, ROW_FOOT, KEY_BUFFER, size=FS_NOTE, color=INK2, ha="left")
buffer_cell(xk - 6.0, ROW_FOOT + 0.3, half=3.4)

ROWS = [
    {"what": f"{inv['tag']} {' '.join(inv['name'])}: condition ids and check (identifiers of the theorem's "
             "statement, not measurements)" + (f"; printed as {' '.join(PRINTED[inv['tag']])}"
                                               if inv["tag"] in PRINTED else ""),
     "value": {"conditions": inv["ids_plain"], "check": inv["check"]}, "source": STATEMENT}
    for inv in INVARIANTS
] + [
    {"what": "the two lanes (producers, carry buffer, weight write, split cores): a schematic of one induction "
             "step of the proof, not a measurement",
     "value": "schematic", "source": STATEMENT},
] + S.caption_keys(NAME)
for t in ax.texts:
    if t.get_fontsize() < FS_FLOOR - 0.01 and abs(t.get_fontsize() - FS_SUB) > 0.01:
        print(f"FAIL {NAME}: text {t.get_text()!r} at {t.get_fontsize():.1f} pt, under the {FS_FLOOR} pt floor")
S.save_fig(fig, NAME, rows=ROWS)
S.save_keys(NAME, ROWS)
