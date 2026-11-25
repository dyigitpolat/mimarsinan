""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.MIN_FONT_PT = 5.5
S.MIN_ARM_PT = S.JOIN_TEST_PT

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

NAME = "fig_capacity"
D = S.load(NAME)
S.setup()


FS = {"label": 6.5, "note": 6.0, "tick": 5.5, "small": 5.5, "beyond": 5.5}

W = S.COLUMN_PT
PAGE_MAX = 162.0 * 72.0 / 72.27
TITLE_A, TITLE_B = "Network weights over chip capacity", "Execution in passes on fewer cores"
TITLE_PAD_A, TITLE_PAD_B = 2.6, 2.2
NAME_GAP = 1.6
PANEL_GAP = 2.8


def ink_y(s, size, weight="normal"):
    ""
    bb = TextPath((0, 0), s, prop=FontProperties(family=S.FONT_FAMILIES[0], weight=weight, size=size)).get_extents()
    return bb.y1, bb.y0


def runs(flags):
    ""
    out, k = [], 0
    while k < len(flags):
        if flags[k]:
            e = k
            while e < len(flags) and flags[e]:
                e += 1
            out.append((k, e))
            k = e
        else:
            k += 1
    return out


def pt_axes(fig, y_top, h):
    ""
    H = fig.get_figheight() * 72.0
    ax = fig.add_axes([0.0, 1.0 - (y_top + h) / H, 1.0, h / H])
    ax.set_xlim(0, W)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


def count_label(n):
    ""
    m = n / 1e6
    return f"{m:.0f}M" if m >= 9.95 else f"{m:.1f}M"


def rbox(ax, x, y, w, h, *, fc, ec, lw=0.7, r=1.6, z=2, ls="-", clip=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=z, joinstyle="round", linestyle=ls, clip_on=clip))


nets, chips, ratio = D["networks"], D["chips"], D["ratio"]
n_r, n_c = len(nets), len(chips)
x_max = max(max(r) for r in ratio)
ORANGE, PALE = S.ROLE["writes"], S.C["gray3"]
LIGHT = 0.56
GRID = 0.5


def cell_color(x):
    if x <= 1.0:
        return PALE
    t = min(1.0, math.log10(x) / math.log10(x_max))
    return S.tint(ORANGE, LIGHT * (1.0 - t))


probe = plt.figure(figsize=S.figsize("column", 100), constrained_layout=False)
w_name = max(S.text_width_pt(probe, n["label"], FS["tick"]) for n in nets)
w_count = max(S.text_width_pt(probe, count_label(n["weights"]), FS["small"]) for n in nets)


def chip_style(c):
    ""
    return "italic" if c["kind"] == "design_study" else "normal"


def chip_name_w(c, text=None):
    ""
    ws = []
    for style in ("normal", chip_style(c)):
        t = probe.text(0, 0, c["label"] if text is None else text, fontsize=FS["tick"], fontstyle=style)
        ws.append(t.get_window_extent(probe.canvas.get_renderer()).width * 72.0 / probe.dpi)
        t.remove()
    return max(ws)


w_chip = [chip_name_w(c) for c in chips]

chip_split = {j: [(t, chip_name_w(c, t)) for t in c["label"].rsplit(" ", 1)] for j, c in enumerate(chips)
              if " " in c["label"]}
w_row_label = max(S.text_width_pt(probe, s, FS["label"]) for s in ("Single pass", "Multi-pass"))
_t = probe.text(0, 0, "Hg", fontsize=FS["tick"])
_r = probe.canvas.get_renderer()
LINE = _t.get_window_extent(_r).height * 72.0 / probe.dpi
_t.remove()
_t = probe.text(0, 0.5, f"(a) {TITLE_A} (b) {TITLE_B}", fontsize=S.FS["title"], fontweight="bold", va="baseline")
TITLE_ASC = (_t.get_window_extent(_r).y1 - 0.5 * probe.bbox.height) * 72.0 / probe.dpi

_t.remove()
_t = probe.text(0, 0.5, "Pass 1 Write", fontsize=FS["label"], va="top")
LABEL_LINE = _t.get_window_extent(_r).height * 72.0 / probe.dpi
_t.remove()
plt.close(probe)
TITLE_H_A, TITLE_H_B = TITLE_PAD_A + TITLE_ASC, TITLE_PAD_B + TITLE_ASC
ASC = ink_y("H", FS["tick"])[0]

RH = 6.8
X0 = w_name + 4.5 + w_count + 3.5
CW = (W - 0.5 - X0) / n_c
MAP_H = n_r * RH


col_x = [X0 + (j + 0.5) * CW for j in range(n_c)]
assert LINE + 3.0 <= CW, "a rotated chip name needs its column"

CORE, GAP, PAD = 6.6, 1.3, 1.7
ROW_H = CORE + 2 * PAD
BAND = 7.6
LAB_GAP = 1.8
HB = ROW_H + BAND + ROW_H + LAB_GAP + LABEL_LINE


NAME_LEAD = LINE / 1.2
NAME_ROOM = PAGE_MAX - (TITLE_H_A + MAP_H + NAME_GAP + PANEL_GAP + TITLE_H_B + HB + 0.1)
chip_lines = [[(c["label"], w_chip[j])] if w_chip[j] <= NAME_ROOM or j not in chip_split else chip_split[j]
              for j, c in enumerate(chips)]
assert all(len(ls) == 1 or LINE + NAME_LEAD + 3.0 <= CW for ls in chip_lines), "a two-line chip name needs its column"
h_chip = max(w for ls in chip_lines for _, w in ls)
HA = MAP_H + NAME_GAP + h_chip
print("chip names (pt):", ", ".join(" / ".join(f"{t} {w:.2f}" for t, w in ls) for ls in chip_lines),
      f"; room of the band {NAME_ROOM:.2f}")

H = TITLE_H_A + HA + PANEL_GAP + TITLE_H_B + HB + 0.1
print(f"geometry (pt): title rooms {TITLE_H_A:.2f} and {TITLE_H_B:.2f} (line over baseline {TITLE_ASC:.2f}), map "
      f"{MAP_H:.1f} ({n_r} rows of {RH}), chip names {h_chip:.2f} after {NAME_GAP}, panel gap {PANEL_GAP}, panel (b) "
      f"{HB:.2f} (label line {LABEL_LINE:.2f}), page {H:.2f} of at most {PAGE_MAX:.2f}; map from {X0:.2f}, column "
      f"pitch {CW:.2f}, name line {LINE:.2f}")
assert H <= PAGE_MAX, f"page {H:.2f} pt over the pass's limit {PAGE_MAX:.2f}"
fig = plt.figure(figsize=S.figsize("column", H), constrained_layout=False)


a = pt_axes(fig, TITLE_H_A, HA)
S.panel_title(a, "a", "Network weights over chip capacity", pad=TITLE_PAD_A, y=1.0)
wide = [sum(x > 1.0 for x in r) for r in ratio]


FOCAL_ROWS = (1, 3)
WITHIN_ROWS = (n_r - 3, n_r - 1)
focal_cols = (1, min(wide[FOCAL_ROWS[0]:FOCAL_ROWS[1]]))
within_cols = (max(wide[WITHIN_ROWS[0]:WITHIN_ROWS[1]]), n_c)
focal_text = next(t for t in (f"{D['focal']['display']} pairs over capacity", f"{D['focal']['display']} over capacity")
                  if S.text_width_pt(fig, t, S.FS["focal"], "bold") + 4.0 <= (focal_cols[1] - focal_cols[0]) * CW)
within_text = "Within capacity"
assert S.text_width_pt(fig, within_text, FS["note"]) + 4.0 <= (within_cols[1] - within_cols[0]) * CW
plates = [(*FOCAL_ROWS, *focal_cols), (*WITHIN_ROWS, *within_cols)]


def joined(i1, j1, i2, j2):
    ""
    return any(r0 <= i1 < r1 and r0 <= i2 < r1 and c0 <= j1 < c1 and c0 <= j2 < c1 for r0, r1, c0, c1 in plates)


for i, n in enumerate(nets):
    yc = (i + 0.5) * RH
    a.text(0.0, yc, n["label"], ha="left", va="center", fontsize=FS["tick"], color=S.C["ink"])
    a.text(X0 - 3.5, yc, count_label(n["weights"]), ha="right", va="center", fontsize=FS["small"],
           color=S.C["ink2"])
    for j in range(n_c):
        fc = cell_color(ratio[i][j])
        a.add_patch(Rectangle((X0 + j * CW, i * RH), CW, RH, facecolor=fc, edgecolor=fc, linewidth=0.2, zorder=2))
for j, c in enumerate(chips):

    two = dict(multialignment="right", linespacing=1.0) if len(chip_lines[j]) > 1 else {}
    a.text(col_x[j], MAP_H + NAME_GAP, "\n".join(t for t, _ in chip_lines[j]), rotation=90, ha="center", va="top",
           fontsize=FS["tick"], color=S.C["ink"], fontstyle=chip_style(c), **two)


gk = dict(color=S.C["white"], linewidth=GRID, solid_capstyle="butt", zorder=3)
for i in range(1, n_r):
    for j0, j1 in runs([not joined(i - 1, j, i, j) for j in range(n_c)]):
        a.plot([X0 + j0 * CW, X0 + j1 * CW], [i * RH, i * RH], **gk)
for j in range(1, n_c):
    for i0, i1 in runs([not joined(i, j - 1, i, j) for i in range(n_r)]):
        a.plot([X0 + j * CW, X0 + j * CW], [i0 * RH, i1 * RH], **gk)


def plate_label(text, rows, cols, size, weight, color, **kw):
    ""
    top, bottom = ink_y(text, size, weight)
    cap = ink_y("H", size, weight)[0]
    h = (rows[1] - rows[0]) * RH
    mid = 0.5 * cap if h - cap >= 2.0 * (1.5 - bottom) else 0.5 * (top + bottom)
    return dict(xy=(X0 + 0.5 * (cols[0] + cols[1]) * CW, 0.5 * (rows[0] + rows[1]) * RH + mid), ha="center",
                va="baseline", color=color, zorder=6, **kw)


k = plate_label(focal_text, FOCAL_ROWS, focal_cols, S.FS["focal"], "bold", S.C["ink"])
with S.marks(fig, "focal", kind="label"):
    S.focal(a, focal_text, k.pop("xy"), **k)

k = plate_label(within_text, WITHIN_ROWS, within_cols, FS["note"], "normal", S.C["ink2"])
a.text(*k.pop("xy"), within_text, fontsize=FS["note"], **k)


edge = [(X0 + wide[0] * CW, 0.0)]
for i in range(n_r):
    xe = X0 + wide[i] * CW
    if xe != edge[-1][0]:
        edge.append((xe, i * RH))
    edge.append((xe, (i + 1) * RH))
a.plot([p[0] for p in edge], [p[1] for p in edge], color=S.C["ink"], linewidth=0.9, solid_joinstyle="miter",
       solid_capstyle="butt", zorder=5)


KEY_NAME = ("Network weights", "per chip weight cell")
RAMP_W, KEY_LINE = 59.41, LINE / 1.2
ticks = [10 ** k for k in range(0, int(math.floor(math.log10(x_max))) + 1)]
RAMP_H, KEY_LEAD, NOTCH = 5.0, 2.4, 1.0
rx0 = 0.4
rw = RAMP_W
rx1 = rx0 + max(rw, *(S.text_width_pt(fig, t, FS["small"]) for t in KEY_NAME))
assert rx1 + 3.0 <= col_x[0] - 0.5 * LINE, "the colour key runs into the first chip name"
base = MAP_H + NAME_GAP + h_chip
ry = base - ASC - NOTCH - 1.2 - RAMP_H

a.text(rx0, ry - KEY_LEAD, "\n".join(KEY_NAME), ha="left", va="baseline", fontsize=FS["small"], color=S.C["ink"],
       linespacing=1.0)
key_top = ry - KEY_LEAD - (len(KEY_NAME) - 1) * KEY_LINE - ink_y(KEY_NAME[0], FS["small"])[0]
assert key_top - MAP_H >= 4.0, f"the colour key's name stands {key_top - MAP_H:.1f} pt under the map (least 4)"
print(f"colour key (pt): ramp {rw:.2f} wide, name lines {KEY_LINE:.2f} apart, its ink {key_top - MAP_H:.2f} under "
      f"the map's foot")
STEPS = 48
for k in range(STEPS):
    x = x_max ** ((k + 0.5) / STEPS)
    fc = cell_color(max(x, 1.0001))
    a.add_patch(Rectangle((rx0 + rw * k / STEPS, ry), rw / STEPS, RAMP_H, facecolor=fc, edgecolor=fc, linewidth=0.2))
prev = None
for t in ticks:
    tx = rx0 + rw * math.log10(t) / math.log10(x_max)
    a.plot([tx, tx], [ry + RAMP_H, ry + RAMP_H + NOTCH], color=S.C["ink2"], linewidth=S.LW["axis"],
           solid_capstyle="butt")
    lab = S.thousands(t)
    half = 0.5 * S.text_width_pt(fig, lab, FS["small"])
    cx = max(tx, rx0 + half)
    assert cx + half + 3.0 <= col_x[0] - 0.5 * LINE, f"key value {lab} runs into the first chip name"
    assert prev is None or cx - half - prev >= 2.0, f"key value {lab} too close to the one before it"
    prev = cx + half
    a.text(cx, base, lab, ha="center", va="baseline", fontsize=FS["small"], color=S.C["ink"], zorder=4)


b = pt_axes(fig, TITLE_H_A + HA + PANEL_GAP + TITLE_H_B, HB)
S.panel_title(b, "b", "Execution in passes on fewer cores", pad=TITLE_PAD_B, y=1.0)
BLUE, FRAME = S.ROLE["cores"], S.C["ink2"]
LOADS = [S.tint(BLUE, 0.0), S.tint(BLUE, 0.45), S.tint(BLUE, 0.78)]
PER = 3
y1, y2 = 0.3, 0.3 + ROW_H + BAND
yc1, yc2 = y1 + ROW_H / 2, y2 + ROW_H / 2
xl = w_row_label
xs = xl + 5.0
b.text(xl, yc1, "Single pass", ha="right", va="center", fontsize=FS["label"], color=S.C["ink"])
b.text(xl, yc2, "Multi-pass", ha="right", va="center", fontsize=FS["label"], color=S.C["ink"])


def chip(ax, x, y, loads, beyond=False):
    ""
    w = len(loads) * CORE + (len(loads) - 1) * GAP + 2 * PAD
    rbox(ax, x, y, w, ROW_H, fc=S.C["white"], ec=S.C["gray"] if beyond else FRAME, lw=0.7, r=2.0, z=2,
         ls=(0, (2.4, 1.6)) if beyond else "-")
    for k, fc in enumerate(loads):
        rbox(ax, x + PAD + k * (CORE + GAP), y + PAD, CORE, CORE, fc=fc, ec=BLUE, lw=0.6, r=1.2, z=3)
    return x + w


x1e = chip(b, xs, y1, [LOADS[0]] * PER)
x_beyond = x1e + 2.0
x1e = chip(b, x_beyond, y1, [LOADS[k // PER] for k in range(PER, PER * len(LOADS))], beyond=True)

_top, _bottom = ink_y("Beyond the chip", FS["beyond"])
assert BAND - (_top - _bottom) >= 2.0, "the band between the rows is too low for its name"
b.text((x_beyond + x1e) / 2, y1 + ROW_H + BAND / 2 + (_top + _bottom) / 2, "Beyond the chip", ha="center",
       va="baseline", fontsize=FS["beyond"], color=S.C["ink2"])
unit = PER * CORE + (PER - 1) * GAP + 2 * PAD
MARK_WORDS = ("Equal", "outputs")
x_mark = W - 7.0 - max(S.text_width_pt(fig, w, FS["label"]) for w in MARK_WORDS) - 1.0


MARK = 7.0
x_end = x_mark - MARK / 2 - S.JOIN_PAD
x_join = x_end - S.JOIN_ARM
JOIN_TAP = 8.0
STEP = (x_join - JOIN_TAP - xs - unit) / (len(LOADS) - 1)
ylab = y2 + ROW_H + LAB_GAP
for p, load in enumerate(LOADS):
    x = xs + p * STEP
    x2e = chip(b, x, y2, [load] * PER)
    b.text(x + unit / 2, ylab, f"Pass {p + 1}", ha="center", va="top", fontsize=FS["label"], color=S.C["ink"])
    if p + 1 < len(LOADS):
        b.add_patch(FancyArrowPatch((x2e + 2.5, yc2), (x + STEP - 2.5, yc2), arrowstyle="-|>", mutation_scale=6.4,
                                    color=ORANGE, linewidth=1.5, shrinkA=0, shrinkB=0, zorder=4))
        b.text((x2e + x + STEP) / 2, ylab, "Write", ha="center", va="top", fontsize=FS["label"], color=S.C["ink"])

ym = (yc1 + yc2) / 2
S.join_arrow(b, [(x1e + 2.5, yc1), (x2e + 2.5, yc2)], x_join, x_end, ym, color=S.C["ink2"])
S.zero_mark(b, x_mark, ym, size=MARK)
b.text(x_mark + 6.0, ym, "\n".join(MARK_WORDS), ha="left", va="center", fontsize=FS["label"], color=S.C["ink"],
       linespacing=1.05)
print(f"join (pt): upright line {x_join:.2f}, arm's end {x_end:.2f}, mark's edge {x_mark - MARK / 2:.2f}, pass pitch "
      f"{STEP:.2f}, write arrow {STEP - unit - 5.0:.2f}")


TILE_NAME = "One core's weight tile"
TILE, TILE_GAP = 5.2, 2.2
yb = -TITLE_PAD_B
xt = W - 0.5 - S.text_width_pt(fig, TILE_NAME, FS["small"])
x_title = S.text_width_pt(fig, f"(b) {TITLE_B}", S.FS["title"], "bold")
assert xt - TILE_GAP - TILE >= x_title + 12.0, "the key entry of (b) runs into the panel's title"
b.text(xt, yb, TILE_NAME, ha="left", va="baseline", fontsize=FS["small"], color=S.C["ink2"])
rbox(b, xt - TILE_GAP - TILE, yb - ink_y("H", FS["small"])[0] / 2 - TILE / 2, TILE, TILE, fc=S.C["gray3"],
     ec=S.C["gray"], lw=0.6, r=1.2 * TILE / CORE, z=3, clip=False)
print(f"key entry of (b) (pt): sample from {xt - TILE_GAP - TILE:.2f}, words from {xt:.2f}, the title's end {x_title:.2f}")


rows = [{"what": "pairs exceeding capacity (focal label, panel a)", "value": D["focal"]["display"],
         "key": D["focal"]["key"], "class": D["focal"]["class"], "form": D["focal"]["form"], "ref": "focal"}]
for n in nets:
    rows.append({"what": f"weight count, {n['label']} (row label and row order)", "value": n["weights"],
                 "key": D["network_key"], "source": n["source"]})
for c in chips:
    rows.append({"what": f"capacity in weight cells, {c['label']} (column order)", "value": c["capacity_cells"],
                 "key": D["chip_key"], "source": c["source"]})
for c in chips:
    rows.append({"what": f"kind of the column {c['label']} (a design study's name in italics)", "value": c["kind"],
                 "source": f"{D['chip_kind']['source']} Work: {c['kind_work']}, {c['kind_basis']}"})
for i, n in enumerate(nets):
    for j, c in enumerate(chips):
        rows.append({"what": f"weights per weight cell, {n['label']} on {c['label']} (cell colour)",
                     "value": ratio[i][j], "source": D["ratio_source"][i][j]})
S.pending_key_at(fig, X0 + 0.5 * (within_cols[0] + n_c) * CW, H - (TITLE_H_A + (n_r - 0.5) * RH), ha="center")
S.save_fig(fig, NAME, rows=rows)
S.save_keys(NAME, rows)
