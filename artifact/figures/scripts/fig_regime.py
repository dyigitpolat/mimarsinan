""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S

S.FS["title"] = 7.5

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

NAME = "fig_regime"
D = S.load(NAME)

HEIGHT_PT = 108.0
FOCAL = "informative"


S.MIN_FONT_PT = 6.0
FS_TICK, FS_KEY, FS_GLOSS = 6.5, 6.0, 6.0
FS_NAME = 7.0
FS_AXIS = 6.5


PRINTED = {"Single pass, no rewrites": "Stationary tiles"}
Y_NAME = "Write-to-compute ratio $\\Pi/K$"
X_NAME = "Tiles per core"
ROLE = {
    "write_dominated": S.ROLE["write_dominated"],
    "informative": S.ROLE["ours"],
    "capacity_free": S.ROLE["series2"],
}
DOT = {
    "straddling": (S.C["gray2"], 0.55),
    "write_dominated": (S.tint(ROLE["write_dominated"], 0.15), 0.42),
    "capacity_free": (ROLE["capacity_free"], 0.55),
    "informative": (ROLE["informative"], 0.62),
}
DRAW_ORDER = ("straddling", "write_dominated", "capacity_free", "informative")
DOT_PT = 1.15
DIAMOND = dict(marker="D", markersize=2.7, markeredgewidth=0.55, markerfacecolor="none",
               markeredgecolor=S.C["gray"])


GLOSS = {
    "write_dominated": "Cost set by\nweight writes",
    "informative": "Cost shared by\nwrites and compute",


    "capacity_free": "System cost model:\nwrite cost at most {tol}",
}


INK_UP, INK_DOWN, INK_CAP, INK_MID = 0.76, 0.24, 0.714, 0.27

KEY_PITCH = 1.32


NAME_TO_GLOSS, GLOSS_PITCH = 8.25, 7.5
COLUMN_GAP_MIN, LEADER_STEP_MIN, TITLE_PAD = 4.5, 4.0, 2.0


def shade(color: str, amount: float) -> str:
    ""
    a, b = color.lstrip("#"), S.C["ink"].lstrip("#")
    return "#" + "".join(f"{round(int(a[i:i + 2], 16) * (1 - amount) + int(b[i:i + 2], 16) * amount):02x}"
                         for i in (0, 2, 4))


def decades(lo: float, hi: float, step: int = 1, phase: int = 0):
    return [10.0 ** k for k in range(math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9) + 1)
            if k % step == phase]


def pow10(v, pos=None):
    k = round(math.log10(v))
    return {0: "1", 1: "10"}.get(k, rf"$10^{{{k}}}$")


def main():
    S.setup()
    W = S.width_pt("column")
    fig = plt.figure(figsize=S.figsize("column", HEIGHT_PT), layout="none")
    fig.canvas.draw()

    lo_b = float(D["bounds"]["capacity_free_max"]["value"])
    hi_b = float(D["bounds"]["write_dominated_min"]["value"])
    tol = D["tolerance"]["display"]
    axes_d = D["axes"]
    order = list(axes_d)
    y_min = min(a["pi_k_min"] for a in axes_d.values())
    y_max = max(a["pi_k_max"] for a in axes_d.values())
    x_min = min(a["x_min"] for a in axes_d.values())
    x_max = max(a["x_max"] for a in axes_d.values())
    y_lo, y_hi = y_min / 2.5, y_max * 3.0
    x_lo, x_hi = x_min / 10.0 ** 0.12, x_max * 1.3

    probe = [{"what": "explored points", "source": s["source"]} for a in axes_d.values()
             for s in a["series"].values()][:1]
    y_lo, y_hi = S.frame_range(y_lo, y_hi, probe, scale="log")


    x_hi = S.frame_range(x_lo, x_hi, probe, scale="log")[1]
    x_ticks = decades(1.0, x_hi)
    if S.withheld(probe):
        x_ticks = [v for v in x_ticks if v < x_hi / 1.001]


    names = {r: PRINTED.get(s["label"], s["label"]) for a in axes_d.values() for r, s in a["series"].items()}
    gloss = {r: t.format(tol=tol).split("\n") for r, t in GLOSS.items()}
    key = [("straddling", names["straddling"]), ("no_weight_writes", names["no_weight_writes"])]
    gut_w = max([S.text_width_pt(fig, names[FOCAL], S.FS["focal"], "bold")] +
                [S.text_width_pt(fig, names[r], FS_NAME) for r in ROLE if r != FOCAL] +
                [S.text_width_pt(fig, lab, FS_KEY) for _, lab in key] +
                [S.text_width_pt(fig, ln, FS_GLOSS) for lines in gloss.values() for ln in lines])
    tick_w = max(S.text_width_pt(fig, b["display"], FS_TICK) for b in D["bounds"].values())
    tick_len, tick_pad = 2.0, 1.0
    left = 1.0 + 1.30 * FS_AXIS + 1.5 + tick_w + tick_pad + tick_len
    gap, gut_pad, right = 7.0, 9.5, 1.0
    top, bottom = 10.5, 22.0
    pw = (W - left - gap - gut_pad - gut_w - right) / 2.0
    ph = HEIGHT_PT - top - bottom

    def rect(x0, w):
        return [x0 / W, bottom / HEIGHT_PT, w / W, ph / HEIGHT_PT]

    ax_a = fig.add_axes(rect(left, pw))
    ax_b = fig.add_axes(rect(left + pw + gap, pw), sharex=ax_a, sharey=ax_a)
    gut_x = left + 2 * pw + gap + gut_pad
    panels = dict(zip(order, (ax_a, ax_b)))


    bands = (("capacity_free", y_lo, lo_b), ("informative", lo_b, hi_b), ("write_dominated", hi_b, y_hi))
    area = DOT_PT ** 2
    for tag, (axis, ax) in zip("ab", panels.items()):
        for r, y0, y1 in bands:
            ax.axhspan(y0, y1, color=S.tint(ROLE[r], 0.93), linewidth=0, zorder=0)
        for b in (lo_b, hi_b):
            ax.axhline(b, color=S.C["white"], linewidth=1.1, zorder=1)
            ax.axhline(b, color=S.C["ink2"], linewidth=0.45, zorder=1.1)
        series = axes_d[axis]["series"]
        for z, r in enumerate(DRAW_ORDER):
            if r in series:
                color, alpha = DOT[r]
                with S.marks(fig, f"{axis}/{r}"):
                    ax.scatter(series[r]["x"], series[r]["pi_k"], s=area, marker="o", color=color, alpha=alpha,
                               linewidths=0, zorder=3 + 0.1 * z)
        s = series.get("no_weight_writes")
        if s:
            with S.marks(fig, f"{axis}/no_weight_writes"):
                ax.plot(s["x"], s["pi_k"], linestyle="none", zorder=6, **DIAMOND)
        ax.set_xscale("log")
        ax.set_yscale("log")
        S.panel_title(ax, tag, axes_d[axis]["title"], pad=TITLE_PAD)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(pow10))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis="both", labelsize=FS_TICK, length=tick_len, pad=tick_pad)
    ax_a.set_xlim(x_lo, x_hi)
    ax_a.set_ylim(y_lo, y_hi)

    bound_text = {float(b["value"]): b["display"] for b in D["bounds"].values()}
    near = lambda v: any(abs(math.log10(v / b)) < 0.5 for b in bound_text)
    ax_a.yaxis.set_major_locator(FixedLocator(sorted(bound_text) +
                                              [v for v in decades(hi_b * 10, y_hi, step=3, phase=1)]))
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: bound_text.get(v) or pow10(v)))
    ax_a.yaxis.set_minor_locator(FixedLocator([v for v in decades(y_lo, y_hi) if not near(v)]))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_b.tick_params(axis="y", which="both", labelleft=False)

    y_len = S.text_width_pt(fig, Y_NAME, FS_AXIS)
    y_c = min(max(bottom + ph / 2.0, 1.5 + y_len / 2.0), HEIGHT_PT - 1.5 - y_len / 2.0)
    fig.text((1.0 + 0.62 * FS_AXIS) / W, y_c / HEIGHT_PT, Y_NAME, rotation=90, ha="center", va="center",
             fontsize=FS_AXIS)
    fig.text((left + pw + gap / 2.0) / W, 0.0, X_NAME, ha="center", va="bottom", fontsize=FS_AXIS)


    def fx(pt):
        return pt / W

    def fy(pt):
        return pt / HEIGHT_PT

    r_ = fig.canvas.get_renderer()

    def y_pt_of(v):
        ""
        return bottom + (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo)) * ph

    def line_box(text, size, **kw):
        ""
        t = fig.text(0.5, 0.5, text, fontsize=size, va="baseline", **kw)
        bb, y0 = t.get_window_extent(r_), 0.5 * fig.bbox.height
        t.remove()
        k = 72.0 / fig.dpi
        return bb.height * k, (bb.y1 - y0) * k, (y0 - bb.y0) * k

    def ink_down(text, size):
        ""
        return INK_DOWN * size if set(text) & set("gjpqy,()") else 0.0

    def left_ink(x_pt, base_pt, text, **kw):
        ""
        t = fig.text(fx(x_pt), fy(base_pt), text, ha="left", va="baseline", **kw)
        fp = t.get_fontproperties()
        bearing = TextPath((0, 0), text, size=fp.get_size_in_points(), prop=fp).get_extents().x0
        t.set_x(fx(x_pt - bearing))
        return t

    x0 = left + 2 * pw + gap
    rail_x = x0 + gut_pad / 2.0


    key_base = HEIGHT_PT - top + TITLE_PAD
    key_top = key_base + INK_UP * FS_KEY
    for i, (name, lab) in enumerate(key):
        base = key_base - i * KEY_PITCH * FS_KEY
        if name == "straddling":
            mk = dict(marker="o", markersize=2.4, markerfacecolor=DOT[name][0], markeredgewidth=0)
        else:
            mk = dict(DIAMOND)
        fig.add_artist(Line2D([fx(rail_x)], [fy(base + INK_MID * FS_KEY)], transform=fig.transFigure,
                              linestyle="none", **mk))
        left_ink(gut_x, base, lab, fontsize=FS_KEY, color=S.C["ink2"])
        key_foot = base - ink_down(lab, FS_KEY)


    name_fs = {r: (S.FS["focal"] if r == FOCAL else FS_NAME) for r in ROLE}
    block_h = {r: name_fs[r] * INK_UP + NAME_TO_GLOSS + (len(gloss[r]) - 1) * GLOSS_PITCH
               + ink_down(gloss[r][-1], FS_GLOSS) for r in ROLE}
    name_mid = {r: name_fs[r] * (INK_UP - INK_MID) for r in ROLE}


    r0 = bands[0][0]
    foot = line_box(X_NAME, FS_AXIS)[2] - ink_down(gloss[r0][-1], FS_GLOSS)
    stack_gap = (key_foot - foot - sum(block_h.values())) / len(bands)
    if stack_gap < COLUMN_GAP_MIN:
        raise SystemExit(f"{NAME}: the column's items stand {stack_gap:.1f} pt apart (least {COLUMN_GAP_MIN} pt)")
    tops, y_next = {}, key_foot - stack_gap
    for r, _, _ in reversed(bands):
        tops[r] = y_next
        y_next -= block_h[r] + stack_gap
    for r, y0, y1 in bands:
        y_top = tops[r]
        focal = r == FOCAL
        ink = ROLE[r] if focal else shade(ROLE[r], 0.30)
        base = y_top - INK_UP * name_fs[r]
        left_ink(gut_x, base, names[r], color=ink, fontsize=name_fs[r], fontweight="bold" if focal else "normal")
        for i, ln in enumerate(gloss[r]):
            left_ink(gut_x, base - NAME_TO_GLOSS - i * GLOSS_PITCH, ln, fontsize=FS_GLOSS, color=S.C["ink2"])


        y_name = y_top - name_mid[r]
        lo, hi = y_pt_of(max(y0, y_lo)), y_pt_of(min(y1, y_hi))
        mid = (lo + hi) / 2.0
        if abs(y_name - mid) < LEADER_STEP_MIN:
            raise SystemExit(f"{NAME}: the leader of {r} steps {abs(y_name - mid):.1f} pt (least {LEADER_STEP_MIN} "
                             f"pt): it would not read as the form of the other leaders")
        lx = [x0 + 0.8, rail_x, rail_x, gut_x - 1.4]
        ly = [mid, mid, y_name, y_name]
        fig.add_artist(Line2D([fx(v) for v in lx], [fy(v) for v in ly], transform=fig.transFigure,
                              color=ink, linewidth=0.5, solid_capstyle="butt", solid_joinstyle="miter",
                              alpha=0.9 if focal else 0.75))
        print(f"{NAME}: {r}: block {y_top - block_h[r]:.1f} to {y_top:.1f} pt, band {lo:.1f} to {hi:.1f} pt, "
              f"name at {y_name:.1f} pt, leader step {y_name - mid:+.1f} pt")
    print(f"{NAME}: plots {pw:.1f} x {ph:.1f} pt, column {gut_w:.1f} pt wide, key {key_foot:.1f} to {key_top:.1f} pt, "
          f"one gap of {stack_gap:.1f} pt between the column's four items, the column's foot {foot:.1f} pt over "
          f"the page's foot")


    rows = [{"what": f"regime bound on the intensity: {k}", "value": b["value"], "key": b["key"]}
            for k, b in D["bounds"].items()]
    rows.append({"what": "tolerance of the capacity-free reading, printed in the regime's gloss",
                 "value": tol, "key": D["tolerance"]["key"]})
    for axis, a in axes_d.items():
        for r, s in a["series"].items():
            rows.append({"what": f"{a['title']} panel, {s['label']}: one mark per point at its recorded position "
                                 f"({s['positions']} distinct positions)",
                         "value": s["points"], "source": s["source"], "ref": f"{axis}/{r}"})
    S.pending_key_at(fig, 0.5, 0.86, ha="center", ax=ax_b, words="Release version,\npending")
    S.save_fig(fig, NAME, rows=rows)
    S.save_keys(NAME, rows)


if __name__ == "__main__":
    main()
