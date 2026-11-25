""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.FS["title"] = 7.5
S.MIN_FONT_PT = 5.5
S.INK_CLEAR_PT = 1.5
import math
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

NAME = "fig_compare"
if "--preview" in sys.argv:
    _dir = Path(sys.argv[sys.argv.index("--preview") + 1]).resolve()
    if S.HERE in _dir.parents or _dir == S.HERE:
        raise SystemExit("a preview is never written under this folder")
    S.DATA = S.OUT = _dir
    S.MOCK = _dir / "mock"
    S.MOCK.mkdir(parents=True, exist_ok=True)
S.setup()
D = S.load(NAME)
C = S.C
FS = {"title": S.FS["title"], "focal": 8.0, "label": 7.0, "tick": 6.5, "small": 6.0, "least": 5.5}
W, H = S.TEXT_PT, 129.5


SEG_PRINTED = {}
ROW_PRINTED = {"Card, published geometry": "FPGA timing model, published core",
               "Card, wider core": "FPGA timing model, wider core",
               "Latency, our weight store": "Latency, weights loaded by port",
               "Latency, matched weight store": "Latency, weights in on-chip memory",
               "Energy, our weight store": "Energy, weights loaded by port",
               "Energy, matched weight store": "Energy, weights in on-chip memory"}
LIMIT_KEY = "Limit of our scripts"


def seg_name(s):
    ""
    n = s.get("short", s["label"])
    return SEG_PRINTED.get(n, n) if s["kind"] == "script" else n


def done_label(done):
    ""
    return f"{done['value']} {done['word']}"
fig = plt.figure(figsize=S.figsize("text", H), layout="none")
_R = fig.canvas.get_renderer()
KEYS = []
LATE = []
REF = [None]


def keyrow(what, value, d):
    ref = {"key": d["key"]} if d.get("key") else {"source": d["source"]}
    if d.get("key") and d.get("source") and d.get("in_ledger") is False:
        ref.update({"source": d["source"], **({"sha256": d["sha256"]} if d.get("sha256") else {})})
    KEYS.append({"what": what, "value": value, **ref, **({"ref": REF[0]} if REF[0] else {})})


def box(x, top, w, h, **kw):
    ""
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)


def canvas(x, top, w, h):
    ""
    ax = box(x, top, w, h)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return ax


def text_w(s, size, weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(_R).width * 72.0 / fig.dpi
    t.remove()
    return w


def times(s):
    return s.replace(" x ", " × ")


def thousands(v):
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


TITLE = 10.2
TITLE_PAD = 2.4
LEFT_W = 276.0
RIGHT_X = 289.0
B_CORES_X = 21.0

INSET = B_CORES_X - text_w("Cores", FS["tick"])
NAME_W = INSET + max(
    [text_w(t["tool"], FS["label"], "bold") if t["role"] == "ours" else
     max(text_w(t["tool"], FS["tick"]), text_w(done_label(t["done"]), FS["small"]))
     for t in D["coverage"]["tools"]])
BAR_X = NAME_W + 3.0
KEY_Y = -4.5
BAR_H, NAME_DY, NAME_LINE = 7.8, 4.5, 6.6
NAME_FOOT = 2.6
NAME_GAP = 0.9
LEAD_TOP, LEAD_END = 0.2, 1.2


BAR_GAP = 3.9
AB_GAP = 3.4
A_FIRST = 10.8
COUNT_BASE = 3.8
A_TOP = TITLE
A_H = A_FIRST + len(D["coverage"]["tools"]) * (BAR_H + NAME_DY + NAME_FOOT + BAR_GAP) - BAR_GAP
B_TITLE_TOP = A_TOP + A_H + AB_GAP
B_TOP = B_TITLE_TOP + TITLE
B_H = H - B_TOP
B_HEAD_Y, B_FOOT = 3.9, 4.5
B_ROWS = max(len(p["below_need"]) for p in D["budgets"]["pairings"])
B_PITCH = min(8.9, (B_H - B_HEAD_Y - B_FOOT) / (B_ROWS + 1))
B_NEED_Y = B_HEAD_Y + (B_ROWS + 1) * B_PITCH
B_RULE_DROP = 0.9
B_FS = next((v for v in (FS["label"], FS["tick"], FS["small"]) if B_PITCH >= 0.68 * 2 * v - 0.6), None)
if B_FS is None:
    raise SystemExit("panel (b): its rows do not fit under panel (a)")


def swatch(ax, x_right, label, size=None, draw=True, **style):
    ""
    size = size or FS["small"]
    tw = text_w(label, size)
    if draw:
        ax.text(x_right, KEY_Y, label, ha="right", va="center", fontsize=size, color=C["ink2"], clip_on=False)
        for st in style.pop("layers", None) or [style]:
            ax.add_patch(Rectangle((x_right - tw - 11.5, KEY_Y - 3.0), 8.5, 6.0, clip_on=False, **st))
    return x_right - tw - 11.5 - 7.0


def _row(items, lo_end, hi_end, bounds=None):
    ""
    if not items:
        return []
    offs, c = [], 0.0
    for i, (pref, w, g) in enumerate(items):
        c += g if i else 0.0
        offs.append(c); c += w
    if c > hi_end - lo_end + 1e-6:
        return None
    pools = []
    for (pref, w, g), o in zip(items, offs):
        pools.append([pref - o, 1])
        while len(pools) > 1 and pools[-2][0] / pools[-2][1] > pools[-1][0] / pools[-1][1]:
            t, n = pools.pop()
            pools[-1][0] += t; pools[-1][1] += n
    free = [t / n for t, n in pools for _ in range(n)]
    lo_u, hi_u = lo_end, hi_end - c
    los = [min(max(u, lo_u), hi_u) + o for u, o in zip(free, offs)]
    if bounds is None:
        return los
    n = len(items)
    low = [max(b[0], lo_end) for b in bounds]
    high = [min(b[1], hi_end - it[1]) for b, it in zip(bounds, items)]
    for i in range(1, n):
        low[i] = max(low[i], low[i - 1] + items[i - 1][1] + items[i][2])
    for i in range(n - 2, -1, -1):
        high[i] = min(high[i], high[i + 1] - items[i][1] - items[i + 1][2])
    if any(a > b + 1e-6 for a, b in zip(low, high)):
        return None
    out = []
    for i in range(n):
        v = min(max(los[i], low[i]), high[i])
        if out:
            v = min(max(v, out[-1] + items[i - 1][1] + items[i][2]), high[i])
        out.append(v)
    return out


def fit_names(segs, x0, scale, size, gap=4.2, clear=3.0, line_cost=60.0, slack=0.2):
    ""
    named, x = [], x0
    for s in segs:
        w = s["count"] * scale
        if s.get("name", True):
            named.append((s, x, w, text_w(seg_name(s), size)))
        x += w
    x_end, best = x, None
    for mask in range(1 << len(named)):
        on2 = [bool(mask >> i & 1) for i in range(len(named))]
        items = [(sx + w / 2, 0.0, clear) if two else (min(max(sx + w / 2 - tw / 2, x0), x_end - tw), tw, gap)
                 for (s, sx, w, tw), two in zip(named, on2)]
        items = [(pref, w, clear if (w == 0.0 or (i and items[i - 1][1] == 0.0)) else g)
                 for i, (pref, w, g) in enumerate(items)]


        bounds = [(sx + 0.8, sx + w - 0.8) if two else
                  (sx - 0.4 * tw, sx + w - 0.6 * tw) if tw <= w else
                  (sx + w / 2 - tw + 2.0, sx + w / 2 - 2.0) for (s, sx, w, tw), two in zip(named, on2)]
        los = _row(items, x0, x_end, bounds)
        if los is None:
            continue
        cost, ok = line_cost * sum(on2), True
        for (s, sx, w, tw), two, lo, (pref, _, _) in zip(named, on2, los, items):
            if two:
                ok &= sx + 0.8 <= lo <= sx + w - 0.8
            elif tw <= w:
                ok &= min(lo + tw, sx + w) - max(lo, sx) >= 0.6 * tw
            else:
                ok &= lo + 2.0 <= sx + w / 2 <= lo + tw - 2.0
            cost += abs(lo - pref)
        if not ok:
            continue
        second = [(lead, tw) for (s, sx, w, tw), two, lead in zip(named, on2, los) if two]
        pref2 = [min(max(lead - tw / 2, x0), x_end - tw) for lead, tw in second]
        los2 = _row([(pf, tw, gap) for pf, (lead, tw) in zip(pref2, second)], x0, x_end)
        if los2 is None or any(abs(lo - pf) > slack * tw or not lo + 2.0 <= lead <= lo + tw - 2.0
                               for lo, pf, (lead, tw) in zip(los2, pref2, second)):
            continue
        cost += sum(abs(lo + tw / 2 - lead) for lo, (lead, tw) in zip(los2, second))
        if best is None or cost < best[0]:
            out, it2 = [], iter(los2)
            for (s, sx, w, tw), two, lo in zip(named, on2, los):
                out.append((s, 2, next(it2) + tw / 2, lo) if two else
                           (s, 1, lo + tw / 2, sx + w / 2 if (tw > w or lo < sx - 1.0 or lo + tw > sx + w + 1.0)
                            and lo + 2 <= sx + w / 2 <= lo + tw - 2 else None))
            best = (cost, out)
    if best is None:
        raise SystemExit("the class names do not fit under their bar on two lines")
    return best[1]


def panel_a():
    cov = D["coverage"]
    total = cov["members"]["value"]
    ax = canvas(0, A_TOP, LEFT_W, A_H)
    title = f"Coverage of {total} classifiers"
    S.panel_title(ax, "a", title, pad=TITLE_PAD)
    keyrow("classifiers of the census", total, cov["members"])
    KEYS[-1]["ref"] = None
    scale = (LEFT_W - BAR_X - 1.0) / total
    bar_h = BAR_H
    partial = S.tint(C["gray"], 0.72)

    striped = [dict(facecolor=partial, edgecolor=C["gray"], linewidth=0.0, hatch="////////"),
               dict(facecolor="none", edgecolor="white", linewidth=0.8)]
    open_seg = dict(facecolor="white", edgecolor=C["gray"], linewidth=0.5)
    fills = {("ours", "done"): (C["blue"], "white"), ("ours", "host"): (S.tint(C["blue"], 0.60), C["ink"]),
             ("ours", "stop"): (C["gray3"], C["ink"]),
             ("prior", "done"): (C["gray"], "white"), ("prior", "stop"): (C["gray3"], C["ink"])}

    for t in cov["tools"]:
        for s in t["segments"]:
            s["name"] = s["kind"] != "partial"

    for name_fs in (FS["small"], FS["least"]):
        fits = [fit_names(t["segments"], BAR_X, scale, name_fs, gap=NAME_GAP * name_fs) for t in cov["tools"]]
        if all(ln == 1 for fit in fits for _, ln, _, _ in fit):
            break
    tops, top = [], A_FIRST
    for fit in fits:
        tops.append(top)
        top += bar_h + NAME_DY + NAME_LINE * (max([ln for _, ln, _, _ in fit] or [1]) - 1) + NAME_FOOT + BAR_GAP
    if top - BAR_GAP > A_H + 0.01:
        raise SystemExit("panel (a): the bars and their names do not fit the panel's height")
    keyed, limits = [], False
    for t, top, fit in zip(cov["tools"], tops, fits):
        ours = t["role"] == "ours"
        mid = top + bar_h / 2
        done = t["done"]
        if ours:
            ax.text(NAME_W, mid, t["tool"], ha="right", va="center", fontsize=FS["label"], color=C["blue"],
                    fontweight="bold")
        else:
            ax.text(NAME_W, mid, t["tool"], ha="right", va="center", fontsize=FS["tick"], color=C["ink"])
        REF[0] = f"a/{t['tool']}"
        _m = S.marks(fig, REF[0], series="a").__enter__()
        if not ours:
            ax.text(NAME_W, top + bar_h + NAME_DY, done_label(done), ha="right",
                    va="center", fontsize=FS["small"], color=C["ink"])
            keyrow(f"{t['tool']}: classifiers {done['word']}", f"{done['value']} of {done['of']}", done)
        x = BAR_X
        for s in t["segments"]:
            w = s["count"] * scale
            if s["kind"] == "script":
                ax.add_patch(Rectangle((x + 0.6, top + 0.25), w - 1.2, bar_h - 0.5, **open_seg))
                ax.text(x + w / 2, mid, str(s["count"]), ha="center", va="center", fontsize=FS["tick"],
                        color=C["ink2"])
                limits = True
            elif s["kind"] == "partial":
                for st in striped:
                    ax.add_patch(Rectangle((x, top), w, bar_h, **st))

                plate = min(text_w(str(s["count"]), FS["tick"]) + 1.0, w - 1.6)
                ax.add_patch(Rectangle((x + w / 2 - plate / 2, mid - 2.9), plate, 5.8, facecolor=partial,
                                       edgecolor="none"))
                ax.text(x + w / 2, mid, str(s["count"]), ha="center", va="center", fontsize=FS["tick"],
                        color=C["ink"])
                if s["label"] not in keyed:
                    keyed.append(s["label"])
            else:
                face, ink = fills[(t["role"], s["kind"])]
                ax.add_patch(Rectangle((x, top), w, bar_h, facecolor=face, edgecolor="white", linewidth=0.8))
                ax.text(x + w / 2, mid, str(s["count"]), ha="center", va="center", fontsize=FS["tick"], color=ink)
            keyrow(f"{t['tool']}: {s['label'].lower()}"
                   + (f" (printed as {s['short']!r})" if s.get("short") else ""), s["count"], s)
            x += w
        base = top + bar_h
        for s, line, xc, lead in fit:
            ax.text(xc, base + NAME_DY + (0.0 if line == 1 else NAME_LINE), seg_name(s), ha="center",
                    va="center", fontsize=name_fs, color=C["ink"] if ours and s["kind"] != "stop" else C["ink2"])
            if lead is not None:
                drop = 0.0 if line == 1 else NAME_LINE
                ax.plot([lead, lead], [base + LEAD_TOP, base + LEAD_END + drop], color=C["gray"],
                        linewidth=0.5, solid_capstyle="butt")
        if ours:
            x1 = BAR_X + done["value"] * scale
            ax.plot([BAR_X + 0.4, BAR_X + 0.4, x1 - 0.4, x1 - 0.4], [top - 0.9, top - 2.2, top - 2.2, top - 0.9],
                    color=C["blue"], linewidth=0.7, solid_joinstyle="miter")


            focal = f"{done['value']} of {done['of']} {done['word']}, {done['qualifier']}"
            fw = text_w(focal, FS["label"], "bold")
            fx = (BAR_X + x1) / 2 - fw / 2
            if fx < BAR_X or fx + fw > x1:
                raise SystemExit("panel (a): the count does not fit over its span")
            ax.text(fx, top - COUNT_BASE, focal, ha="left", va="baseline", fontsize=FS["label"], fontweight="bold",
                    color=C["blue"])
            keyrow("MimarSinan: classifiers converted", done["value"], done)
        _m.__exit__(None, None, None)
        REF[0] = None


    items = ([(LIMIT_KEY, open_seg)] if limits else []) + [(lab, dict(layers=striped))
                                                                      for lab in reversed(keyed)]
    title_end = text_w(f"(a) {title}", FS["title"], "bold")
    for key_fs in (FS["small"], FS["least"]):
        kx = LEFT_W - 1.0
        for lab, st in items:
            kx = swatch(ax, kx, lab, key_fs, draw=False)
        if kx + 7.0 >= title_end + 6.0:
            break
    else:
        raise SystemExit("panel (a): the key does not fit beside the title")
    kx = LEFT_W - 1.0
    for lab, st in items:
        kx = swatch(ax, kx, lab, key_fs, **{k: (list(v) if k == "layers" else v) for k, v in st.items()})
    print(f"panel (a): class names at {name_fs} pt, key at {key_fs} pt; panel (b): rows at a pitch of "
          f"{B_PITCH:.2f} pt, text at {B_FS} pt")


def panel_b():
    bud = D["budgets"]
    ax = canvas(0, B_TOP, LEFT_W, B_H)
    S.panel_title(ax, "b", f"{bud['network']} on fewer cores than one pass needs", pad=TITLE_PAD)
    pitch, head_y = B_PITCH, B_HEAD_Y
    b_small = min(B_FS, FS["tick"])
    x_cores, x_ours, x_tool = B_CORES_X, 33.0, 91.0

    def per_core(p):
        ""
        m = re.fullmatch(r"cores \((\S+) crossbars? each\)", p["budgets_cores"]["unit"])
        if not m:
            raise SystemExit(f"panel (b): {p['tool']}: the budgets' unit {p['budgets_cores']['unit']!r} names no "
                             f"crossbars per core")
        return m.group(1)

    def need_lines(p):
        ""
        kinds = {q["compiler"]["outcome"] for q in p["below_need"]}
        geom = times(p["geometry"])
        if kinds == {"refused"} and p["geometry_of"] == "crossbars":
            return [f"{p['tool']}'s fewest cores", f"Core: {per_core(p)} crossbars, {geom}"]
        if kinds == {"stages"} and p["geometry_of"] == "cores" and per_core(p) == "1":
            return [f"Cores of {geom} for one stage"]
        raise SystemExit(f"panel (b): {p['tool']}: no words for a need below which it {sorted(kinds)}")

    def block_width(p):
        ""
        cells = [f"{thousands(q['compiler']['stages'])} stages" if q["compiler"]["outcome"] == "stages" else "Rejected"
                 for q in p["below_need"]]
        return max([x_tool + 5.6 + text_w(c, B_FS) for c in cells] + [x_tool - 2.6 + text_w(p["tool"], B_FS)]
                   + [x_ours - 2.6 + text_w(ln, min(B_FS, FS["tick"])) for ln in need_lines(p)]) + 1.0

    widths = [block_width(p) for p in bud["pairings"]]
    if len(widths) != 2 or sum(widths) + 8.0 > LEFT_W:
        raise SystemExit("panel (b): its blocks do not fit side by side")
    lab = 5.6
    x0s = [0.0, LEFT_W - widths[1]]
    for p, x0 in zip(bud["pairings"], x0s):
        block = S.marks(fig, f"b/{p['tool']}", series="b",
                        block=f"b/{p['tool']}")
        S.block(fig, f"b/{p['tool']}")


        waits = "refusal" if {q["compiler"]["outcome"] for q in p["below_need"]} == {"refused"} else "stages"
        n_rows = len(p["below_need"])
        S.relabel(fig, f"b/{p['tool']}", ax.text(x0 + x_ours - 2.6, head_y + (n_rows + 2) * pitch / 2.0 + 0.4 * B_FS,
                  "", ha="left", va="baseline", fontsize=B_FS, color=C["ink2"]),
                  f"Passes, or {waits}: {S.PENDING_TOKEN}", as_tag=True)
        ax.text(x0 + x_cores, head_y, "Cores", ha="right", va="center", fontsize=b_small, color=C["ink2"])
        ax.text(x0 + x_ours - 2.6, head_y, "MimarSinan", ha="left", va="center", fontsize=B_FS,
                color=C["blue"])
        ax.text(x0 + x_tool - 2.6, head_y, p["tool"], ha="left", va="center", fontsize=B_FS, color=C["ink"])
        y = head_y
        for q in p["below_need"]:
            y += pitch
            ax.text(x0 + x_cores, y, thousands(q["cores"]), ha="right", va="center", fontsize=B_FS,
                    color=C["ink"])
            REF[0] = block.ref
            block.__enter__()
            ax.plot([x0 + x_ours], [y], linestyle="none", marker="o", markersize=4.6, color=C["blue"],
                    markeredgewidth=0)
            ax.text(x0 + x_ours + lab, y, f"{thousands(q['toolchain_passes'])} passes", ha="left", va="center",
                    fontsize=B_FS, color=C["ink"])
            keyrow(f"toolchain passes at {q['cores']} cores ({p['geometry']} {p['geometry_of']})",
                   q["toolchain_passes"], q)
            if q["compiler"]["outcome"] == "refused":
                ax.plot([x0 + x_tool], [y], linestyle="none", marker="x", markersize=4.2, color=C["gray"],
                        markeredgewidth=1.1)
                ax.text(x0 + x_tool + lab, y, "Rejected", ha="left", va="center", fontsize=B_FS, color=C["ink"])
                keyrow(f"{p['tool']} at {q['cores']} cores ({p['geometry']} {p['geometry_of']})", "refused", q)
            elif q["compiler"]["outcome"] == "stages":
                ax.plot([x0 + x_tool], [y], linestyle="none", marker="s", markersize=4.0, color=C["gray"],
                        markeredgewidth=0)
                ax.text(x0 + x_tool + lab, y, f"{thousands(q['compiler']['stages'])} stages", ha="left", va="center",
                        fontsize=B_FS, color=C["ink"])
                keyrow(f"{p['tool']} stages at {q['cores']} cores ({p['geometry']} {p['geometry_of']})",
                       q["compiler"]["stages"], q)
            else:
                raise SystemExit(f"{p['tool']}: outcome {q['compiler']['outcome']!r} has no mark")
            block.__exit__(None, None, None)
            REF[0] = None


        need = need_lines(p)
        if len(p["below_need"]) + len(need) > B_ROWS + 1:
            raise SystemExit(f"panel (b): {p['tool']}: the need's {len(need)} lines need rows the block does not have")
        rule_w = widths[x0s.index(x0)]
        rule_y = y + pitch / 2 + B_RULE_DROP
        ax.plot([x0, x0 + rule_w], [rule_y, rule_y], color=C["gray2"], linewidth=0.6, solid_capstyle="butt")
        need_y = y + pitch


        REF[0] = f"b/{p['tool']}/need"
        S.relabel(fig, REF[0], ax.text(x0 + x_cores, need_y, thousands(p["need_cores"]["value"]), ha="right",
                                       va="center", fontsize=B_FS, color=C["ink"]), "")
        for i, line in enumerate(need):
            ax.text(x0 + x_ours - 2.6, need_y + i * pitch, line, ha="left", va="center", fontsize=b_small,
                    color=C["ink2"])
        keyrow(f"{p['tool']}: need for ResNet-18 (cores)", p["need_cores"]["value"], p["need_cores"])
        LATE.append(S.category({"what": f"{p['tool']}: crossbars of one core, the unit of the block's core counts "
                                        f"(the need's words)", "value": per_core(p),
                                "source": p["budgets_cores"]["source"],
                                "derivation": "the data file's budgets_cores.unit"}))
        REF[0] = None


def reading(ax, r, yy, color, msize=4.6):
    ""
    line = S.tint(color, 0.45)
    filled = r["form"] == "measured"
    marks = r["marks"]
    if not marks and filled and r["hi"] > r["lo"]:
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=color, linewidth=RANGE_BAR, solid_capstyle="butt", zorder=3)
        ax.plot([r["lo"], r["hi"]], [yy, yy], linestyle="none", marker="|", markersize=msize + 1.6, color=color,
                markeredgewidth=1.1, zorder=4)
        return
    if r["hi"] > r["lo"]:
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=line, linewidth=0.9, solid_capstyle="butt", zorder=3)
    if marks:
        ax.plot(marks, [yy] * len(marks), linestyle="none", marker="o", zorder=4,
                markersize=msize, markerfacecolor=color if filled else "white",
                markeredgecolor="white" if filled else color, markeredgewidth=0.5 if filled else 0.8)
    else:
        ax.plot([r["lo"], r["hi"]], [yy, yy], linestyle="none", marker="|", markersize=msize - 0.2, color=color,
                markeredgewidth=0.9, zorder=4)


RANGE_BAR = 2.4


def minus(text):
    ""
    return text.replace("-", "−")


def nice(v):
    ""
    return next((s for s in (1, 2, 5, 10, 25, 50, 100) if s >= v), math.ceil(v / 100) * 100)


PEND_X = 3.0
ROW_TAG_FS = 6.5


def pending_mark(ax, yy, ax_w, size=None):
    ""
    S.tag_label(ax, PEND_X / ax_w, yy, transform=ax.get_yaxis_transform(), size=size or ROW_TAG_FS, ha="left")


def row_tag_at(ax, yy, ax_w, size):
    ""
    k = ROW_TAG_FS / S.PENDING_STYLE["tag_fs"]
    w = (S.TAG_RULE["length"] + S.TAG_RULE["gap"]) * k + text_w(S.PENDING_WORD, ROW_TAG_FS)
    x0, x1 = ax.get_xlim()
    return (ax, x0 + (x1 - x0) * (PEND_X + w / 2.0 - 0.25 * ROW_TAG_FS) / ax_w, yy, "data")


ROW_PAD = 5.0
TICK_LEN, TICK_PAD = 1.6, 1.0
TICK_FS = FS["least"]
TICK_DROP = TICK_LEN + TICK_PAD + 0.68 * TICK_FS
LOW_GROUP = "One layer"
GROUP_GAP = 4.0
SIDE = 2.6


ZERO_GAP, ZERO_PAD = 2.6, 3.2
SCALE_GAP = 2.8
ACCENT_FS = FS["focal"]


def row_names(ax, ys, rows, size):
    ""
    ax.set_yticks(ys, [ROW_PRINTED.get(r.get("row") or r["label"], r.get("row") or r["label"]) for r in rows])
    ax.tick_params(axis="y", length=0, pad=ROW_PAD, labelsize=size)
    ax.tick_params(axis="x", length=TICK_LEN, pad=TICK_PAD, labelsize=TICK_FS)
    for t, r in zip(ax.get_yticklabels(), rows):


        t.set_color(C["ink"])
        t.set_verticalalignment("center")


def panel_c():
    ""
    fid = D["fidelity"]
    band = fid["band_pct"]
    drawn = ("valued", "pending")
    rows = [r for r in fid["rows"] if r["status"] in drawn]
    apart = [r for r in fid["apart"] if r["status"] in drawn]
    for r in rows + apart:
        if r["status"] == "pending":
            KEYS.append({"what": f"{r['label']}: pending, a gray hollow mark on its row", "value": "pending",
                         "key": r["key"], "ref": f"c/{r['label']}"})
    for r in fid["rows"] + fid["apart"]:
        if r["status"] == "not_listed":
            KEYS.append({"what": f"{r['label']}: neither drawn nor listed ({r['statement']['text']})",
                         "value": r["statement"]["display"], "key": r["statement"]["key"]})
    axis_names = ("Estimate error", "Estimate over reference")
    full_w = W - RIGHT_X
    band_fill = S.tint(C["gray3"], 0.25)
    head = canvas(RIGHT_X, TITLE, full_w, H - TITLE)
    title_c = S.panel_title(head, "c", "Estimate against reference", pad=TITLE_PAD)

    def name_of(r):
        ""
        n = r.get("row") or r["label"]
        return ROW_PRINTED.get(n, n)


    low_groups = []
    for i, r in enumerate(apart):
        g = r.get("group") or LOW_GROUP
        if low_groups and low_groups[-1][0] == g:
            low_groups[-1][2] += 1
        else:
            low_groups.append([g, i, 1])

    def group_lines(name, n):
        ""
        words = name.split()
        best = [name]
        if n >= 2:
            for k in range(1, len(words)):
                cand = [" ".join(words[:k]), " ".join(words[k:])]
                if max(len(c) for c in cand) < max(len(c) for c in best):
                    best = cand
        return best

    def group_names(fs):
        ""
        broken = {g: group_lines(g, n) for g, _, n in low_groups}
        whole = {g: text_w(g, fs) for g in broken}
        col = max([text_w(ln, fs) for g, lines in broken.items() for ln in lines] + [0.0])
        return {g: [g] if whole[g] <= col + 0.05 else lines for g, lines in broken.items()}


    valued = [r for r in rows if r["status"] == "valued"
              and not S.withheld([{"what": r["label"], "key": r.get("key"), "source": r.get("source")}])]
    if band["status"] == "valued":
        lim = band["value"]
    else:
        lim = nice(max([abs(v) for r in valued for v in (r["lo"], r["hi"])] or [1]))
    xl = lim * 1.42
    msize = 4.8


    tick_row = TICK_DROP + 0.68 * TICK_FS + 0.2
    top_pad, group_gap, low_pad = 0.2, 0.0, 0.8
    up_pad = 1.4
    label_room = 7.6
    scale_gap = SCALE_GAP
    groups = len({r["group"] for r in rows})
    n_rows = len(rows) + len(apart)

    def plan(fs):
        ""
        group_w = max([text_w(ln, fs) for lines in group_names(fs).values() for ln in lines] + [0.0])
        group_w += GROUP_GAP if low_groups else 0.0
        label_w = max([text_w(name_of(r), fs) + ROW_PAD + 1.0 for r in rows]
                      + [group_w + text_w(name_of(r), fs) + ROW_PAD + 1.0 for r in apart]
                      + [text_w(a, TICK_FS) + 9.5 for a in axis_names])
        ax_w = W - RIGHT_X - label_w - 1.0

        def x_pt(v):
            return ax_w * (v + xl) / (2 * xl)

        def place(r):
            ""
            if r["status"] != "valued":
                return None
            if r["marks"]:
                if len(r.get("mark_texts") or []) != len(r["marks"]):
                    raise SystemExit(f"{r['label']}: a mark without the ledger's print form of its value")
                if len(r["marks"]) > 1:
                    return "over"
                text = r["mark_texts"][0]
            elif r.get("range_text"):
                text = r["range_text"]
            else:
                return None
            tw, off = text_w(minus(text), fs), msize / 2 + SIDE
            lo, hi = min([r["lo"]] + r["marks"]), max([r["hi"]] + r["marks"])
            zero = x_pt(0.0)

            def clear(a, b):
                return b <= zero - ZERO_GAP or a >= zero + ZERO_GAP

            right, past = x_pt(hi) + off, False
            if not clear(right, right + tw):
                right, past = zero + ZERO_PAD, True
            left = x_pt(lo) - off
            if right + tw <= ax_w - 0.5:
                return "right0" if past else "right"
            if left - tw >= PEND_X and clear(left - tw, left):
                return "left"
            return "over"

        places = [place(r) for r in rows]
        over = [p == "over" for p in places]
        fixed = (top_pad + (groups - 1) * group_gap + label_room * sum(over) + up_pad
                 + (tick_row + scale_gap + low_pad if apart else 0.0))
        room = B_TOP + B_NEED_Y - TICK_DROP - TITLE - fixed
        pitch = min(13.0, room / max(n_rows, 1))
        return label_w, ax_w, places, over, room, pitch


    for name_fs in (FS["tick"], FS["small"], FS["least"]):
        label_w, ax_w, places, over, room, pitch = plan(name_fs)
        print(f"panel (c): at {name_fs} pt the names' column is {label_w:.1f} pt, the scale {ax_w:.1f} pt, "
              f"{sum(over)} reading(s) over a mark, pitch {pitch:.2f} pt (least {0.68 * 2 * name_fs - 0.6:.2f})")
        if pitch >= 0.68 * 2 * name_fs - 0.6:
            break
    else:
        raise SystemExit(f"panel (c): {n_rows} lines do not fit the panel's height")
    ax_x = RIGHT_X + label_w
    gap_low = room - pitch * n_rows + (scale_gap if apart else 0.0)
    print(f"panel (c): {n_rows} lines at a pitch of {pitch:.2f} pt, names and readings at {name_fs} pt, "
          f"scale {ax_w:.1f} pt wide")


    ys, y, prev = [], top_pad + pitch / 2, None
    for i, r in enumerate(rows):
        if prev is not None and r["group"] != prev:
            y += group_gap
        if over[i]:
            y += label_room
        ys.append(y); prev = r["group"]; y += pitch
    up_h = y - pitch / 2 + up_pad
    ax = box(ax_x, TITLE, ax_w, up_h)
    ax.patch.set_alpha(0)
    if band["status"] == "valued":
        swatch(head, full_w - 1.0, f"Band of {lim}%", facecolor=band_fill, edgecolor="none")
        ax.axvspan(-lim, lim, color=band_fill, linewidth=0, zorder=0)
        keyrow("band of the cost estimates (percent)", lim, band)
    else:
        KEYS.append({"what": "band of the cost estimates (percent)", "value": "pending", "key": band["key"],
                     "ref": None})
    ax.set_xlim(-xl, xl); ax.set_ylim(up_h, 0)
    ax.spines["left"].set_visible(False)
    row_names(ax, ys, rows, name_fs)
    ax.set_xticks([-lim, 0, lim], [f"−{lim}%", "0", f"+{lim}%"])

    ax.plot([0, 0], [ys[0] if ys else 0.0, up_h], color=C["gray2"], linewidth=0.6, solid_capstyle="butt", zorder=1)
    head.text(ax_x - RIGHT_X - 9.0, up_h + TICK_DROP, axis_names[0], ha="right", va="center", fontsize=TICK_FS,
              color=C["ink2"])

    def put(text, x_over, lo, hi, yy, where):
        ""
        if where == "over":
            tw, half = text_w(minus(text), name_fs), ax_w / 2.0
            centre = ax_w * (x_over + xl) / (2 * xl)
            if yy > ys[0] and centre - tw / 2 < half + ZERO_GAP and centre + tw / 2 > half - ZERO_GAP:
                sides = (1, -1) if (lo + hi) / 2 >= 0 else (-1, 1)
                fits = [d for d in sides if (half + ZERO_PAD + tw <= ax_w - 0.5 if d > 0 else half - ZERO_PAD - tw >= PEND_X)]
                if fits:
                    ax.annotate(minus(text), (0.0, yy), xytext=(ZERO_PAD * fits[0], msize / 2 + 1.2),
                                textcoords="offset points", ha="left" if fits[0] > 0 else "right", va="bottom",
                                fontsize=name_fs, color=C["ink"])
                    return
            ax.annotate(minus(text), (x_over, yy), xytext=(0, msize / 2 + 1.2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=name_fs, color=C["ink"])
        elif where == "right0":
            ax.annotate(minus(text), (0.0, yy), textcoords="offset points", xytext=(ZERO_PAD, 0), ha="left",
                        va="center", fontsize=name_fs, color=C["ink"])
        else:
            right = where == "right"
            ax.annotate(minus(text), (hi if right else lo, yy), textcoords="offset points",
                        xytext=((msize / 2 + SIDE) * (1 if right else -1), 0), ha="left" if right else "right",
                        va="center", fontsize=name_fs, color=C["ink"])

    for r, yy, where in zip(rows, ys, places):
        REF[0] = f"c/{r['label']}"
        S.block(fig, f"c/{r['group']}", across=ax)
        _m = S.marks(fig, REF[0], series="c", block=f"c/{r['group']}", tag_size=ROW_TAG_FS,
                     tag_at=row_tag_at(ax, yy, ax_w, name_fs)).__enter__()
        if r["status"] == "pending":
            pending_mark(ax, yy, ax_w)
            _m.__exit__(None, None, None)
            continue
        reading(ax, r, yy, C["blue"], msize)
        lo, hi = min([r["lo"]] + r["marks"]), max([r["hi"]] + r["marks"])
        for v, shown in zip(r["marks"], r.get("mark_texts") or []):
            put(shown, v, lo, hi, yy, where)
        if not r["marks"] and r.get("range_text"):
            put(r["range_text"], (r["lo"] + r["hi"]) / 2, lo, hi, yy, where)
            keyrow(f"{r['label']}: error of the estimate (percent), {r.get('range_words', 'the range the ledger prints')}"
                   f", drawn alone (a bar with end caps, no single reading)", r["range_text"], r)
        else:
            keyrow(f"{r['label']}: error of the estimate (percent), low and high, then the readings drawn as "
                   f"marks", [round(r["lo"], 3), round(r["hi"], 3)] + [round(v, 3) for v in r["marks"]], r)
            if r["marks"]:
                keyrow(f"{r['label']}: the label of each mark, the ledger's print form", r["mark_texts"], r)
        _m.__exit__(None, None, None)
    REF[0] = None
    y_end = up_h + tick_row


    if apart:
        low_top = TITLE + y_end + gap_low
        low_h = len(apart) * pitch + low_pad
        head_fs = min(FS["small"], name_fs)
        head.text(ax_x - RIGHT_X - 9.0, low_top - TITLE + low_h + TICK_DROP, axis_names[1], ha="right",
                  va="center", fontsize=TICK_FS, color=C["ink2"])
        lx = box(ax_x, low_top, ax_w, low_h)
        lx.patch.set_alpha(0)
        lx.set_xscale("log")
        seen = [r for r in apart if r["status"] == "valued"]
        lo = min([r["lo"] for r in seen] + [1.0]); hi = max([r["hi"] for r in seen] + [1.0])
        lx.set_xlim(*S.frame_range(lo / 1.6, hi * 1.4, [{"what": r["label"], "key": r.get("key"),
                                                         "source": r.get("source")} for r in seen], scale="log"))
        lx.set_ylim(low_h, 0)
        lx.spines["left"].set_visible(False)
        lys = [low_pad / 2 + pitch / 2 + i * pitch for i in range(len(apart))]
        names = group_names(head_fs)
        for gi, (name, first, n) in enumerate(low_groups):
            for j, ln in enumerate(names[name]):
                head.text(0.0, low_top - TITLE + lys[first + j], ln, ha="left", va="center", fontsize=head_fs,
                          color=C["ink2"])
            if gi:
                y_rule = low_top - TITLE + lys[first] - pitch / 2
                head.plot([0.0, ax_x - RIGHT_X - ROW_PAD + 1.0], [y_rule, y_rule], color=C["gray3"], linewidth=0.5,
                          solid_capstyle="butt")
        row_names(lx, lys, apart, name_fs)
        x0, x1 = lx.get_xlim()
        to_pt = lambda v: ax_w * math.log(v / x0) / math.log(x1 / x0)
        to_v = lambda pt: x0 * (x1 / x0) ** (pt / ax_w)

        decades = [10.0 ** k for k in range(-6, 9) if x0 <= 10.0 ** k <= x1]
        order = sorted(decades, key=lambda v: (v != 1.0, -abs(math.log10(v) - math.log10(decades[len(decades) // 2]))))
        kept = []
        for v in order:
            half = text_w(f"{thousands(v)}×", TICK_FS) / 2
            if to_pt(v) + half > W - ax_x - 0.5:
                continue
            if all(abs(to_pt(v) - to_pt(k)) >= half + text_w(f"{thousands(k)}×", TICK_FS) / 2 + 4.0 for k in kept):
                kept.append(v)
        lx.xaxis.set_major_locator(FixedLocator(sorted(kept)))
        lx.xaxis.set_minor_locator(NullLocator())
        lx.xaxis.set_major_formatter(lambda v, _: f"{thousands(v)}×")
        centre = math.sqrt(x0 * x1)
        one = to_pt(1.0)
        bar_h = 4.4
        ends = {}
        for i, (r, yy) in enumerate(zip(apart, lys)):
            REF[0] = f"c/{r['label']}"
            low = f"c/{r.get('group') or LOW_GROUP}"
            S.block(fig, low, across=lx)
            if r["status"] == "pending":
                with S.marks(fig, REF[0], series="c", block=low):
                    pending_mark(lx, yy, ax_w)
                continue

            lx.plot([1, 1], [yy - pitch / 2, low_h if i == len(apart) - 1 else yy + pitch / 2], color=C["gray2"],
                    linewidth=0.6, solid_capstyle="butt", zorder=1)
            _m = S.marks(fig, REF[0], series="c", block=low).__enter__()
            accent, own = bool(r.get("accent")), r.get("store") == "own"

            edge = C["blue"] if accent else C["gray"] if own else C["ink2"]
            ink = C["blue"] if accent else C["ink2"] if own else C["ink"]
            if r["marks"] or r["form"] == "measured" or not r["hi"] > r["lo"]:
                reading(lx, r, yy, C["gray"], msize)
            else:
                bh = bar_h + (0.8 if accent else 0.0)
                lx.add_patch(Rectangle((r["lo"], yy - bh / 2), r["hi"] - r["lo"], bh, facecolor="white",
                                       edgecolor=edge, linewidth=1.3 if accent else 1.0, joinstyle="miter",
                                       zorder=4))
            keyrow(f"{r['label']}: estimate over reference (ratio), low and high, then the readings drawn as marks",
                   [round(r["lo"], 4), round(r["hi"], 4)] + [round(v, 4) for v in r["marks"]], r)
            if r.get("row") and r.get("range_text"):
                read_fs = ACCENT_FS if accent else name_fs
                tw = text_w(r["range_text"], read_fs, "bold" if accent else "normal")
                a, b = to_pt(r["lo"]), to_pt(r["hi"])
                x = b + 3.6
                if x < one + 2.6 and x + tw > one - 2.6:
                    x = one + 3.2
                right = x + tw <= ax_w - 1.0
                if not right:
                    x = a - 3.6
                    if x - tw < one + 2.6:
                        raise SystemExit(f"{r['label']}: its factor range does not fit beside its bar")
                lx.text(x / ax_w, yy, r["range_text"], transform=lx.get_yaxis_transform(),
                        ha="left" if right else "right", va="center", fontsize=read_fs, color=ink,
                        fontweight="bold" if accent else "normal", zorder=5)
                ends[id(r)] = (a, b, x + tw if right else x, yy)
                keyrow(f"{r['label']}: the factor range printed beside its bar (two significant figures)",
                       r["range_text"], r)
            elif r.get("cause"):
                right = math.sqrt(r["lo"] * r["hi"]) < centre
                x = to_pt(r["hi"]) + 4.0 if right else to_pt(r["lo"]) - 4.0
                tw = text_w(r["cause"], name_fs)
                if right and x < one + 2.6 and x + tw > one - 2.6:
                    x = one + 3.2
                if (x + tw > ax_w - 1.0) if right else (x - tw < to_pt(1.0) + 3.0):
                    raise SystemExit(f"{r['label']}: its cause does not fit beside its range")
                lx.text(x / ax_w, yy, r["cause"], transform=lx.get_yaxis_transform(), ha="left" if right else "right",
                        va="center", fontsize=name_fs, color=C["ink2"], zorder=5)
            _m.__exit__(None, None, None)
            if r.get("cause_display"):
                KEYS.append({"what": f"{r['label']}: named cause ({r['cause']})"
                                     + (", carried by the row's name" if r.get("row") else ""),
                             "value": r["cause_display"], "key": r["cause_key"], "ref": REF[0]})
        REF[0] = None

        for r in apart:
            src = next((q for q in apart if r.get("accent") and q.get("store") == "own"
                        and (q.get("group"), q.get("axis")) == (r.get("group"), r.get("axis"))), None)
            if src is None or id(r) not in ends or id(src) not in ends:
                continue
            (sa, sb, _, sy), (_, _, label_end, ry) = ends[id(src)], ends[id(r)]
            start = (sa + sb) / 2
            if start - label_end < 14.0:
                continue
            down = 1 if ry > sy else -1
            with S.marks(fig, f"c/{r['label']}", series="c",
                         block=f"c/{r.get('group') or LOW_GROUP}"):
                lx.annotate("", xy=(to_v(label_end + 3.0), ry), xytext=(to_v(start), sy + down * (bar_h / 2 + 1.2)),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.26", mutation_scale=6.0,
                                            connectionstyle="angle,angleA=90,angleB=0,rad=0", color=C["blue"],
                                            linewidth=0.7, shrinkA=0, shrinkB=0), zorder=3)
    S.pending_key_at(fig, W - 1.0, 0.0, ha="right", anchor=title_c, x_in="points", size=FS["small"])


panel_a()
panel_b()
REF[0] = None
panel_c()
KEYS += LATE
for r in KEYS:
    r.setdefault("ref", None)
S.save_fig(fig, NAME, rows=KEYS)
S.save_keys(NAME, KEYS)
