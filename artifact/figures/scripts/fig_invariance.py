""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.MIN_FONT_PT = 5.5

S.FS["title"] = 7.5

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

NAME = "fig_invariance"
FS = {"focal": 8.0, "label": 6.8, "tick": 6.5, "small": 6.0}

LINE_EM, DESC_EM, AIR_EM = 1.362, 0.293, 0.31


NOTE_BASE, TITLE_UP, BAND_UP, BAND_TOP = 2.6, 8.4, 9.8, 8.5
NOTE_SIZES = (FS["tick"], FS["small"])
NOTE_PITCH_EM, NOTE_GAP = 1.27, 3.0
SPACE_EM = 0.26
PASS_H, FOOT = 46.7, 20.6
EDGE, RIGHT, COL_GAP = 0.6, 5.0, 9.0
YLAB_GAP, TICK_PAD, TICK_LEN, TICK_MINOR = 1.2, 1.0, 2.2, 1.3
XTICK_PAD, XTICK_LEN = 0.8, 1.8
XLABEL_UP = 0.0
PAD_DEC = 0.15
ZERO_PT = 4.0
AXIS_AIR = 2.4
COUNT_AIR = 0.8

EXEC_PT, PLAN_PT = 4.2, 2.3
LIN_DEC = 0.42
TOP_ROOM = 7.7

DASH = (0, (3.2, 2.2))
EQUAL = ("equal", "checked")
LABEL_GAP = 3.0
NAME_FILL = 30.0
RESIDENT_SLIDE = (8.0, 16.0, 24.0, 32.0, 44.0)
GEOMETRY_GAP = 6.0
GEOMETRY_JOINS = (" of ", ", ")
GEOMETRY_SLIDE = (0.0, -4.0, 4.0, -8.0, 8.0, -12.0, 12.0)
COUNT_MIN_DX = 13.0
COUNT_PLACES = ("NW", "N", "NE", "SE", "S", ("N", 3.0), ("N", -3.0))

STEP_MIN = 0.10
KEY_FORMS = (("checked", "Equal outputs"), ("pending", "Pending"), ("planned", "Planned only"))
SPLIT_WORDS = ("Development-split accuracy", "Dev. accuracy")
EVERY = "at every executed budget"


def geometry_label(g: str) -> str:
    w, h = g.split("x")
    return f"{w} × {h}"


def status_of(r) -> str:
    ""
    s = r.get("status")
    if s == "reference":
        return "reference"
    if s in EQUAL:
        return "checked"
    if s == "planned":
        return "planned"
    return "pending"


def share_of(r, n) -> float:
    ""
    if status_of(r) == "reference":
        return 1.0
    if r.get("share"):
        return float(r["share"])
    if r.get("share_den"):
        return 1.0 / float(r["share_den"])
    return r["cores"] / float(n["resident_cores"])


def budget_label(r) -> str:
    ""
    if status_of(r) == "reference":
        return "1"
    if r.get("share_den"):
        return f"1/{r['share_den']:g}"
    return r["budget"]


def accuracy_of(r):
    ""
    if r.get("correct") is not None and r.get("images"):
        return 100.0 * r["correct"] / r["images"]
    return r.get("accuracy_pct")


def accuracy_note(n, rows):
    ""
    ran = [r for r in rows if status_of(r) != "planned"]
    held = [(r, accuracy_of(r)) for r in ran]
    vals = [v for _, v in held if v is not None]
    if not vals:
        return {"value": None, "where": "", "every": False, "keys": lambda text: []}
    lo, hi = min(vals), max(vals)
    same = hi - lo < 1e-9
    digits = 1 if same or f"{lo:.1f}" != f"{hi:.1f}" else 2
    value = f"{lo:.{digits}f}%" if same else f"{lo:.{digits}f}% to {hi:.{digits}f}%"
    if len(vals) == len(ran):
        where = EVERY if same else "over the executed budgets"
    else:
        where = f"at {len(vals)} of {len(ran)} executed budgets"
    names = ", ".join(r["budget"] for r, v in held if v is not None)

    def keys(text):
        out = [{"what": f"{n['label']}: header line, deployed accuracy of the executed budgets, development split",
                "value": text, "key": n["count"]["key"],
                "source": f"derived: 100 x correct / images of each executed row ({names}), "
                          f"{len(set(round(v, 9) for v in vals))} distinct value(s) on {len(vals)} of {len(ran)} "
                          "executed rows; each row's count is its keys row 'correct development-split images'"}]
        if same and abs(lo - n["accuracy_pct"]) < 1e-9:
            out.append({"what": f"{n['label']}: deployed accuracy, development split (%)",
                        "value": f"{n['accuracy_pct']:.1f}", "key": n["count"]["key"],
                        "source": n["accuracy_source"]})
        return out

    return {"value": value, "where": where, "every": same and len(vals) == len(ran), "keys": keys}


def note_parts(note, words, with_where):
    ""
    if note["value"] is None:
        return f"{words} pending", ""
    return f"{words} {note['value']}", (note["where"] if with_where else "")


def fit_notes(fig, notes, rooms):
    ""
    forms = [(SPLIT_WORDS[0], True), (SPLIT_WORDS[1], True)]
    if all(n["every"] for n in notes if n["value"] is not None):
        forms.append((SPLIT_WORDS[0], False))
    for words, with_where in forms:
        parts = [note_parts(n, words, with_where) for n in notes]
        texts = [" ".join(t for t in pair if t) for pair in parts]
        for fs in NOTE_SIZES:
            if all(S.text_width_pt(fig, t, fs) <= room for t, room in zip(texts, rooms)):
                return fs, parts, [True] * len(texts), with_where
    fs = NOTE_SIZES[-1]
    for words in SPLIT_WORDS:
        parts = [note_parts(n, words, True) for n in notes]
        if all(S.text_width_pt(fig, pair[0], fs) <= room for pair, room in zip(parts, rooms)):
            break
    else:
        print(f"FAIL {NAME}: a header's value line fits no panel at {fs} pt")
    texts = [" ".join(t for t in pair if t) for pair in parts]
    return fs, parts, [S.text_width_pt(fig, t, fs) <= room or not pair[1]
                       for t, pair, room in zip(texts, parts, rooms)], True


RECORD_BASIS = ("a field of the row's record; the record is a source of the ledger key (which holds its sha256); "
                "no ledger key prints the value")
PLANNED_BASIS = "a planned row: the planner's mapping and pass schedule alone, no execution, no comparison"
COLLECTED_BASIS = ("an executed row that is not a ledger key yet: a row of the collected table of the dense ladder "
                   "and a field of the row's own record")


def collected_basis(r):
    ""
    if r.get("record_key"):
        return None
    return (r.get("collected") or {}).get("basis") or COLLECTED_BASIS


def record_row(r, what, value, source):
    ""
    if status_of(r) == "planned":
        return {"what": what, "value": value, "source": source or r.get("source"),
                "record_sha256": r.get("source_sha256"), "basis": PLANNED_BASIS}
    if collected_basis(r):
        return {"what": what, "value": value, "source": source, "record_sha256": r.get("record_sha256"),
                "basis": collected_basis(r)}
    return {"what": what, "value": value, "key": r["record_key"], "source": source,
            "record_sha256": r["record_sha256"], "basis": RECORD_BASIS}


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


class Placer:
    ""

    def __init__(self, ax, width, height, gap=0.0):
        self.ax, self.w, self.h = ax, width, height
        self.gap = gap
        self.texts, self.lines, self.marks = [], [], []

    def add_line(self, pts):
        self.lines += list(zip(pts, pts[1:]))

    def add_mark(self, x, y, radius):
        self.marks.append((x, y, radius))

    def boxes(self, x, y, w, fs, radius, places):
        h, d, air = LINE_EM * fs, DESC_EM * fs, AIR_EM * fs
        gap = self.gap
        reach = max(0.0, radius - (EXEC_PT + 0.4) / 2.0)
        for place in places:
            shift = 0.0
            if not isinstance(place, str):
                place, shift = place
            x += shift
            if place == "NW":
                x0, y0 = x - 1.8 - reach - gap - w, y + 0.6 + reach + gap / 2.0 - d
            elif place == "NE":
                x0, y0 = x + 1.8 + reach + gap, y + 0.6 + reach + gap / 2.0 - d
            elif place == "N":
                x0, y0 = x - w / 2.0, y + radius + 1.2 + gap - d
            elif place == "C0":
                x0, y0 = x - w / 2.0, AXIS_AIR - d
            elif place == "C1":
                x0, y0 = x - w / 2.0, AXIS_AIR - d + h
            elif place == "E":
                x0, y0 = x + radius + 2.2, y - h / 2.0
            elif place == "E0":
                x0, y0 = x + radius + 2.2, AXIS_AIR - d
            elif place == "SE":
                x0, y0 = x + 2.2 + reach + gap, y - 0.6 - reach - gap / 2.0 + air - h
            else:
                x0, y0 = x - w / 2.0, y - radius - 1.0 - gap + air - h
            x -= shift
            yield place, (x0, y0, x0 + w, y0 + h), (x0 - 0.6, y0 + d - 0.6, x0 + w + 0.6, y0 + h - air + 0.6)

    def place(self, x, y, text, fs, radius, places, **kw):
        w = S.text_width_pt(self.ax.figure, text, fs, kw.get("fontweight", "normal"))
        for _, box, ink in self.boxes(x, y, w, fs, radius, places):
            if box[0] < 0.8 or box[2] > self.w + 4.0 or ink[1] < 0.2 or box[3] > self.h + 1.5:
                continue
            if any(overlap(box, t, 0.4) for t in self.texts):
                continue
            if any(crosses(ink, p, q) for p, q in self.lines):
                continue
            if any(overlap(ink, (mx - r, my - r, mx + r, my + r)) for mx, my, r in self.marks
                   if abs(mx - x) > 1e-6 or abs(my - y) > 1e-6):
                continue
            self.texts.append(box)
            px = self.ax.figure.dpi / 72.0
            inv = self.ax.transAxes.inverted()
            fx, fy = inv.transform((self.ax.bbox.x0 + box[0] * px, self.ax.bbox.y0 + box[1] * px))
            return self.ax.text(fx, fy, text, transform=self.ax.transAxes, ha="left", va="bottom", fontsize=fs,
                                clip_on=False, zorder=7, **kw)
        return None


def tick_names(rows, xs_pt, fig):
    ""
    def box(i):
        w = S.text_width_pt(fig, budget_label(rows[i]), FS["tick"])
        return xs_pt[i] - w / 2.0, xs_pt[i] + w / 2.0

    def free(i, named):
        a = box(i)
        return all(a[0] - box(j)[1] >= LABEL_GAP or box(j)[0] - a[1] >= LABEL_GAP for j in named)

    named = []
    for i in [i for i, r in enumerate(rows) if status_of(r) != "planned"] + [len(rows) - 1]:
        if i not in named and free(i, named):
            named.append(i)
    def binary(i):
        den = rows[i].get("share_den")
        return bool(den) and float(den) >= 1 and math.log2(float(den)).is_integer()

    def reach(i):
        return min(abs(xs_pt[i] - xs_pt[j]) for j in named)

    for i in range(len(rows)):
        if i not in named and binary(i) and free(i, named):
            named.append(i)
    while True:
        rest = [i for i in range(len(rows)) if i not in named and free(i, named) and reach(i) >= NAME_FILL]
        if not rest:
            return sorted(named)
        named.append(max(rest, key=reach))


def main():
    d = S.load(NAME)
    nets = d["networks"]
    S.setup()
    plt.rcParams["figure.constrained_layout.use"] = False
    W = S.width_pt("text")
    probe = plt.figure(figsize=S.figsize("text", PASS_H))
    probe.canvas.draw()

    views = []
    for n in nets:
        rows = sorted((r for r in n["rows"] if "cores" in r), key=lambda r: -share_of(r, n))
        views.append((n, rows, [share_of(r, n) for r in rows]))
    spans = [math.log10(max(xs) / min(xs)) for _, _, xs in views]
    widths = [max(1.3, 0.55 + 0.75 * s) for s in spans]
    p_hi = max(r["passes"] for _, rows, _ in views for r in rows)
    tick_w = S.text_width_pt(probe, S.thousands(10 ** int(math.log10(max(p_hi, 10)))), FS["tick"])
    ylab_w = 2 * 0.95 * LINE_EM * FS["label"]
    left = EDGE + ylab_w + YLAB_GAP + tick_w + TICK_PAD + TICK_LEN
    plot_w = W - left - RIGHT - COL_GAP * (len(nets) - 1)
    pws = [plot_w * w / sum(widths) for w in widths]
    lefts = [left + sum(pws[:j]) + COL_GAP * j for j in range(len(pws))]


    notes = [accuracy_note(n, rows) for n, rows, _ in views]
    rooms = [pw + COL_GAP - NOTE_GAP for pw in pws[:-1]] + [pws[-1] + RIGHT - EDGE]
    note_fs, note_text, note_one, note_where = fit_notes(probe, notes, rooms)
    plt.close(probe)
    if not note_where:
        print(f"NOTE {NAME}: the header lines print the split and the value alone: '{EVERY}' fits no panel on "
              f"one line at {NOTE_SIZES[-1]} pt with either form of the split's words")
    note_rows = 1 if all(note_one) else 2
    note_up = (note_rows - 1) * NOTE_PITCH_EM * note_fs
    note_big = LINE_EM * (note_fs - NOTE_SIZES[-1])
    pass_h = PASS_H - note_big
    title_base = NOTE_BASE + note_up + TITLE_UP + note_big
    band_pt = FOOT + pass_h + title_base + BAND_UP
    HEIGHT_PT = band_pt + BAND_TOP
    if note_rows > 1:
        print(f"NOTE {NAME}: a header line stands on two lines (it does not fit its panel at {note_fs} pt), "
              f"the page is {HEIGHT_PT:.1f} pt high")

    fig = plt.figure(figsize=S.figsize("text", HEIGHT_PT))
    fig.canvas.draw()
    axes = [fig.add_axes([x0 / W, FOOT / HEIGHT_PT, pw / W, pass_h / HEIGHT_PT]) for x0, pw in zip(lefts, pws)]

    keys, forms, checks = [], set(), 0
    blue, green, gray = S.C["blue"], S.C["green"], S.C["gray"]

    for j, (n, rows, xs) in enumerate(views):
        b = axes[j]
        pw = pws[j]
        k0 = len(keys)
        ref = rows[0]
        ran = [(r, x) for r, x in zip(rows, xs) if status_of(r) != "planned"]
        b.set_xscale("log")
        b.set_xlim(max(xs) * 10 ** PAD_DEC, min(xs) / 10 ** PAD_DEC)
        b.set_yscale("symlog", linthresh=1.0, linscale=LIN_DEC)
        b.set_ylim(*S.frame_range(0, p_hi * TOP_ROOM, [record_row(ref, "scheduled passes", ref["passes"], None)],
                                  scale="symlog"))


        name, _, qualifier = n["label"].partition(", ")
        title = S.panel_title(b, "abcdefgh"[j], name, pad=title_base)
        if qualifier:
            room = W - EDGE - lefts[j] - S.text_width_pt(fig, title.get_text(), S.FS["title"], "bold") - 3.0

            fs_q = note_fs if S.text_width_pt(fig, qualifier, note_fs) <= room else FS["small"]
            b.annotate(qualifier, xy=(1.0, 0.0), xycoords=title, xytext=(3.0, DESC_EM * (S.FS["title"] - fs_q)),
                       textcoords="offset points", ha="left", va="bottom", fontsize=fs_q, color=S.C["ink"])
        head, where = note_text[j]
        y_note = NOTE_BASE + (note_rows - 1) * NOTE_PITCH_EM * note_fs
        kw = dict(textcoords="offset points", ha="left", fontsize=note_fs)
        _panel = S.marks(fig, f"n{j}", series=f"n{j}").__enter__()
        first = b.annotate(head, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, y_note), va="baseline",
                           color=S.C["ink"], **kw)
        value = notes[j].get("value")
        if value and head.endswith(value):


            S.relabel(fig, f"n{j}", first, f"{head[:-len(value)].rstrip()} {S.PENDING_TOKEN}", tag_size=5.5)
        if where and note_one[j]:
            b.annotate(where, xy=(1.0, 0.0), xycoords=first, xytext=(SPACE_EM * note_fs, 0.0), va="bottom",
                       color=S.C["ink2"], **kw)
        elif where:
            b.annotate(where, xy=(0.0, 1.0), xycoords="axes fraction",
                       xytext=(0.0, y_note - NOTE_PITCH_EM * note_fs), va="baseline", color=S.C["ink2"], **kw)
        keys += notes[j]["keys"](" ".join(t for t in note_text[j] if t))


        place = Placer(b, pw, pass_h, gap=COUNT_AIR)
        home = [(r, x) for r, x in zip(rows, xs) if r["geometry"] == ref["geometry"]]
        b.plot([x for _, x in home], [r["passes"] for r, _ in home], color=blue, zorder=2, linewidth=S.LW["line"],
               solid_capstyle="round", solid_joinstyle="round")
        place.add_line([to_pt(b, x, r["passes"]) for r, x in home])
        for (r0, x0_), (r1, x1_) in zip(ran, ran[1:]):
            if r1["geometry"] != r0["geometry"]:
                b.plot([x0_, x1_], [r0["passes"], r1["passes"]], color=blue, zorder=2, linewidth=S.LW["thin"],
                       linestyle=DASH)
                place.add_line([to_pt(b, x0_, r0["passes"]), to_pt(b, x1_, r1["passes"])])
        pts = []
        for r, x in zip(rows, xs):
            form = status_of(r)
            forms.add(form)
            if form == "planned":
                b.plot([x], [r["passes"]], marker="o", markersize=PLAN_PT, markerfacecolor="white",
                       markeredgecolor=blue, markeredgewidth=0.7, zorder=4, clip_on=False)
                size = PLAN_PT + 0.7
            elif form == "checked":
                S.zero_mark(b, x, r["passes"], size=ZERO_PT, clip_on=False)
                size = ZERO_PT
                keys.append({"what": f"{n['label']} {r['budget']}: output codes differing from the resident deployment",
                             "value": f"{r['differing']} of {S.thousands(r['outputs'])}", "source": r["differing_source"],
                             **({"basis": collected_basis(r)} if collected_basis(r) else {"key": n["check"]["key"]})})
                checks += 1
            elif form == "pending":
                b.plot([x], [r["passes"]], marker="o", markersize=EXEC_PT + 1.0, markerfacecolor="white",
                       markeredgecolor=gray, markeredgewidth=0.9, zorder=6, clip_on=False)
                size = EXEC_PT + 1.9
            else:
                b.plot([x], [r["passes"]], marker="o", markersize=EXEC_PT + 0.4, color=blue, markeredgewidth=0,
                       zorder=6, clip_on=False)
                size = EXEC_PT + 0.4
            pts.append(to_pt(b, x, r["passes"]) + (size / 2.0,))
            place.add_mark(*pts[-1])
            if "correct" in r:
                keys.append(record_row(r, f"{n['label']} {r['budget']}: correct development-split images",
                                       f"{S.thousands(r['correct'])} of {S.thousands(r['images'])}",
                                       r["correct_source"]))
            keys.append(record_row(r, f"{n['label']} {r['budget']}: scheduled passes of the whole deployment"
                                      + (" (planned)" if form == "planned" else ""), r["passes"], r["passes_source"]))
            keys.append(record_row(r, f"{n['label']} {r['budget']}: declared core budget", r["cores"],
                                   r["cores_source"]))
            if form == "planned":
                keys.append(record_row(r, f"{n['label']} {r['budget']}: budget as a share of the resident core count",
                                       budget_label(r), r.get("share_source")))
            elif r.get("share_den") and form != "reference":
                keys.append({"what": f"{n['label']} {r['budget']}: budget as a share of the resident core count",
                             "value": budget_label(r), "source": r["share_source"],
                             **({"basis": collected_basis(r)} if collected_basis(r) else {"key": d["ladder"]["key"]})})
            elif not r.get("share_den") and form != "reference":
                keys.append({"what": f"{n['label']} {r['budget']}: the tick's name, the budget's name on the declared "
                                     "ladder", "value": r["budget"], "key": d["ladder"]["key"],
                             "source": d["ladder"]["source"]})
            if r["geometry"] != ref["geometry"]:
                keys.append(record_row(r, f"{n['label']} {r['budget']}: core geometry", r["geometry"],
                                       r["geometry_source"]))


        if not place.place(pts[0][0], pts[0][1], f"Single pass, {S.thousands(ref['cores'])} cores", FS["small"], pts[0][2],
                           ("E0",) + tuple(("E0", dx) for dx in RESIDENT_SLIDE), color=S.C["ink2"]):
            print(f"FAIL {NAME}: no free place for the resident core count of {n['label']}")


        done = [i for i, r in enumerate(rows) if status_of(r) not in ("planned", "reference")]
        fresh = [i for k, i in enumerate(done) if k == 0 or rows[i]["passes"] != rows[done[k - 1]]["passes"]]
        order = fresh + [len(rows) - 1, 1] + [i for i in range(3, len(rows))
                                              if rows[i - 1]["passes"] == rows[i - 2]["passes"] > 0
                                              and abs(rows[i]["passes"] - rows[i - 1]["passes"]) >= STEP_MIN * rows[i - 1]["passes"]]
        order += [i for i in done if i not in fresh]
        counted = []
        for i in dict.fromkeys(i for i in order if 0 < i < len(rows)):
            if any(abs(pts[i][0] - pts[k][0]) < COUNT_MIN_DX for k in counted):
                continue
            if place.place(pts[i][0], pts[i][1], S.thousands(rows[i]["passes"]), FS["label"], pts[i][2],
                           COUNT_PLACES, color=blue):
                counted.append(i)
        for i, r in enumerate(rows):
            if status_of(r) not in ("planned", "reference") and i not in counted:
                print(f"NOTE {NAME}: no free place for the pass count of {n['label']} {r['budget']} "
                      f"({r['passes']} passes, its check is drawn)")


        for i, r in enumerate(rows):
            if r["geometry"] != ref["geometry"] and status_of(r) != "planned":
                head = f"{budget_label(r)}: {S.thousands(r['cores'])} cores"
                notes_ = [f"{head}{join}{geometry_label(r['geometry'])}" for join in GEOMETRY_JOINS]
                count_w = S.text_width_pt(fig, S.thousands(r["passes"]), FS["label"])
                beside = (("NW", -(count_w + GEOMETRY_GAP)),)
                rest = tuple(("C0", dx) for dx in GEOMETRY_SLIDE) + ("C1", "N", "S")
                kw = dict(color=S.C["ink2"])


                bare = [f"{S.thousands(r['cores'])} cores{join}{geometry_label(r['geometry'])}" for join in GEOMETRY_JOINS]
                named_ok = (any(place.place(pts[i][0], pts[i][1], t, FS["small"], pts[i][2], beside, **kw)
                                for t in notes_)
                            or place.place(pts[i][0], pts[i][1], notes_[0], FS["small"], pts[i][2], rest, **kw))
                if not named_ok:
                    print(f"NOTE {NAME}: no free place for the name {budget_label(r)!r} in the geometry note of "
                          f"{n['label']}: the note gives the cores and geometry alone")
                if not (named_ok or any(place.place(pts[i][0], pts[i][1], t, FS["small"], pts[i][2], beside, **kw)
                                        for t in bare)
                        or place.place(pts[i][0], pts[i][1], bare[0], FS["small"], pts[i][2], rest, **kw)):
                    print(f"FAIL {NAME}: no free place for the geometry note of {n['label']} {r['budget']}")


        xs_pt = [p[0] for p in pts]
        named = tick_names(rows, xs_pt, fig)
        for i, r in enumerate(rows):
            if status_of(r) != "planned" and i not in named:
                print(f"NOTE {NAME}: no free place for the tick name of {n['label']} {r['budget']}")
        _panel.__exit__(None, None, None)
        for r in keys[k0:]:
            r["ref"] = f"n{j}"
        b.xaxis.set_major_locator(FixedLocator([xs[i] for i in named]))
        names = {xs[i]: budget_label(rows[i]) for i in named}
        b.xaxis.set_major_formatter(FuncFormatter(lambda v, pos=None, names=names: names.get(v, "")))
        b.xaxis.set_minor_locator(FixedLocator([x for i, x in enumerate(xs) if i not in named]))
        b.xaxis.set_minor_formatter(NullFormatter())
        b.tick_params(axis="x", which="major", labelsize=FS["tick"], pad=XTICK_PAD, length=XTICK_LEN)
        b.tick_params(axis="x", which="minor", length=TICK_MINOR)
        S.grid(b)
        b.yaxis.set_major_locator(FixedLocator([0] + [10 ** k for k in range(1, int(math.log10(max(p_hi, 10))) + 1)]))
        b.yaxis.set_minor_locator(NullLocator())
        b.yaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: S.thousands(v)))
        b.tick_params(axis="y", labelsize=FS["tick"], pad=TICK_PAD, length=TICK_LEN)
        if j:
            b.spines["left"].set_visible(False)
            b.tick_params(axis="y", which="both", left=False, labelleft=False)

    off = -(TICK_LEN + TICK_PAD + tick_w + YLAB_GAP) / pws[0]
    axes[0].set_ylabel("Scheduled\npasses", fontsize=FS["label"], linespacing=0.95)
    axes[0].yaxis.set_label_coords(off, 0.5)
    fig.text((left + plot_w / 2.0 + COL_GAP * (len(nets) - 1) / 2.0) / W, XLABEL_UP / HEIGHT_PT,
             "Core budget $B/M$", ha="center", va="bottom", fontsize=FS["label"])


    fs = FS["small"]
    y_band = band_pt / HEIGHT_PT
    y_mark = (band_pt + 0.36 * fs) / HEIGHT_PT
    x = W - RIGHT
    for form, text in reversed(KEY_FORMS):
        if form not in forms:
            continue
        w = S.text_width_pt(fig, text, fs)
        fig.text(x / W, y_band, text, ha="right", va="baseline", fontsize=fs, color=S.C["ink2"])
        xm = (x - w - 5.6) / W
        kw = dict(transform=fig.transFigure, linestyle="none", marker="o")
        if form == "checked":
            fig.add_artist(Line2D([xm], [y_mark], markersize=ZERO_PT, color=green, markeredgewidth=0, **kw))
            fig.add_artist(Line2D([xm], [y_mark], transform=fig.transFigure, linestyle="none", marker="$\\checkmark$",
                                  markersize=ZERO_PT * 0.6, color="white", markeredgewidth=0))
        elif form == "pending":
            fig.add_artist(Line2D([xm], [y_mark], markersize=EXEC_PT + 1.0, markerfacecolor="white",
                                  markeredgecolor=gray, markeredgewidth=0.9, **kw))
        else:
            fig.add_artist(Line2D([xm], [y_mark], markersize=PLAN_PT, markerfacecolor="white", markeredgecolor=blue,
                                  markeredgewidth=0.7, **kw))
        x -= w + 5.6 + 4.0 + 9.0


    cmp_ = d["compared"]
    if checks != cmp_["rows_drawn"]:
        print(f"FAIL {NAME}: {checks} check marks drawn, the data file's compared rows are {cmp_['rows_drawn']}")
    if "checked" in forms and cmp_["rows_drawn"]:
        per = {r["differing"] for r in cmp_["rows"]}
        if cmp_["outputs_per_row"] and len(per) == 1:
            count = f"{per.pop()} of {S.thousands(cmp_['outputs_per_row'])}"
            text = f"{count} differing output codes at each of {cmp_['rows_drawn']} executed budgets"


            words = f"Differing output codes at the executed budgets: {S.PENDING_TOKEN}"
            what = (f"output codes differing from the resident deployment on one compared budget row (the same "
                    f"count on each of the {cmp_['rows_drawn']} rows drawn with a check)")
        else:
            count = f"{S.thousands(cmp_['differing_drawn'])} of {S.thousands(cmp_['outputs_drawn'])}"
            text = f"{count} differing output codes over {cmp_['rows_drawn']} executed budgets"
            words = f"Differing output codes over the executed budgets: {S.PENDING_TOKEN}"
            what = (f"output codes differing from the resident deployment, summed over the {cmp_['rows_drawn']} "
                    "compared budget rows drawn with a check")
        with S.marks(fig, "focal", series="n0"):
            t = S.focal(axes[0], text, (lefts[0] / W, y_band), coords=fig.transFigure, arrow=False, color=green,
                        ha="left", va="baseline")
        t.set_fontsize(FS["focal"])
        S.relabel(fig, "focal", t, words)
        keys.append({"what": what, "value": count, "source": cmp_["outputs_per_row_source"],
                     "key": ", ".join(cmp_["keys"]), "ref": "focal"})
        keys.append({"what": "compared budget rows drawn with a check (executed budgets below the resident one)",
                     "value": cmp_["rows_drawn"],
                     "source": "data/fig_invariance.json#compared/rows_drawn: the rows with status equal, "
                               f"{cmp_.get('rows_ledger', cmp_['rows_drawn'])} under the check keys and "
                               f"{cmp_.get('rows_collected', 0)} in the collected table (not ledger keys yet)",
                     "key": ", ".join(cmp_["keys"]), "ref": "focal"})

    S.pending_key_at(fig, x + 4.0, band_pt + 0.36 * fs + 0.6, ha="right", size=fs)
    S.save_fig(fig, NAME, rows=keys)
    S.save_keys(NAME, keys)


if __name__ == "__main__":
    main()
