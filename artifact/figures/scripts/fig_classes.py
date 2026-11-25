""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
S.MIN_ARM_PT = S.JOIN_TEST_PT
S.join_theme()
S.FS["title"] = 7.5

from sciviz import (Anchor, Badge, Box, Chip, Column, Connect, Icon, MiniRaster, Row, SizeGroup, Span,
                    Spacer, Table, TableSpan, Text, ThinArrow)

NAME = "fig_classes"
D = S.load(NAME)


ICON = {"placement": "grid", "splitting": "scissor", "passes": "layers", "resident": "cpu",
        "batch": "file-stack", "geometry": "aspect-ratio",
        "grid": "grid-02", "converter": "stairs-01", "neuron": "pulse-01",
        "endurance": "edit-02", "buffer": "inbox"}
EFFECT_ICON = {"cost": "gauge", "adm": "filter"}
CELL = 4.0
MARK_CELLS = 3
TURN = "corner-down-right"


ROW_GAP_A, ROW_GAP_B, PANEL_GAP = 0.6, 0.8, 2.74
HEAD_GAP = 1.0
INK, INK2 = S.C["ink"], S.C["ink2"]
COST = S.ROLE["cost_only"]
LABEL_PT, SMALL_PT = 7.0, 6.5
LABEL = dict(size=LABEL_PT, color=INK)
SMALL = dict(size=SMALL_PT)
ICON_PT = 1.2 * LABEL_PT


def strip(cells, color, cell=CELL):
    ""
    return MiniRaster([[color if i % 2 == 0 else S.tint(color, 0.62) for i in range(cells)]], cell=cell)


def effect_mark(cid, color):
    ""
    if cid in EFFECT_ICON:
        return Icon(EFFECT_ICON[cid], size=ICON_PT, color=color)
    return strip(MARK_CELLS, color, cell=CELL - 0.8)


def band(*cells):
    ""
    line = SizeGroup("height", fit="slot", align="baseline")
    return [TableSpan(line(c.content), c.columns, align=c.align) if isinstance(c, TableSpan)
            else c if isinstance(c, str) else line(c) for c in cells]


pictures = SizeGroup("both", fit="slot", align="center")
marks = SizeGroup("both", fit="slot", align="center")
counts = {"cost": D["counts"]["costonlyrows"]["display"]}
classes = D["classes"]
drawn = [[k for k in c["keys"] if k["drawn"]] for c in classes]


def head(c):
    color = S.ROLE[c["role"]]
    pill = Chip(c["name"], color=color, fill=color, text_color="white", text_size=LABEL_PT, text_weight="700")
    count = f"{counts[c['id']]}" if c["id"] in counts else ""
    if count and PEND["a/count"]:
        tail = (Text(count, color=S.PENDING_STYLE["ink"], **SMALL) if GRAY
                else S.sv_blank(Text(count, color=color, **SMALL),
                                stand_in=S.sv_tag("label", size=SMALL_PT, align="start",
                                                  width=Text(count, **SMALL).measure(S.THEME).w)))
        return Row(pill, tail, gap="xs", align="center")
    return Row(pill, Text(count, color=color, **SMALL), gap="xs", align="center")


def key(c, k):
    return Row(pictures(Icon(ICON[k["picture"]], size=ICON_PT, color=S.ROLE[c["role"]])), Text(k["label"], **LABEL),
               gap="xs", align="center")


def effect(c):
    color = S.ROLE[c["role"]]
    return Row(pictures(Icon(TURN, size=ICON_PT, color=color, stroke_width=2.0)),
               Text(c["effect"], size=LABEL_PT, weight="700", color=color), marks(effect_mark(c["id"], color)),
               gap="xs", align="center")


pair = D["pair"]
deps = pair["deployments"]
single = next(d for d in deps if d["passes"].get("label"))
multi = next(d for d in deps if not d["passes"].get("label"))
N_PASS = multi["passes"]["value"]


keys = []
for c, ks in zip(classes, drawn):
    for k in ks:
        keys.append({"what": f"{c['name']} key: {k['label']}", "value": k["wording"], "source": k["source"], "ref": None})
keys.append(S.category({"what": "network of the pair (panel b title)", "value": pair["network"]["display"],
                         "key": pair["network"]["key"], "source": pair["network"]["source"]}))
keys.append({"what": "the pair's rows and values: read by the row reader of fig_invariance (the same ledger keys "
                     "and record fields)", "value": [dep["row"] for dep in deps], "key": pair["key"]["key"],
             "source": pair["read_by"], "ref": "b"})
for dep in deps:
    keys.append({"what": f"declared cores, {dep['budget']}", "value": dep["cores"]["value"],
                 "key": dep["cores"]["key"], "source": dep["cores"]["source"], "ref": "b"})
    ps = dep["passes"]
    if ps.get("label"):
        keys.append(S.category({"what": f"passes, {dep['budget']}: the label \"{ps['label']}\", no count printed "
                                        f"({ps['measure']}: {ps['value']})",
                                "value": ps["label"], "key": ps["key"], "source": ps["label_basis"]}))
    else:
        keys.append({"what": f"passes, {dep['budget']} ({ps['measure']})", "value": ps["value"], "key": ps["key"],
                     "source": ps["source"], "ref": "b"})
r = pair["ratios"]["cores"]
keys.append({"what": f"\"{r['display']}× fewer cores\": ratio of declared cores between the two settings (derived "
                     f"in build_fig_classes_data.py)", "value": r["value"],
             "key": f"{deps[0]['cores']['key']}, {r['key']}",
             "source": r["source"], "ref": "b"})
keys.append({"what": "keys that change only cost", "value": D["counts"]["costonlyrows"]["display"],
             "key": D["counts"]["costonlyrows"]["key"], "ref": "a/count"})
keys.append({"what": "compared output codes", "value": pair["outputs"]["value"], "key": pair["outputs"]["key"],
             "source": pair["outputs"]["source"], "ref": "b"})
keys.append({"what": "differing output codes", "value": pair["outputs_differ"]["value"],
             "key": pair["outputs_differ"]["key"], "source": pair["outputs_differ"]["source"], "ref": "b"})
keys.append({"what": "strip mark beside Outputs (panel a)", "value": None,
             "source": "pictogram, no value: an alternating pattern of drawn cells", "ref": None})
keys.append({"what": "cores blocks (panel b): one pass wide, heights to scale across the pair",
             "value": [dep["cores"]["value"] for dep in deps], "key": pair["key"]["key"],
             "source": "; ".join(dep["cores"]["source"] for dep in deps), "ref": "b"})
sch = multi["schedule"]
keys.append({"what": f"pass blocks and pass numbers 1 to {N_PASS} (panel b): one block per scheduled pass, every "
                     f"pass drawn as the declared cores ({sch['per_pass_cores']['note']})",
             "value": N_PASS, "key": multi["passes"]["key"], "source": multi["passes"]["source"], "ref": "b"})
keys.append({"what": "weight-write marks (panel b): a pictogram, one between two successive passes; the record's "
                     "passes that rewrite weights", "value": sch["weight_write_passes"]["value"],
             "source": sch["weight_write_passes"]["source"], "ref": "b"})
keys.append({"what": "the mark \"Equal outputs\" (panel b): the form and words of Fig. 1(b), drawn because the "
                     "differing output codes above are 0", "value": pair["outputs_differ"]["value"],
             "key": pair["outputs_differ"]["key"], "source": pair["outputs_differ"]["source"], "ref": "b"})
for row in keys:
    S.ask(row)
PEND = {ref: any(r["state"] == S.FAMILY.PENDING for r in keys if r.get("ref") == ref)
        for ref in ("a/count", "b", "b/title")}
GRAY = S.release_values_drawn()
TAG_B = PEND["b"] and not GRAY


def pend_b(element, gray_element=None):
    ""
    if not PEND["b"]:
        return element
    return gray_element if GRAY else S.sv_blank(element)


UNIT = 2.8
PASS_W = 6.0
GUTTER = 3.8
BLOCK_R = 0.8
WRITE_W = 5.28
WRITE = S.ROLE["writes"]


rows_a = [band(*[head(c) for c in classes])]
for i in range(max(len(ks) for ks in drawn)):
    rows_a.append(band(*[key(c, ks[i]) if i < len(ks) else "" for c, ks in zip(classes, drawn)]))
rows_a.append(band(*[effect(c) for c in classes]))
panel_a = Table(rows_a, col_align=("start",) * len(classes), gap_x="md", gap_y=ROW_GAP_A)


pair = D["pair"]
deps = pair["deployments"]
single = next(d for d in deps if d["passes"].get("label"))
multi = next(d for d in deps if not d["passes"].get("label"))
ratio = pair["ratios"]["cores"]
N_PASS = multi["passes"]["value"]
def cores_block(d):
    ""
    h = UNIT * d["cores"]["value"] / multi["cores"]["value"]
    return pend_b(Box(width=PASS_W, height=h, fill=COST, stroke="none", radius=BLOCK_R),
                  Box(width=PASS_W, height=h, fill=S.PENDING_STYLE["fill"], stroke=S.PENDING_STYLE["edge"],
                      stroke_width=0.6, radius=BLOCK_R))


def write_mark(key=False):
    ""
    mark = Icon("arrow-right", size=WRITE_W, color=WRITE, stroke_width=3.0)
    return mark if key else pend_b(mark, Icon("arrow-right", size=WRITE_W, color=S.PENDING_STYLE["edge"],
                                              stroke_width=3.0))


def b_text(text, **kw):
    ""
    return pend_b(Text(text, **kw), Text(text, **dict(kw, color=S.PENDING_STYLE["ink"])))


def axis_name(text, words):
    ""
    if not PEND["b"]:
        return Text(text, **LABEL)
    if GRAY:
        return Text(text, **dict(LABEL, color=S.PENDING_STYLE["ink"]))
    from sciviz import FixedSize
    return S.sv_blank(Text(text, **LABEL), stand_in=FixedSize(Text(words, **LABEL),
                                                            width=Text(text, **LABEL).measure(S.THEME).w, align="end"))


def cores_label(d):
    return b_text(f"{S.thousands(d['cores']['value'])} cores", **LABEL)


passes = []
for q in range(N_PASS):
    block = cores_block(multi)
    passes.append(Anchor("multi", block) if q == N_PASS - 1 else block)
    if q + 1 < N_PASS:
        passes.append(write_mark())
numbers = [b_text(str(q // 2 + 1), color=INK2, **SMALL) if q % 2 == 0 else "" for q in range(len(passes))]
focal = Text(f"{ratio['display']}× fewer cores", size=S.FS["focal"], weight="700", color=COST)
if PEND["b"]:
    focal = (Text(f"{ratio['display']}× fewer cores", size=S.FS["focal"], color=S.PENDING_STYLE["ink"]) if GRAY
             else S.sv_blank(focal, stand_in=S.sv_tag("bar")))
write_key = Row(write_mark(key=True), Text("Weight write", color=INK2, **SMALL), gap=1.5, align="center")

schedule = Table(
    [[Column(cores_label(single), Text(single["passes"]["label"], **LABEL), gap=0.0, align="end"), Spacer(GUTTER, 0),
      TableSpan(Row(Anchor("single", cores_block(single)), focal, gap=6.0, align="end"),
                columns=len(passes), align="start")],
     [cores_label(multi), Spacer(GUTTER, 0), *passes],
     [axis_name(f"{N_PASS} passes", "Passes"), Spacer(GUTTER, 0), *numbers]],
    col_align=("end", "start") + ("center",) * len(passes), gap_x=0.0, gap_y=ROW_GAP_B)

same = Row(Anchor("same", Row(Spacer(S.JOIN_PAD, 0),
                              pend_b(Badge(Icon("check", color="white", size=8.0, stroke_width=3.0),
                                           color=S.C["green"], size=11.0, label_pad=3.4),
                                     Badge(Icon("check", color=S.PENDING_STYLE["edge"], size=8.0, stroke_width=3.0),
                                           color=S.PENDING_STYLE["fill"], size=11.0, label_pad=3.4, bordered=True,
                                           stroke_color=S.PENDING_STYLE["edge"])), gap=0.0, align="center")),
           Column(b_text("Equal outputs", **LABEL),
                  b_text(f"{S.thousands(pair['outputs_differ']['value'])} of {S.thousands(pair['outputs']['value'])}",
                         weight="700", color=INK, **SMALL),
                  b_text("differing output codes", color=INK, **SMALL), gap=0.0, align="start"),
           gap="xs", align="center")


ARM = S.JOIN_ARM
JOIN_TAP = 6.5
join = Connect(["single", "multi"], "same", orientation="horizontal", color=S.sv_ink(INK2, PEND["b"]),
               src_room=0.0, dst_room=0.0, min_shaft=S.JOIN_MIN_SHAFT, corner_radius=S.JOIN["radius"],
               flush_head=True)

panel_b = Row(schedule, same, *([] if TAG_B else [join]), gap=JOIN_TAP + ARM, align="center")
if multi["schedule"]["per_pass_cores"]["value"] is not None:
    print(f"FAIL {NAME}: the record states the cores of each pass; panel (b) draws every pass as the declared cores")


def panel(tag, title, child, key=None):
    ""
    head = Text(f"({tag}) {title}", size=S.FS["title"], weight="700", color=INK)
    if key is not None:
        head = Row(head, key, gap="sm", align="center", justify="between")
    return Column(head, child, gap=HEAD_GAP, align="stretch")


body = Column(panel("a", "Parameter classes", panel_a),
              panel("b", "Two cost-only settings" if PEND["b/title"] else
                    f"Two cost-only settings of {pair['network']['display']}", panel_b, key=write_key),
              gap=PANEL_GAP, align="stretch")
body.inflate_to(S.COLUMN_PT - 6.0)


d = S.paper(body, math.floor(S.COLUMN_PT * 100) / 100, min_effective_font_pt=min(LABEL_PT, SMALL_PT))
print("natural size:", d.measure())

S.save_diagram(d, NAME, rows=keys)
S.save_keys(NAME, keys)
