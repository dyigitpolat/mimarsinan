""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S

S.FS["title"] = 7.5

from sciviz import (Anchor, Badge, Box, Captioned, Card, Chip, Column, Connect, Core, CoreChip,
                    Document, FixedSize, Icon, LayerNet, MiniRaster, NetLayer, Palette, Region, Row, SizeGroup,
                    SparkLine, Sparkline, Text, TextBlock)

NAME = "fig_toolchain"
D = S.load(NAME)
PIC = D["pictograms"]
BLUE, GRAY, INK, INK2 = S.ROLE["ours"], S.C["gray"], S.C["ink"], S.C["ink2"]
WHITE = Palette.literal(S.C["white"])
CLASS_COLOR = {k: S.ROLE[c["role"]] for k, c in D["classes"].items()}
LAYER_TINT = [S.tint(BLUE, a) for a in (0.0, 0.2, 0.4, 0.55)]
WIRE = INK2
ICON_BOX = 8.0


REGION_PAD_X = 4.0
FIT_WIDTH_PT = S.TEXT_PT - 1.35


STAIR_W, STAIR_H = 78.0, 23.0
ROW_GAP = 0.6


PASS_CORE = 7.0
FRAME_PAD_Y = 2.0
REGION_PAD_Y = 1.5
LEGEND_INSET, LEGEND_PAD = 9.0, 2.5
HEIGHT_CAP_PT = 112.14

OUT_GAP = 1.9
MAIN_GAP = 16.0


STRIP_NAME_LEAD = 1.06
PAGE_MAX_PT = 112.14


def disc(icon, color, size=ICON_BOX):
    ""
    return Badge(Icon(icon, size=size * 0.66, color=WHITE, stroke_width=2.6), color=color, size=size)


def title(text, color=INK, size="label"):
    return Text(text, size=size, weight="700", color=color)


def note(text, color=INK2, weight="400"):
    return Text(text, size="label", weight=weight, color=color)


def network():
    layers = [NetLayer("", n, LAYER_TINT[i % len(LAYER_TINT)]) for i, n in enumerate(PIC["network_layers"])]
    return LayerNet(layers, spacing=17.0, node=4.6, pitch=7.0)


def chip_description():
    ""
    n_rows, n_cols = PIC["chip_grid"]
    grid = CoreChip([Core("", GRAY) for _ in range(n_rows * n_cols)], columns=n_cols, core=10.0, gap=2.5, pad=2.5)
    lines = [Row(Box(width=5.0, height=5.0, fill=CLASS_COLOR[k], stroke="none", radius=1.0),
                 Box(width=float(n), height=2.6, fill=S.C["gray2"], stroke="none", radius=1.3),
                 gap=2.6, align="center")
             for k, n in PIC["property_bars"].items()]
    sheet = Document(Column(*lines, gap=3.0, align="start", share_columns=False), size="sm", fill=S.C["white"],
                     stroke=S.C["gray2"], padding="sm")
    return Row(grid, sheet, gap="sm", align="center")


inputs = Column(
    Anchor("network", Column(title(D["inputs"]["network"]["label"]), network(), gap="xs", align="start")),
    Anchor("chip", Column(title(D["inputs"]["chip"]["label"]), chip_description(), gap="xs", align="start")),
    gap="sm", align="start", justify="center")


def staircase():
    ""
    n = PIC["staircase_levels"]
    ramp = [(0.0, 0.0), (1.0, 1.0)]
    steps = []
    for i in range(n):
        y = (i + 0.5) / n
        steps += [(i / n, y), ((i + 1) / n, y)]
    plot = Sparkline([SparkLine(ramp, color=INK2, width=1.0), SparkLine(steps, color=CLASS_COLOR["comp"], width=1.7)],
                     width=STAIR_W, height=STAIR_H)
    before, after = D["stages"][0]["picture_labels"]
    return Column(Column(note(after, CLASS_COLOR["comp"], "700"), plot, gap="xs", align="start"),
                  note(before), gap="xs", align="end")


def passes():
    lines = []
    for i, row in enumerate(PIC["passes"]):
        layers = [j for j, on in enumerate(row) if on]
        cores = CoreChip([Core("", LAYER_TINT[j]) for j in layers], core=PASS_CORE, gap=2.2, pad=2.0)
        lines.append(Row(note(f"{D['stages'][1]['picture_labels'][0]} {i + 1}"), cores, gap="sm", align="center"))
    return Column(*lines, gap="xs", align="center")


def equal_strips():
    codes = PIC["strip_cells"]
    shade = [S.tint(INK2, a) for a in (0.82, 0.6, 0.35, 0.1)]

    def strip():
        return MiniRaster([[shade[c] for c in codes]], cell=6.5, gap=0.14)

    def name(text):
        ""
        return TextBlock("\n".join(text.rsplit(" ", 1)), size="label", weight="400", color=INK2, align="center",
                         line_spacing=STRIP_NAME_LEAD)
    ref, prog = D["stages"][2]["picture_labels"]
    return Column(name(ref), strip(), disc("check", S.ROLE["exact"], 11.0), strip(), name(prog),
                  gap="xs", align="center")


PICTURE = {"staircase": staircase, "passes": passes, "equal_strips": equal_strips}


def tag(k):
    ""
    c = CLASS_COLOR[k]
    return Chip(D["classes"][k]["tag"], color=c, fill=S.tint(c, 0.86), text_color=INK, text_size="tiny",
                text_weight="600")


def stage(s):
    picture = Column(PICTURE[s["picture"]](), grow=True, justify="center", align="center")
    parts = [picture]
    if s["tags"]:
        parts.append(Column(*[tag(k) for k in s["tags"]], gap="xs", align="center"))
    card = Card(title(s["label"]), Column(*parts, gap="xs", align="center"), role=S.C["gray3"],
                body_fill=S.C["white"], padding="sm")
    return Anchor(s["id"], card)


NAMES = SizeGroup("width", fit="slot", align="start")


def frame(label, body, focal=False):
    ""
    name = NAMES.add(TextBlock(label.replace(" ", "\n"), size="label", weight="700", color=BLUE if focal else INK,
                               line_spacing=1.2))
    return Region(Row(name, body, gap=3.0, align="center"), color=BLUE if focal else S.C["gray2"],
                  fill=S.tint(BLUE, 0.9) if focal else S.C["white"], dashed=False, pad_x=3.5, pad_y=FRAME_PAD_Y,
                  margin_x=0.0, margin_y=0.0)


def program():
    ""
    o = D["outputs"]["program"]
    groups = [CoreChip([Core("", LAYER_TINT[j]) for j, on in enumerate(row) if on], core=6.0, gap=2.0, pad=2.0)
              for row in PIC["passes"]]
    body = Row(Captioned(Row(*groups, gap="xs", align="center"), title=o["places"][0], title_size="label",
                         placement="bottom", gap=1.5),
               Captioned(Icon("cpu", size=12.0, color=INK2), title=o["places"][1], title_size="label",
                         placement="bottom", gap=1.5),
               gap=3.0, align="end")
    return frame(o["label"], body, focal=True)


def boxed(mark):
    return FixedSize(mark, width=ICON_BOX, height=ICON_BOX)


VERDICT_MARKS = [("check", BLUE, INK), ("check", INK2, INK2), ("x", GRAY, INK2)]


def verdict():
    ""
    v = D["outputs"]["verdict"]
    assert len(v["forms_printed"]) == len(v["forms"]) == len(VERDICT_MARKS) == len(v["form_basis"])
    marks = [Row(boxed(disc(icon, color)), note(form, ink), gap=2.5, align="center")
             for form, (icon, color, ink) in zip(v["forms_printed"], VERDICT_MARKS)]
    return frame(v["label"], Column(*marks, gap=ROW_GAP, align="start", share_columns=False))


def estimate():
    ""
    e = D["outputs"]["estimate"]
    cells = [Row(boxed(Icon(t["icon"], size=ICON_BOX, color=INK2)), note(t["label"]), gap=2.5, align="center")
             for t in e["terms"]]
    half = (len(cells) + 1) // 2
    grid = Row(Column(*cells[:half], gap=ROW_GAP, align="start", share_columns=False),
               Column(*cells[half:], gap=ROW_GAP, align="start", share_columns=False), gap=2.0, align="start")
    return frame(e["label"], grid)


OUT = SizeGroup("width")
outputs = Anchor("outputs", Column(*[Anchor(n, OUT.add(f())) for n, f in (("program", program),
                                                                         ("verdict", verdict),
                                                                         ("estimate", estimate))],
                                   gap=OUT_GAP, align="start", justify="between"))


st = [stage(s) for s in D["stages"]]
stages = Row(*st, gap="md", align="stretch", grow=True)


toolchain = Anchor("toolchain", Region(
    stages, label=D["toolchain_label"], label_style="tab", tab_style="legend", label_size="label",
    label_align="start", tab_inset=LEGEND_INSET, tab_pad=LEGEND_PAD, balance_label_padding=True,
    color=GRAY, fill=S.tint(S.C["gray3"], 0.55), dashed=False, pad_x=REGION_PAD_X, pad_y=REGION_PAD_Y,
    margin_x=0.0, margin_y=0.0))
body = Column(

    Row(inputs, Row(toolchain, outputs, gap=MAIN_GAP, align="center", equal_heights=True), gap=MAIN_GAP,
        align="center"),
    Connect(["network", "chip"], "toolchain", color=WIRE, orientation="horizontal"),
    Connect("conversion", "mapping", src_side="right", dst_side="left", color=WIRE),
    Connect("mapping", "check", src_side="right", dst_side="left", color=WIRE),
    Connect("toolchain", ["program", "verdict", "estimate"], color=WIRE, orientation="horizontal"),
    Connect("outputs", "mapping", src_side="bottom", dst_side="bottom", src_at="estimate", label=D["loop"]["label"],
            label_style="on-wire", color=BLUE, label_color=BLUE, dst_room=0.0),
    gap="xs")

KEYS = [{"what": f"input: {v['label']}", "value": v["quote"], "source": v["source"]} for v in D["inputs"].values()]
for s in D["stages"]:
    KEYS.append({"what": f"stage: {s['label']}", "value": s["quote"], "source": s["source"]})
    for k in s["tags"]:
        c = D["classes"][k]
        KEYS.append({"what": f"class tag on {s['label']}: {c['tag']} (the keys of the class)",
                     "value": c["keys"]["value"], "source": c["keys"]["source"]})
        for b in c["stage_basis"]:
            KEYS.append({"what": f"class tag on {s['label']}: {c['tag']} (the stage of the tag)",
                         "value": b["quote"], "source": b["source"]})
for v in D["outputs"].values():
    KEYS.append({"what": f"output: {v['label']}", "value": v["quote"], "source": v["source"]})

    for form, name, b in zip(v.get("forms", []), v.get("forms_printed", []), v.get("form_basis", [])):
        KEYS.append({"what": f"output: {v['label']}: outcome {name} (the frozen text's name: {form})",
                     "value": b["quote"], "source": b["source"]})
    if "forms_printed_basis" in v:
        KEYS.append({"what": f"output: {v['label']}: the printed outcome names ({', '.join(v['forms_printed'])})",
                     "value": v["forms_printed_basis"]["quote"], "source": v["forms_printed_basis"]["source"]})
KEYS.append({"what": f"loop: {D['loop']['label']}", "value": D["loop"]["quote"], "source": D["loop"]["source"]})

DIAGRAM = S.paper(body, FIT_WIDTH_PT, theme=S.THEME.with_overrides(connector_label_size="label"),
                  target_aspect=HEIGHT_CAP_PT / FIT_WIDTH_PT)
S.save_diagram(DIAGRAM, NAME, rows=KEYS)
S.save_keys(NAME, KEYS)
print(DIAGRAM.fit_report())
_page_h = S._pdf_page_pt(S.OUT / f"{NAME}.pdf")[1]
if _page_h > PAGE_MAX_PT + 0.005:
    print(f"FAIL {NAME}: the page is {_page_h:.2f} pt high, over the {PAGE_MAX_PT} pt limit")
