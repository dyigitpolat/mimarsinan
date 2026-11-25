""
import os
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

B = C.load_tree()
import rlib7 as L

OUT = sys.argv[1]
VAR = os.path.join(C.SCR, "variants")
p0_wo = float(B._HL["p0_chip_wide_without_link_mW"]) * 1e-3
P0S = OrderedDict(p_tile="nominal", p_global_W=p0_wo, p_dram_bit="nominal", store={}, link={})
ZS = OrderedDict(p_tile="zero", p_global_W=0.0, p_dram_bit="zero", store=dict(mode="none"), link=dict(mode="none"))
BOTH = ("composer", "l_ref")
jobs = []


def job(i, setting, over, spec, evc="main", kind="point", forms="ruled", detail=False, fronts=False, units=BOTH,
        swap=None, note="", **meta):
    jobs.append(OrderedDict(id=i, kind=kind, r5=dict(setting=setting, over=over), spec=spec, forms=forms,
                            detail=detail, fronts=fronts, units=list(units), swap=swap, evc=evc, note=note, **meta))


def S(**kw):
    s = OrderedDict(P0S)
    for k, v in kw.items():
        s[k] = v
    return s


for i, st, ov, nt in (("P0", "P0", {}, "THE POINT OF RECORD P0 (the router at the shorter of its measured cycle and "
                                       "the chip's)"),
                      ("hop1", "A", dict(t_hop_end="low"), "the router's cycle at the hop time's low end, 1 ns"),
                      ("hop2", "hop_2ns", {}, "the router's cycle at the declared sweep point, 2 ns"),
                      ("A", "A", {}, "THE NAMED ALTERNATIVE A, the router's cycle at the hop time's nominal, 4 ns"),
                      ("hop8", "A", dict(t_hop_end="high"), "the router's cycle at the hop time's high end, 8 ns")):
    job(i, st, ov, "P0", kind="full", detail=True, fronts=True, note=nt)

job("zero_ruled", "P0", {}, ZS, detail=True, fronts=True, note="P0's round-5 and round-7 items, no static power")
job("zero_record_forms", "P0", {}, ZS, forms="record", note="no static power, the record's forms (no working memory)")

for i, st, ov, evc, nt in (
        ("fill_residents_first", "fill_residents_first", {}, "resfirst", "the residents-first order"),
        ("carry_once", "carry_once", {}, "main", "the carried value without its legs (out and in once, at the "
                                                 "attachments)"),
        ("carry_copies", "carry_copies", {}, "main", "the carried value with its copies only"),
        ("compute_only_census", "compute_only_census", {}, "main", "the sixth build's compute-only figure"),
        ("r6_forms", "r6_forms", {}, "r6", "the sixth build's passes and floor (the gate's forms)"),
        ("attach_site", "attach_site", {}, "attach", "the memory ports at the median router"),
        ("attach_site_A", "A", dict(store_site="median"), "attach", "the memory ports at the median router, at A"),
        ("mesh_near_square", "P0", dict(mesh="near_square"), "mesh", "the near-square mesh with the plan's passes"),
        ("carry_once_A", "A", dict(carry_in="once"), "main", "at A, the carried value without its legs"),
        ("lane_pass", "lane_pass", {}, "main", "the toolchain's balanced pass, the lane's named sensitivity"),
        ("lane_columns", "lane_columns", {}, "main", "the lane block's reach of one lane per 64 columns"),
        ("chip_clock", "chip_clock", {}, "main", "the router at the chip's clock everywhere")):
    job(i, st, ov, "P0", evc=evc, kind="whole", note=nt)

over_sets = [("occupancy", "P0", dict(time_form="occupancy")),
             ("net_lower", "P0", dict(net_end="lower")), ("net_upper", "P0", dict(net_end="upper")),
             ("hop1_net_lower", "A", dict(t_hop_end="low", net_end="lower")),
             ("hop1_net_upper", "A", dict(t_hop_end="low", net_end="upper")),
             ("hop2_net_lower", "hop_2ns", dict(net_end="lower")),
             ("hop2_net_upper", "hop_2ns", dict(net_end="upper")),
             ("A_net_lower", "A", dict(net_end="lower")), ("A_net_upper", "A", dict(net_end="upper")),
             ("hop8_net_lower", "A", dict(t_hop_end="high", net_end="lower")),
             ("hop8_net_upper", "A", dict(t_hop_end="high", net_end="upper")),
             ("chip_net_lower", "chip_clock", dict(net_end="lower")),
             ("chip_net_upper", "chip_clock", dict(net_end="upper")),
             ("lane_low", "P0", dict(lane_end="low")), ("lane_high", "P0", dict(lane_end="high")),
             ("launch_record", "P0", dict(launch="record")),
             ("launch_low", "P0", dict(launch_end="low")), ("launch_high", "P0", dict(launch_end="high")),
             ("ebh_low", "P0", dict(e_byte_hop_end="low")), ("ebh_high", "P0", dict(e_byte_hop_end="high")),
             ("barrier_low", "P0", dict(barrier_end="low")), ("barrier_high", "P0", dict(barrier_end="high")),
             ("cp_low", "P0", dict(core_program_end="low")), ("cp_high", "P0", dict(core_program_end="high")),
             ("entries_low", "P0", dict(entries_end="low")), ("entries_high", "P0", dict(entries_end="high")),
             ("host_low", "P0", dict(host_end="low")), ("host_high", "P0", dict(host_end="high"))]
for nm, st, ov in over_sets:
    job(nm, st, ov, "P0", kind="whole", note=f"{st} with {ov}")
for nm, ov, evc in (("psum_low", dict(psum_end="low"), "psum_low"), ("psum_high", dict(psum_end="high"),
                                                                     "psum_high")):
    job(nm, "P0", ov, "P0", evc=evc, kind="whole", note=f"P0 with {ov} (the partial sums at 8 and 32 bits)")

for e in ("low", "nominal", "high"):
    job(f"link_L1_{e}", "P0", {}, S(link=dict(mode="state", state="L1", end=e)),
        note=f"the link in L1 for the whole period at L1's {e} figure (the mainline kernel's default host)")
job("link_L0_bound", "P0", {}, S(link=dict(mode="state", state="L0")),
    note="the link in L0 for the whole period at its stated bound (the vendor's default host, not priced)")
for t in ("25C", "75C", "max"):
    job(f"temp_{t}", "P0", {}, S(store=dict(temperature=t)), note=f"the store's rule at the board temperature {t}")
job("ports_parallel", "P0", {}, S(store=dict(wake="parallel")), note="the ports waking in parallel (10 us)")
job("gaps_per_pass", "P0", {}, S(store=dict(gap="per_pass")), note="one rest per pass of the batch")
job("pages_open", "P0", {}, S(store=dict(page="open")), note="an open-page controller (the band's high end)")
busy_var = os.path.join(VAR, "store_states_r3_busy_closed_floor.json")
sha_busy = L.write_variant(B.STATES_FILE, lambda t: t.__setitem__("busy_state", "standby with refresh, ports idle"),
                           busy_var)
job("busy_closed_floor", "P0", {}, S(), swap=dict(store=["joint", busy_var]),
    note="the busy state at the closed-page floor (standby with refresh)", variant_sha256=sha_busy)

for p in L.box_points(B):
    swap = None if p["table_swap"] is None else dict(store=list(p["table_swap"]))
    spec = OrderedDict((k, p[k]) for k in ("p_tile", "p_global_W", "p_dram_bit", "store", "link"))
    job(p["tag"].replace(" ", "_"), "P0", {}, spec, units=("composer",), swap=swap, case=p["case"],
        store_label=p["store_label"], chip_label=p["chip_label"], tile_label=p["tile_label"])
ids = [j["id"] for j in jobs]
if len(set(ids)) != len(ids):
    raise SystemExit("a job id is repeated")
for j in jobs:
    B.r5_spec(j["r5"]["setting"], **j["r5"]["over"])
    if j["spec"] != "P0":
        L.make_point(B, j["spec"], None)
plan = OrderedDict(jobs=jobs, chip_wide_axes=L.chip_wide_axes(B), p0_chip_wide_without_link_W=p0_wo,
                   point_P0_equal=bool(L.make_point(B, P0S, B.r5_spec("P0")) == B.point_P0(B.r5_spec("P0"))),
                   r34=L.R34, r34_result_sha256=L.R34_SHA, variants={busy_var: sha_busy})
print(len(jobs), "jobs", C.dump(plan, OUT), "make_point(P0S) == point_P0:", plan["point_P0_equal"])
