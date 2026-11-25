""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


A_L_MM2 = 66.528
CAP_TOP_MM2 = 0.5 * A_L_MM2
CAP_R1L_MM2 = 0.25 * A_L_MM2
AREA_TOL_MM2 = 1e-9
BUFFER_BYTES = 32768
BUFFER_UM2_PER_BYTE = 8.29840087890625
BUFFER_UM2 = BUFFER_BYTES * BUFFER_UM2_PER_BYTE
CORES_PER_NOC_TILE = 4
COLUMNS_PER_LANE = 64
LANE_UM2 = (1.13 - 1.0) * 0.1155 * 1e6
ROUTER_UM2 = 22308.6
E_MAC_STUDY_PJ = 0.15873015873015872
MAIN_JS = tuple(range(9))
RP_JS = (0, 2, 4, 6, 8)
RUNG_SETS = {"main": MAIN_JS, "A_rp_s": RP_JS, "A_rp": RP_JS}

READING_RUNGS = {"main": "main", "T_rs": "main", "E_ops": "main", "E_rp2": "main", "E_RC": "main",
                 "router_per_bit": "main", "lane_prorated": "main", "A_rp_s": "A_rp_s", "A_rp": "A_rp"}
BUFFERED_ARMS = ("pipe+bufS+P+B",)
MJ_PER_PJ = 1e-9


def limit_mm2(j: int) -> float:
    ""
    return CAP_TOP_MM2 * 1e6 * 2 ** (-j / 4) / 1e6


def noc_tiles(n: int) -> int:
    return -(-int(n) // CORES_PER_NOC_TILE)


def area_form_um2(n: int, core_um2: float, noc_um2: float, buffered: bool) -> float:
    ""
    return n * core_um2 + noc_tiles(n) * noc_um2 + (BUFFER_UM2 if buffered else 0.0)


def rung_brute(limit_um2: float, core_um2: float, noc_um2: float) -> int:
    ""
    n = 0
    while area_form_um2(n + 1, core_um2, noc_um2, True) <= limit_um2:
        n += 1
    return n


def rung_set_brute(core_um2: float, noc_um2: float, js: Sequence[int]) -> Dict[int, int]:
    return {j: rung_brute(CAP_TOP_MM2 * 1e6 * 2 ** (-j / 4), core_um2, noc_um2) for j in js}


def row_slot_increment_um2(s: int, i0: float, i1: float) -> float:
    ""
    return i0 + i1 * s


def reading_core_add_um2(reading: str, R: int, C: int, s: int, consts: Mapping[str, float]) -> float:
    ""
    if reading == "A_rp_s":
        return (R / 8 - 1) * C * row_slot_increment_um2(s, consts["i0_um2"], consts["i1_um2_per_bit"])
    if reading == "A_rp":
        return (R / 8 - 1) * C * consts["inc_um2"]
    return 0.0


def rung_tags(geometry: Mapping[str, Any], n: int) -> Dict[str, List[int]]:
    ""
    main = [j for j, c in enumerate(geometry["ladder_main_by_j"]) if c == n]
    rps = [int(j) for j, c in sorted(geometry["ladder_A_rp_s_by_j"].items(), key=lambda kv: int(kv[0])) if c == n]
    rp = [int(j) for j, c in sorted(geometry["ladder_A_rp_by_j"].items(), key=lambda kv: int(kv[0])) if c == n]
    return {"main": main, "A_rp_s": rps, "A_rp": rp}


def plane(counts_areas: Iterable[Tuple[int, Mapping[str, List[int]], float]], rung_set: str,
          cap_mm2: float) -> List[int]:
    ""
    return sorted(n for n, tags, area in counts_areas if tags.get(rung_set) and area <= cap_mm2 + AREA_TOL_MM2)


def _add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _const(v: float) -> List[float]:
    return [v, v, v]


def rho_w(s: int, a_pj: float, b_pj: float) -> float:
    ""
    return (a_pj + b_pj * s) / (a_pj + 4 * b_pj)


def reading_area_mm2(reading: str, d: Mapping[str, Any], consts: Mapping[str, float]) -> List[float]:
    ""
    area = list(d["design_area_mm2"])
    if reading in ("A_rp_s", "A_rp"):
        add = d["cores_physical"] * reading_core_add_um2(reading, d["R"], d["C_built"], d["s"], consts) * 1e-6
        return _add(area, _const(add))
    if reading == "router_per_bit":
        add = d["tiles"] * ROUTER_UM2 * (d["R"] * d["C_built"] * d["s"] / 65536 - 1) * 1e-6
        return _add(area, _const(add))
    if reading == "lane_prorated":
        lanes = d["lanes_per_core"] * CORES_PER_NOC_TILE * d["tiles"]
        frac = min(d["weight_columns"], COLUMNS_PER_LANE * d["lanes_per_core"]) / (COLUMNS_PER_LANE * d["lanes_per_core"])
        return _add(area, _const(-lanes * LANE_UM2 * (1 - frac) * 1e-6))
    return area


def reading_time_s(reading: str, d: Mapping[str, Any]) -> List[float]:
    if reading == "T_rs":
        if d.get("write_overlap_mode", "serial") != "serial":
            raise ValueError("REFUSED: T_RS_OVERLAP: the design's writes are overlapped, so T_rs is not exact")
        f = d["R"] / 256 - 1
        return [t + c * f for t, c in zip(d["time_s"], d["compute_time_s"])]
    return list(d["time_s"])


def reading_energy_mj(reading: str, d: Mapping[str, Any], consts: Mapping[str, Any]) -> List[float]:
    ""
    e = list(d["energy_mj"])
    k, s = d["cells_per_weight"], d["s"]
    if reading == "T_rs":
        if d.get("write_overlap_mode", "serial") != "serial":
            raise ValueError("REFUSED: T_RS_OVERLAP: the design's writes are overlapped, so T_rs is not exact")
        f = d["R"] / 256 - 1
        return [x + p * c * f * 1e3 for x, p, c in zip(e, d["static_power_w"], d["compute_time_s"])]
    if reading == "E_ops":
        dr = consts["rho_w"][str(s)]
        add = d["synaptic_events"] * E_MAC_STUDY_PJ * k * (s / 4 - dr) * MJ_PER_PJ
        return _add(e, _const(add))
    if reading == "E_rp2":
        add = d["synaptic_events"] * E_MAC_STUDY_PJ * k * (consts["rho_w2"][str(s)] - consts["rho_w"][str(s)]) * MJ_PER_PJ
        return _add(e, _const(add))
    if reading == "E_RC":
        add = d["synaptic_events"] * d["e_mac_pj"] * (consts["rho_r"][str(d["R"])] - 1) * MJ_PER_PJ
        return _add(e, _const(add))
    return e


def refusal_name(mark: Any) -> str:
    ""
    if isinstance(mark, Mapping):
        if isinstance(mark.get("name"), str) and mark["name"]:
            return mark["name"]
        inner = mark.get("refusal")
        if isinstance(inner, Mapping) and isinstance(inner.get("name"), str) and inner["name"]:
            return inner["name"]
    raise ValueError(f"REFUSED: REFUSAL_SHAPE: a mark of neither shape (refusal.name, refusal.refusal.name): "
                     f"{str(mark)[:300]}")


def matched_T(J: Sequence[Optional[float]], S: Sequence[Optional[float]]) -> Optional[float]:
    ""
    if any(j is None for j in J) or any(s is None for s in S):
        return None
    return min((s - j) / j for j, s in zip(J, S))


def margin_p(T_obs: Optional[float], beyond_obs: bool, draws: Sequence[Optional[float]]) -> Dict[str, Any]:
    ""
    undetermined = sum(1 for t in draws if t is None)
    if T_obs is None or not beyond_obs:
        return {"p": 1.0, "undetermined_draws": undetermined, "hits": None}
    hits = sum(1 for t in draws if t is None or t >= T_obs)
    return {"p": (1 + hits) / (len(draws) + 1), "undetermined_draws": undetermined, "hits": hits}


def bh_adjusted(pvals: Sequence[float]) -> List[float]:
    ""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


def definitive(pvals: Sequence[float], beyond: Sequence[bool], q: float = 0.10) -> List[bool]:
    adj = bh_adjusted(pvals)
    return [b and a <= q for a, b in zip(adj, beyond)]


def quantile_nearest_rank(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def chance_count(beyond_by_draw: Sequence[Sequence[Optional[bool]]]) -> Dict[str, Any]:
    ""
    n_d = [sum(1 for b in row if b is None or b) for row in beyond_by_draw]
    n_d_other = [sum(1 for b in row if b) for row in beyond_by_draw]
    und = [sum(1 for b in row if b is None) for row in beyond_by_draw]
    nd = len(n_d)
    return {"E0": sum(n_d) / nd, "q95": quantile_nearest_rank(n_d, 0.95),
            "beside_undetermined_not_beyond": {"E0": sum(n_d_other) / nd, "q95": quantile_nearest_rank(n_d_other, 0.95)},
            "mean_undetermined_per_draw": sum(und) / nd}
