""

from __future__ import annotations

import math


def t_cdf(t: float, df: int) -> float:
    ""
    if int(df) != df or df < 1:
        raise ValueError(f"df must be a positive integer, got {df}")
    df = int(df)
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    th = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(th), math.cos(th)
    if df % 2 == 1:
        if df == 1:
            a = 2.0 * th / math.pi
        else:
            term = acc = c
            for k in range(3, df - 1, 2):
                term *= c * c * (k - 1) / k
                acc += term
            a = 2.0 / math.pi * (th + s * acc)
    else:
        term = acc = 1.0
        for k in range(2, df - 1, 2):
            term *= c * c * (k - 1) / k
            acc += term
        a = s * acc
    a = min(1.0, max(0.0, a))
    return 0.5 * (1.0 + a) if t >= 0 else 0.5 * (1.0 - a)


def t_ppf(p: float, df: int) -> float:
    ""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def paired(d_images: list, total: int = 5000, conf: float = 0.95, mu0_images: float = 0.0) -> dict:
    ""
    d = [int(x) for x in d_images]
    if any(int(x) != x for x in d_images):
        raise ValueError("differences must be integer image counts")
    n = len(d)
    if n < 2:
        raise ValueError("a paired series needs at least two seeds")
    s = sum(d)
    mean = s / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    tq = t_ppf(conf, df)
    lower = mean - tq * se
    if se == 0.0:
        t_stat = math.inf if mean > mu0_images else (0.0 if mean == mu0_images else -math.inf)
        p = 0.0 if mean > mu0_images else 1.0
    else:
        t_stat = (mean - mu0_images) / se
        p = 1.0 - t_cdf(t_stat, df)
    pp = 100.0 / total
    return {"n": n, "df": df, "d_images": d, "sum_images": s, "mean_images": mean, "mean_pp": mean * pp,
            "sd_images": sd, "sd_pp": sd * pp, "se_images": se, "se_pp": se * pp, "conf": conf, "t_quantile": tq,
            "lower_images": lower, "lower_pp": lower * pp, "mu0_images": mu0_images, "t_stat": t_stat, "p": p,
            "min_images": min(d), "max_images": max(d)}


def holm(pvals: dict, alpha: float) -> dict:
    ""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    out, running = {}, 0.0
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * float(pvals[k]))
        running = max(running, adj)
        out[k] = {"p": float(pvals[k]), "p_holm": running, "rank": i + 1, "multiplier": m - i,
                  "level": alpha / (m - i), "rejected": running <= alpha}
    return out


def pooled_sd(series: dict) -> dict:
    ""
    num, df, used = 0.0, 0, []
    for k, xs in series.items():
        xs = [int(x) for x in xs]
        if len(xs) < 2:
            continue
        m = sum(xs) / len(xs)
        num += sum((x - m) ** 2 for x in xs)
        df += len(xs) - 1
        used.append(k)
    if df == 0:
        return {"sd_images": None, "df": 0, "members": used}
    return {"sd_images": math.sqrt(num / df), "df": df, "members": used}


def pooled_lower(mean_images: float, n: int, pooled: dict, conf: float = 0.95) -> dict | None:
    ""
    if not pooled.get("sd_images") or not pooled.get("df"):
        return None
    sd_d = math.sqrt(2.0) * pooled["sd_images"]
    tq = t_ppf(conf, pooled["df"])
    se = sd_d / math.sqrt(n)
    return {"sd_diff_images": sd_d, "df": pooled["df"], "t_quantile": tq, "se_images": se,
            "lower_images": mean_images - tq * se}
