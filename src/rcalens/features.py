"""Per-service features for one failure case: "how differently did this service behave after the fault?"

For every candidate service of a case we compare the window AFTER the fault was injected with the
normal window BEFORE it (12 minutes each). Nothing here uses the label, and every statistic of the
normal behaviour comes from the pre-fault window of the same case.

  metrics  robust z-scores (median / IQR of the normal window, like BARO), mean shift, per metric type
  logs     log volume change, error-line change, share of never-seen-before log templates
  traces   span count change, latency (p50 / p95) change, worst operation latency change, error rate
  ranks    each feature's rank among the services of the same case (systems have 12-68 services,
           so "most anomalous in this case" is comparable across systems while raw values are not)

Service names are never used as features: the model must learn *behaviour*, not *which services
the benchmark happens to break*.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import METRIC_TYPES, Case, split_metric

Z_CAP = 1e4
ERROR_RE = r"(?i)\b(?:error|exception|fail(?:ed|ure)?|timeout|timed out|refused|unavailable|denied|5\d\d)\b"


def norm_name(s: str) -> str:
    """Make service names from metrics, logs and traces comparable ('frontendservice' == 'frontend')."""
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return s.replace("service", "")


def slog(x):
    """Signed log: keeps the direction, tames huge ratios."""
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------- metrics
def metric_features(case: Case) -> pd.DataFrame:
    m = case.metrics
    t = m["time"].to_numpy()
    pre, post = t < case.inject_time, t >= case.inject_time
    rows: dict[str, dict] = {}
    for col in m.columns:
        if col == "time":
            continue
        svc, typ = split_metric(col)
        if typ not in METRIC_TYPES:
            continue
        x = m[col].to_numpy(dtype=float)
        a, b = x[pre], x[post]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        med = np.median(a)
        q75, q25 = np.percentile(a, [75, 25])
        scale = q75 - q25
        if scale <= 0:
            scale = a.std()
        if scale <= 0:
            scale = 1e-3 * max(abs(med), 1e-3)
        z = np.clip((b - med) / scale, -Z_CAP, Z_CAP)
        r = rows.setdefault(svc, {})
        key = typ.replace("-", "")
        cand = {f"m_{key}_maxz": float(np.log1p(np.abs(z).max())),
                f"m_{key}_meanz": float(slog(z.mean())),
                f"m_{key}_rel": float(slog((b.mean() - a.mean()) / (abs(a.mean()) + 1e-9)))}
        for k, v in cand.items():                       # several columns of one type: keep the strongest
            if k not in r or abs(v) > abs(r[k]):
                r[k] = v
    df = pd.DataFrame.from_dict(rows, orient="index")
    maxz = [c for c in df.columns if c.endswith("_maxz")]
    df["m_all_maxz"] = df[maxz].max(axis=1)
    df["m_n_metrics"] = df[maxz].notna().sum(axis=1)
    return df


# ---------------------------------------------------------------- logs
def log_features(case: Case, services: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=services)
    lg = case.logs
    if lg is None or lg.empty:
        return out
    pre_s = max(case.inject_time - int(case.metrics["time"].min()), 1)
    post_s = max(int(case.metrics["time"].max()) - case.inject_time + 1, 1)
    ts = lg["timestamp"].astype("int64").to_numpy()
    is_post = ts >= case.inject_time
    msg = lg["message"].astype(str)
    is_err = msg.str.contains(ERROR_RE, regex=True).to_numpy()
    tmpl = msg.str.replace(r"[0-9a-fA-F]{8,}|\d+", "#", regex=True).str.slice(0, 120)
    d = pd.DataFrame({"svc": lg["container_name"].astype(str).map(norm_name).to_numpy(), "post": is_post, "err": is_err,
                      "tmpl": tmpl.to_numpy()})
    by = {norm_name(s): s for s in services}
    res = {}
    for key, g in d.groupby("svc", sort=False):
        if key not in by:
            continue
        pre, post = g[~g["post"]], g[g["post"]]
        seen = set(pre["tmpl"])
        res[by[key]] = {
            "l_rate_ratio": float(np.log((len(post) / post_s + 0.1) / (len(pre) / pre_s + 0.1))),
            "l_err_ratio": float(np.log((post["err"].sum() / post_s + 0.01) / (pre["err"].sum() / pre_s + 0.01))),
            "l_err_share_post": float(post["err"].mean()) if len(post) else 0.0,
            "l_new_tmpl_share": float((~post["tmpl"].isin(seen)).mean()) if len(post) else 0.0,
        }
    return out.join(pd.DataFrame.from_dict(res, orient="index"))


# ---------------------------------------------------------------- traces
def trace_features(case: Case, services: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=services)
    tr = case.traces
    if tr is None or tr.empty:
        return out
    t = tr["startTimeMillis"].astype("int64").to_numpy() / 1000.0
    d = pd.DataFrame({"svc": tr["serviceName"].astype(str).map(norm_name).to_numpy(),
                      "op": tr["operationName"].astype(str).to_numpy(),
                      "dur": tr["duration"].astype("float64").to_numpy(),
                      "err": (tr["statusCode"].fillna(0).astype("int64").to_numpy() != 0),
                      "post": t >= case.inject_time})
    by = {norm_name(s): s for s in services}
    res = {}
    for key, g in d.groupby("svc", sort=False):
        if key not in by:
            continue
        pre, post = g[~g["post"]], g[g["post"]]
        if len(pre) < 20 or len(post) < 20:
            res[by[key]] = {"t_count_ratio": float(np.log((len(post) + 1) / (len(pre) + 1)))}
            continue
        q = lambda s, p: float(np.percentile(s, p))  # noqa: E731
        ops = g.groupby(["op", "post"])["dur"].quantile(0.95).unstack()
        ops = ops.dropna()
        worst = float(np.log((ops[True] + 1) / (ops[False] + 1)).max()) if len(ops) else np.nan
        res[by[key]] = {
            "t_count_ratio": float(np.log((len(post) + 1) / (len(pre) + 1))),
            "t_p50_ratio": float(np.log((q(post["dur"], 50) + 1) / (q(pre["dur"], 50) + 1))),
            "t_p95_ratio": float(np.log((q(post["dur"], 95) + 1) / (q(pre["dur"], 95) + 1))),
            "t_worst_op_p95_ratio": worst,
            "t_err_rate_diff": float(post["err"].mean() - pre["err"].mean()),
        }
    return out.join(pd.DataFrame.from_dict(res, orient="index"))


# ---------------------------------------------------------------- one case
def case_features(case: Case) -> pd.DataFrame:
    """One row per candidate service (= every service that has metrics in this case)."""
    mf = metric_features(case)
    services = list(mf.index)
    df = mf.join(log_features(case, services)).join(trace_features(case, services))
    df.index.name = "service"
    base = [c for c in df.columns if c != "m_n_metrics"]
    for c in base:                                   # 1.0 = most extreme service of this case
        df[c + "_rank"] = df[c].abs().rank(pct=True, ascending=True)
    df["n_services"] = len(df)
    return df.reset_index()
