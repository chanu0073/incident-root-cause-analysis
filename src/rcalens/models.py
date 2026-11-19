"""Learned models: an XGBoost learning-to-rank model for the root-cause service, and a fault-type classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

FAULTS = ["cpu", "delay", "disk", "loss", "mem", "socket"]
META = {"case", "system", "service", "is_rc", "fault", "pair", "repetition", "root_cause_service", "n_services", "score"}


def gpu_device() -> str:
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=1).fit(np.random.rand(20, 2), np.random.randint(0, 2, 20))
        return "cuda"
    except Exception:
        return "cpu"


def feature_columns(df: pd.DataFrame, sources: tuple[str, ...] = ("m_", "l_", "t_"), ranks: bool = True) -> list[str]:
    cols = [c for c in df.columns if c not in META and c.startswith(sources)]
    return cols if ranks else [c for c in cols if not c.endswith("_rank")]


RANKER_PARAMS = dict(objective="rank:pairwise", learning_rate=0.05, n_estimators=300, max_depth=4, min_child_weight=1,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist")


def fit_ranker(train: pd.DataFrame, cols: list[str], device: str = "cpu", seed: int = 0, **params) -> xgb.XGBRanker:
    """Learning to rank: inside each case (= query), the root-cause service should score highest."""
    train = train.sort_values("case")
    qid = pd.factorize(train["case"])[0]
    m = xgb.XGBRanker(**{**RANKER_PARAMS, **params}, device=device, random_state=seed)
    m.fit(train[cols], train["is_rc"].astype(int), qid=qid)
    return m


def fault_frame(df: pd.DataFrame, score: str = "score") -> pd.DataFrame:
    """One row per case: the raw features of the service ranked first (+ the system it belongs to)."""
    top = df.sort_values(["case", score], ascending=[True, False]).drop_duplicates("case")
    keep = [c for c in df.columns if c.startswith(("m_", "l_", "t_")) and not c.endswith("_rank")]
    out = top[["case", "system", "fault"] + keep].reset_index(drop=True)
    for s in ["ob", "ss", "tt"]:
        out[f"sys_{s}"] = (out["system"] == s).astype(int)
    return out


def fit_fault_classifier(train: pd.DataFrame, device: str = "cpu", seed: int = 0) -> tuple[xgb.XGBClassifier, list[str]]:
    cols = [c for c in train.columns if c.startswith(("m_", "l_", "t_", "sys_"))]
    y = train["fault"].map({f: i for i, f in enumerate(FAULTS)})
    m = xgb.XGBClassifier(objective="multi:softprob", learning_rate=0.05, n_estimators=300, max_depth=3,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=device, random_state=seed)
    m.fit(train[cols], y)
    return m, cols
