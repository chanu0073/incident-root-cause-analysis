"""Ranking metrics and honest data splits.

The task is a ranking: for each case, order its services from "most likely root cause" to least.
  AC@k   share of cases whose true root cause is in the top k (AC@1 = top-1 accuracy)
  Avg@5  mean of AC@1..AC@5 (the summary metric used by the RCAEval benchmark)
  MRR    mean of 1 / rank of the true root cause

Ties are counted pessimistically: if the true service shares its score with others, it gets the
worst of the tied positions (a constant score cannot look good by luck).

Splits: every (system, service, fault) pair was injected 3 times. The three repetitions are
near-duplicates, so they must always stay on the same side of a split (`grouped_folds`).
`leave_one_system_out` is the harder test: train on two systems, test on the third.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def true_ranks(df: pd.DataFrame, score: str = "score") -> pd.Series:
    """Rank of the true root cause in each case (1 = best), ties counted pessimistically."""
    r = df.groupby("case")[score].rank(ascending=False, method="max")
    return r[df["is_rc"].astype(bool)].set_axis(df.loc[df["is_rc"].astype(bool), "case"].to_numpy())


def rank_metrics(ranks: pd.Series) -> dict:
    ac = {f"AC@{k}": float((ranks <= k).mean()) for k in (1, 3, 5)}
    ac["Avg@5"] = float(np.mean([(ranks <= k).mean() for k in range(1, 6)]))
    ac["MRR"] = float((1 / ranks).mean())
    return ac


def evaluate(df: pd.DataFrame, score: str = "score", by_system: bool = True) -> pd.DataFrame:
    ranks = true_ranks(df, score)
    rows = {"all": rank_metrics(ranks)}
    if by_system:
        sys_of = df.drop_duplicates("case").set_index("case")["system"]
        for s, r in ranks.groupby(sys_of.reindex(ranks.index).to_numpy()):
            rows[s] = rank_metrics(r)
    return pd.DataFrame(rows).T


def bootstrap_diff(ranks_a: pd.Series, ranks_b: pd.Series, metric: str = "AC@1", n: int = 2000, seed: int = 0) -> tuple:
    """95% interval of metric(a) - metric(b) over resampled cases (same cases for both)."""
    common = ranks_a.index.intersection(ranks_b.index)
    a, b = ranks_a[common].to_numpy(), ranks_b[common].to_numpy()
    f = {"AC@1": lambda r: (r <= 1).mean(), "AC@3": lambda r: (r <= 3).mean(), "AC@5": lambda r: (r <= 5).mean(),
         "MRR": lambda r: (1 / r).mean(), "Avg@5": lambda r: np.mean([(r <= k).mean() for k in range(1, 6)])}[metric]
    rng = np.random.default_rng(seed)
    d = [f(a[i]) - f(b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return tuple(np.percentile(d, [2.5, 50, 97.5]).round(3))


def grouped_folds(cases: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Yield (train_cases, test_cases); repetitions of the same (system, service, fault) never split."""
    pairs = cases["pair"].unique()
    rng = np.random.default_rng(seed)
    order = {p: i for i, p in enumerate(rng.permutation(pairs))}          # shuffle pairs reproducibly
    g = cases["pair"].map(order).to_numpy()
    for tr, te in GroupKFold(n_splits=n_splits).split(cases, groups=g):
        yield cases["case"].iloc[tr].to_numpy(), cases["case"].iloc[te].to_numpy()


def leave_one_system_out(cases: pd.DataFrame):
    for s in sorted(cases["system"].unique()):
        yield s, cases.loc[cases["system"] != s, "case"].to_numpy(), cases.loc[cases["system"] == s, "case"].to_numpy()
