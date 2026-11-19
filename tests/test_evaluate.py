import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcalens.evaluate import grouped_folds, rank_metrics, true_ranks  # noqa: E402
from rcalens.features import norm_name  # noqa: E402


def test_ties_count_against_the_model():
    df = pd.DataFrame({"case": ["a"] * 3 + ["b"] * 3, "service": list("xyz") * 2,
                       "score": [1, 1, 0, 3, 2, 1], "is_rc": [1, 0, 0, 0, 1, 0]})
    r = true_ranks(df)
    assert r["a"] == 2 and r["b"] == 2          # tie at the top -> worst tied position
    assert rank_metrics(r)["AC@1"] == 0 and rank_metrics(r)["AC@3"] == 1


def test_repetitions_never_split():
    cases = pd.DataFrame({"case": [f"c{i}" for i in range(30)], "pair": [f"p{i // 3}" for i in range(30)]})
    pair = cases.set_index("case")["pair"]
    for tr, te in grouped_folds(cases, n_splits=5):
        assert not set(pair[tr]) & set(pair[te])


def test_service_names_match_across_sources():
    assert norm_name("frontendservice") == norm_name("frontend")
    assert norm_name("front-end") == norm_name("frontend")
    assert norm_name("ts-auth-service") != norm_name("ts-auth-mongo")
