"""Build data/processed/service_features.parquet: one row per (case, candidate service), with the label.

  python scripts/build_features.py             # metrics + logs + traces (training server)
  python scripts/build_features.py --no-traces # skip trace features (laptop / quick runs)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcalens.data import PROCESSED, case_index, load_case  # noqa: E402
from rcalens.features import case_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-traces", action="store_true")
ap.add_argument("--jobs", type=int, default=-1)
args = ap.parse_args()


def one(row) -> pd.DataFrame:
    f = case_features(load_case(row.case, traces=not args.no_traces))
    f.insert(0, "case", row.case)
    f["system"], f["fault"], f["pair"], f["repetition"] = row.system, row.fault, row.pair, row.repetition
    f["root_cause_service"] = row.root_cause_service
    f["is_rc"] = (f["service"] == row.root_cause_service).astype(int)
    return f


idx = case_index()
t = time.time()
parts = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(r) for r in idx.itertuples())
df = pd.concat(parts, ignore_index=True)
missing = idx.loc[~idx["case"].isin(df.loc[df["is_rc"] == 1, "case"]), "case"].tolist()
assert not missing, f"root-cause service has no metrics in: {missing}"
PROCESSED.mkdir(parents=True, exist_ok=True)
out = PROCESSED / ("service_features_notraces.parquet" if args.no_traces else "service_features.parquet")
df.to_parquet(out, index=False)
print(f"{len(idx)} cases, {len(df)} service rows, {df.shape[1]} columns -> {out} ({time.time() - t:.0f}s)")
