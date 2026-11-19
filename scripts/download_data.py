"""Download the RCAEval RE2 suite (270 cases) from Hugging Face into data/raw/.

  python scripts/download_data.py              # metrics + logs + traces (~3.5 GB, use on the training server)
  python scripts/download_data.py --no-traces  # metrics + logs only (~0.5 GB, enough for notebooks 01-04 without trace features)
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ap = argparse.ArgumentParser()
ap.add_argument("--no-traces", action="store_true", help="skip traces.parquet (the largest files)")
ap.add_argument("--systems", default="ob,ss,tt", help="comma list of RE2 systems: ob, ss, tt")
args = ap.parse_args()

RAW.mkdir(parents=True, exist_ok=True)
hf_hub_download("phamquiluan/RCAEval", "cases.parquet", repo_type="dataset", local_dir=RAW)
files = ["inject_time.txt", "metrics.parquet", "logs.parquet"] + ([] if args.no_traces else ["traces.parquet"])
patterns = [f"re2{s}_*/{f}" for s in args.systems.split(",") for f in files]
snapshot_download("phamquiluan/RCAEval", repo_type="dataset", local_dir=RAW, allow_patterns=patterns, max_workers=8)
n = len(list(RAW.glob("re2*/metrics.parquet")))
print(f"done: {n} RE2 cases in {RAW}")
