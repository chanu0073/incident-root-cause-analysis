"""Loading RCAEval RE2 cases.

A case = one injected failure: 12 minutes of normal operation, then the fault is injected
(`inject_time`), then 12 more minutes. Files per case (data/raw/<case>/):
  metrics.parquet  1 row per second; columns "<service>_<metric>" (cpu, mem, diskio, socket,
                   workload, error, latency-50, latency-90)
  logs.parquet     timestamp, container_name, message
  traces.parquet   one row per span (serviceName, duration in microseconds, statusCode, ...);
                   not available for Sock Shop
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METRIC_TYPES = ["cpu", "mem", "diskio", "socket", "workload", "error", "latency-50", "latency-90"]


def case_index(raw: Path = RAW) -> pd.DataFrame:
    """The 270 RE2 cases that are present on disk, with their ground truth."""
    idx = pd.read_parquet(raw / "cases.parquet")
    idx = idx[idx["dataset"].astype(str).str.startswith("RE2")].copy()
    idx = idx[[(raw / c / "metrics.parquet").exists() for c in idx["case"]]]
    idx["system"] = idx["system"].astype(str)
    idx["pair"] = idx["system"] + "|" + idx["root_cause_service"].astype(str) + "|" + idx["fault"].astype(str)
    return idx.sort_values("case").reset_index(drop=True)


def split_metric(col: str) -> tuple[str, str]:
    """'ts-auth-service_latency-50' -> ('ts-auth-service', 'latency-50')"""
    service, metric = col.rsplit("_", 1)
    return service, metric


@dataclass
class Case:
    name: str
    inject_time: int
    metrics: pd.DataFrame
    logs: pd.DataFrame | None
    traces: pd.DataFrame | None

    @property
    def services(self) -> list[str]:
        return sorted({split_metric(c)[0] for c in self.metrics.columns if c != "time"})


def load_case(name: str, raw: Path = RAW, traces: bool = True) -> Case:
    d = raw / name
    inject = int(float((d / "inject_time.txt").read_text().strip()))
    metrics = pd.read_parquet(d / "metrics.parquet")
    logs = pd.read_parquet(d / "logs.parquet") if (d / "logs.parquet").exists() else None
    tr = pd.read_parquet(d / "traces.parquet") if traces and (d / "traces.parquet").exists() else None
    return Case(name, inject, metrics, logs, tr)


def load_features(prefer_traces: bool = True) -> tuple[pd.DataFrame, bool]:
    """The per-service feature table built by scripts/build_features.py (with traces if available)."""
    full, light = PROCESSED / "service_features.parquet", PROCESSED / "service_features_notraces.parquet"
    path = full if prefer_traces and full.exists() else light
    if not path.exists():
        raise FileNotFoundError("run `python scripts/build_features.py` first")
    return pd.read_parquet(path), path == full
