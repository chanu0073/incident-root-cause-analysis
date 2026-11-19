"""Deep-learning ranker: a 1D CNN reads each service's raw metric time series and scores it.

Input per (case, service): 8 metric types x 144 steps (24 min at 10 s). Each series is standardised
with the median / IQR of the SAME case's pre-fault window (no information from other cases or from
the label), clipped and log-compressed. 8 extra 0/1 channels say which metric types exist.

Training is listwise: inside one case the scores of all services go through a softmax, and the
loss is the cross-entropy of the true root-cause service ("pick the right one among these").
The network is shared by all services, so it learns what a *root-cause-like* pattern looks like,
not service names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import METRIC_TYPES, load_case, split_metric

STEP = 10                                  # seconds per time step after down-sampling
C = 2 * len(METRIC_TYPES)                  # values + presence mask


def case_tensor(name: str, step: int = STEP) -> tuple[list[str], np.ndarray]:
    case = load_case(name, traces=False)
    m = case.metrics
    t = m["time"].to_numpy()
    half = 720                                                   # 12 minutes before / after the injection
    bins = ((t - (case.inject_time - half)) // step).astype(int)  # same time axis for every case
    n_bins = 2 * half // step
    keep = (bins >= 0) & (bins < n_bins)
    pre = t < case.inject_time
    services = case.services
    X = np.zeros((len(services), C, n_bins), dtype=np.float32)
    s_idx = {s: i for i, s in enumerate(services)}
    for col in m.columns:
        if col == "time":
            continue
        svc, typ = split_metric(col)
        if typ not in METRIC_TYPES:
            continue
        x = m[col].to_numpy(dtype=float)
        a = x[pre][~np.isnan(x[pre])]
        if len(a) < 10:
            continue
        med = np.median(a)
        scale = np.subtract(*np.percentile(a, [75, 25])) or a.std() or 1e-3 * max(abs(med), 1e-3)
        z = np.clip(np.nan_to_num((x - med) / scale), -1e4, 1e4)
        z = np.sign(z) * np.log1p(np.abs(z))
        zb = np.bincount(bins[keep], weights=z[keep], minlength=n_bins) / np.maximum(np.bincount(bins[keep], minlength=n_bins), 1)
        k = METRIC_TYPES.index(typ)
        X[s_idx[svc], k] = zb if not X[s_idx[svc], len(METRIC_TYPES) + k].any() else np.maximum(X[s_idx[svc], k], zb)
        X[s_idx[svc], len(METRIC_TYPES) + k] = 1.0
    return services, X


def build_dataset(idx: pd.DataFrame) -> dict:
    out = {}
    for r in idx.itertuples():
        services, X = case_tensor(r.case)
        out[r.case] = {"services": services, "X": X, "y": services.index(r.root_cause_service)}
    return out


class ServiceEncoder(nn.Module):
    def __init__(self, c_in: int = C, width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, x):                   # x: (N, C, T) -> (N,)
        h = self.conv(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h).squeeze(-1)


def _batches(data: dict, cases: list[str], bs: int, rng: np.random.Generator | None):
    order = list(cases) if rng is None else list(rng.permutation(cases))
    for i in range(0, len(order), bs):
        chunk = order[i:i + bs]
        X = np.concatenate([data[c]["X"] for c in chunk])
        sizes = [len(data[c]["services"]) for c in chunk]
        y = [data[c]["y"] for c in chunk]
        yield chunk, torch.from_numpy(X), sizes, y


def _listwise_loss(scores, sizes, y):
    loss, start = 0.0, 0
    for n, t in zip(sizes, y, strict=True):
        loss = loss + nn.functional.cross_entropy(scores[start:start + n].unsqueeze(0), torch.tensor([t], device=scores.device))
        start += n
    return loss / len(sizes)


def train_cnn(data: dict, train_cases: list[str], device: str, epochs: int = 60, lr: float = 1e-3, bs: int = 16,
              seed: int = 0, width: int = 64) -> ServiceEncoder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ServiceEncoder(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for _, X, sizes, y in _batches(data, train_cases, bs, rng):
            noise = 0.05 * torch.randn_like(X)                    # light augmentation
            scores = model((X + noise).to(device))
            loss = _listwise_loss(scores, sizes, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def score_cases(model: ServiceEncoder, data: dict, cases: list[str], device: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for chunk, X, sizes, _ in _batches(data, cases, 32, None):
        s = model(X.to(device)).float().cpu().numpy()
        start = 0
        for c, n in zip(chunk, sizes, strict=True):
            for j, svc in enumerate(data[c]["services"]):
                rows.append((c, svc, float(s[start + j]), int(j == data[c]["y"])))
            start += n
    return pd.DataFrame(rows, columns=["case", "service", "score", "is_rc"])
