"""Deep-learning ranker: a 1D CNN reads each service's metric time series and scores it.

The inputs are built in notebook 02 (Part B) with pandas: per (case, service) an array of
16 channels x 144 steps (8 metric types scaled with the case's own normal median / spread,
plus 8 0/1 channels saying which metric types exist), 10-second steps, fault in the middle.

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

C = 16                                     # input channels: 8 metric types + 8 presence flags


def to_cases(X: np.ndarray, rows: pd.DataFrame) -> dict:
    """Group the rows saved by notebook 02 by case: {case: {"services", "X", "y" (index of the root cause)}}."""
    data = {}
    for case, g in rows.groupby("case", sort=False):
        data[case] = {"services": g["service"].tolist(), "X": X[g.index.to_numpy()], "y": int(g["is_rc"].to_numpy().argmax())}
    return data


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
