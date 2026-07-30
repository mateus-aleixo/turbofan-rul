"""Point-forecast metrics for the C-MAPSS benchmark."""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM08 scoring function, summed over units. Asymmetric by design: a late
    prediction (pred > true — the engine fails before you expected) is penalized
    exp(d/10), an early one only exp(-d/13). That asymmetry is the whole point
    of the benchmark: in maintenance, optimism is the expensive failure mode."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def point_metrics(pred: np.ndarray, true: np.ndarray, cap: int) -> dict:
    p = np.clip(pred, 0, cap)
    return {
        "rmse": round(rmse(p, true), 3),
        "nasa_score": round(nasa_score(p, true), 1),
        "mae": round(float(np.mean(np.abs(p - true))), 3),
        "bias": round(float(np.mean(p - true)), 3),
    }
