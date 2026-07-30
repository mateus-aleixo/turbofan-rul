"""LightGBM baseline on windowed summary statistics.

The baseline every sequence model must beat to justify itself: gradient
boosting on mean/std/last/slope per sensor. Same output convention as the
nets (mean column, then quantile columns) via one booster per objective.
"""

from __future__ import annotations

import numpy as np

from conformal_rul.config import QUANTILES

_BASE = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


def fit_gbm(X, y, X_val, y_val, seed: int = 42) -> dict:
    import lightgbm as lgb

    train_set = lgb.Dataset(X, y)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    boosters: dict[str, lgb.Booster] = {}
    objectives = [("mean", {"objective": "l2"})] + [
        (f"q{q}", {"objective": "quantile", "alpha": q}) for q in QUANTILES
    ]
    for name, obj in objectives:
        params = {**_BASE, **obj, "seed": seed}
        boosters[name] = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    return boosters


def predict_gbm(boosters: dict, X) -> np.ndarray:
    cols = [boosters["mean"].predict(X)] + [boosters[f"q{q}"].predict(X) for q in QUANTILES]
    return np.stack(cols, axis=1).astype(np.float32)
