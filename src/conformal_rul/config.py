"""Run configuration. One place for every knob that affects a result."""

from __future__ import annotations

from dataclasses import dataclass

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Quantile heads trained with pinball loss. Symmetric pairs around the median
# give raw 95/90/80 % intervals; CQR then calibrates each pair.
QUANTILES = (0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975)

# Coverage levels reported everywhere (percent).
COVERAGES = (80, 90, 95)

# Predicted-RUL band edges (cycles) used for risk labels and Mondrian buckets.
RUL_BAND_EDGES = (30, 80)
RUL_BAND_NAMES = ("critical", "warning", "healthy")


@dataclass(frozen=True)
class DataConfig:
    subset: str = "FD001"
    window: int = 30
    rul_cap: int = 125  # piecewise-linear target, standard for C-MAPSS since Heimes (2008)
    val_frac: float = 0.15
    cal_frac: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    model: str = "lstm"  # lstm | transformer | gbm
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.25
    heads: int = 4  # transformer only
    ff_mult: int = 4  # transformer feed-forward width multiplier
    pinball_weight: float = 1.0
    seed: int = 42


# Per-model overrides on top of TrainConfig defaults.
PRESETS: dict[str, dict] = {
    "lstm": {},
    "transformer": {"hidden": 96, "layers": 3, "lr": 5e-4, "dropout": 0.15},
    "gbm": {},
}


def train_config(model: str, **overrides) -> TrainConfig:
    kwargs = {"model": model, **PRESETS.get(model, {}), **overrides}
    return TrainConfig(**kwargs)


@dataclass
class Paths:
    data_root: str = "data/raw/CMAPSSData"
    models_root: str = "models"
    figures_root: str = "docs/figures"
