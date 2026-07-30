"""C-MAPSS turbofan data: download, parsing, labels, normalization, windowing.

Split policy: splits are BY ENGINE UNIT, never by window. Windows from one
engine are near-duplicates; a window-level split leaks the test engines'
degradation paths into training and inflates every downstream metric,
including conformal coverage.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# Documented number of operating conditions per subset (readme.txt of the dataset).
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i:02d}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

DATA_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


# ---------------------------------------------------------------- download

def download(root: str | Path = "data/raw/CMAPSSData", url: str = DATA_URL) -> Path:
    """Fetch the NASA archive (a zip containing CMAPSSData.zip) and extract the
    text files into *root*. Idempotent: skips the download if all files exist."""
    root = Path(root)
    expected = [f"{kind}_{s}.txt" for s in SUBSETS for kind in ("train", "test", "RUL")]
    if all((root / f).exists() for f in expected):
        return root

    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in inner.namelist():
        if name.endswith(".txt"):
            (root / Path(name).name).write_bytes(inner.read(name))
    missing = [f for f in expected if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"archive did not contain: {missing}")
    return root


# ---------------------------------------------------------------- parsing

def load_frame(subset: str, kind: str, root: str | Path) -> pd.DataFrame:
    """*kind* is 'train' or 'test'. Returns unit, cycle, settings and sensors."""
    path = Path(root) / f"{kind}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str, root: str | Path) -> np.ndarray:
    """True RUL at each test unit's last observed cycle, ordered by unit id."""
    path = Path(root) / f"RUL_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None)[0].to_numpy(float)


def add_train_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Training units run to failure, so RUL = cycles remaining, capped."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).clip(upper=cap).astype(float)
    return out


# ---------------------------------------------------------------- preprocessing

def _nearest(z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


@dataclass
class Preprocessor:
    """Regime-conditional z-scoring, fitted on training units only.

    In multi-regime subsets the same sensor sits at different set-points per
    operating condition; a global z-score would encode the regime, not the
    degradation. Normalizing within each regime leaves the degradation signal.
    Serialized to JSON so serving needs neither pickle nor scikit-learn.
    """

    subset: str
    window: int
    rul_cap: int
    feature_names: list[str]
    setting_mean: np.ndarray  # [3]
    setting_std: np.ndarray  # [3]
    regime_centers: np.ndarray  # [R, 3], in standardized-settings space
    mean: np.ndarray  # [R, F]
    std: np.ndarray  # [R, F]

    @classmethod
    def fit(cls, train: pd.DataFrame, subset: str, window: int, rul_cap: int) -> Preprocessor:
        n_regimes = N_REGIMES[subset]
        settings = train[SETTING_COLS].to_numpy(float)
        smean = settings.mean(0)
        sstd = settings.std(0)
        sstd[sstd < 1e-8] = 1.0
        z = (settings - smean) / sstd

        if n_regimes == 1:
            centers = z.mean(0, keepdims=True)
        else:
            from sklearn.cluster import KMeans  # train-time only; serving stays sklearn-free

            km = KMeans(n_clusters=n_regimes, n_init=10, random_state=0).fit(z)
            centers = km.cluster_centers_
        regime = _nearest(z, centers)

        # Keep a sensor iff it varies within at least one regime. Constant
        # channels carry no degradation signal and destabilize z-scoring.
        sensors = train[SENSOR_COLS].to_numpy(float)
        keep = [
            name
            for j, name in enumerate(SENSOR_COLS)
            if max(sensors[regime == r, j].std() for r in range(len(centers))) > 1e-6
        ]
        feats = train[keep].to_numpy(float)
        mean = np.stack([feats[regime == r].mean(0) for r in range(len(centers))])
        std = np.stack([feats[regime == r].std(0) for r in range(len(centers))])
        std[std < 1e-8] = 1.0
        return cls(subset, window, rul_cap, keep, smean, sstd, centers, mean, std)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_regimes(self) -> int:
        return len(self.regime_centers)

    def assign_regime(self, settings: np.ndarray) -> np.ndarray:
        z = (settings - self.setting_mean) / self.setting_std
        return _nearest(z, self.regime_centers)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features [n, F] z-scored per regime, regime [n])."""
        regime = self.assign_regime(df[SETTING_COLS].to_numpy(float))
        feats = df[self.feature_names].to_numpy(float)
        out = (feats - self.mean[regime]) / self.std[regime]
        return out.astype(np.float32), regime

    # JSON round-trip: the serving image reconstructs this object without pickle.
    def to_json(self) -> str:
        d = {
            "subset": self.subset,
            "window": self.window,
            "rul_cap": self.rul_cap,
            "feature_names": self.feature_names,
            "setting_mean": self.setting_mean.tolist(),
            "setting_std": self.setting_std.tolist(),
            "regime_centers": self.regime_centers.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Preprocessor:
        d = json.loads(text)
        return cls(
            d["subset"],
            d["window"],
            d["rul_cap"],
            d["feature_names"],
            np.asarray(d["setting_mean"]),
            np.asarray(d["setting_std"]),
            np.asarray(d["regime_centers"]),
            np.asarray(d["mean"]),
            np.asarray(d["std"]),
        )


# ---------------------------------------------------------------- windowing

def _pad_to(feat: np.ndarray, window: int) -> np.ndarray:
    """Left-pad a short sequence by repeating its first row."""
    pad = np.repeat(feat[:1], window - len(feat), axis=0)
    return np.concatenate([pad, feat], axis=0)


def sliding_windows(
    feat: np.ndarray,
    regime: np.ndarray,
    y: np.ndarray,
    units: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full windows with stride 1 per unit; label/regime taken at the window end.
    A unit shorter than *window* yields one left-padded window."""
    xs, ys, rs, us = [], [], [], []
    for u in np.unique(units):
        m = units == u
        f, yy, rr = feat[m], y[m], regime[m]
        if len(f) < window:
            xs.append(_pad_to(f, window)[None])
            ys.append(yy[-1:])
            rs.append(rr[-1:])
            us.append(np.array([u]))
            continue
        w = np.lib.stride_tricks.sliding_window_view(f, window, axis=0)
        xs.append(np.ascontiguousarray(w.transpose(0, 2, 1)))
        ys.append(yy[window - 1 :])
        rs.append(rr[window - 1 :])
        us.append(np.full(len(f) - window + 1, u))
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(rs),
        np.concatenate(us),
    )


def last_windows(
    feat: np.ndarray, regime: np.ndarray, units: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per unit ending at its last observed cycle (benchmark protocol)."""
    xs, rs, us = [], [], []
    for u in np.unique(units):
        f = feat[units == u][-window:]
        if len(f) < window:
            f = _pad_to(f, window)
        xs.append(f[None])
        rs.append(regime[units == u][-1:])
        us.append(np.array([u]))
    return np.concatenate(xs).astype(np.float32), np.concatenate(rs), np.concatenate(us)


# ---------------------------------------------------------------- assembly

@dataclass
class Bundle:
    X: np.ndarray  # [N, window, F]
    y: np.ndarray  # [N]
    regime: np.ndarray  # [N]
    unit: np.ndarray  # [N]


def split_units(units: np.ndarray, val_frac: float, cal_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(units))
    n_val = max(1, round(len(ids) * val_frac))
    n_cal = max(1, round(len(ids) * cal_frac))
    return {
        "val": ids[:n_val],
        "cal": ids[n_val : n_val + n_cal],
        "train": ids[n_val + n_cal :],
    }


def prepare(
    cfg: DataConfig, root: str | Path = "data/raw/CMAPSSData"
) -> tuple[dict[str, Bundle], Preprocessor]:
    """Windowed train/val/cal/test bundles plus the fitted preprocessor.

    The preprocessor (regime centers, normalization stats, sensor selection) is
    fitted on the train-split units only — val steers early stopping and cal
    feeds conformal calibration, so neither may inform normalization.
    """
    raw = add_train_rul(load_frame(cfg.subset, "train", root), cfg.rul_cap)
    parts = split_units(raw["unit"].to_numpy(), cfg.val_frac, cfg.cal_frac, cfg.seed)
    pre = Preprocessor.fit(
        raw[raw["unit"].isin(parts["train"])], cfg.subset, cfg.window, cfg.rul_cap
    )

    bundles: dict[str, Bundle] = {}
    for name, ids in parts.items():
        df = raw[raw["unit"].isin(ids)]
        feat, regime = pre.transform(df)
        X, y, r, u = sliding_windows(
            feat, regime, df["rul"].to_numpy(float), df["unit"].to_numpy(), cfg.window
        )
        bundles[name] = Bundle(X, y, r, u)

    test = load_frame(cfg.subset, "test", root)
    feat, regime = pre.transform(test)
    X, r, u = last_windows(feat, regime, test["unit"].to_numpy(), cfg.window)
    y = np.clip(load_test_rul(cfg.subset, root), 0, cfg.rul_cap).astype(np.float32)
    bundles["test"] = Bundle(X, y, r, u)
    return bundles, pre


# ---------------------------------------------------------------- GBM features

STAT_NAMES = ("mean", "std", "last", "slope")


def window_stats(X: np.ndarray) -> np.ndarray:
    """[N, L, F] -> [N, 4F]: per-channel mean, std, last value and OLS slope.
    Order matches stat_feature_names()."""
    n, length, _ = X.shape
    t = np.arange(length, dtype=np.float64)
    t = t - t.mean()
    denom = (t**2).sum()
    mean = X.mean(1)
    std = X.std(1)
    last = X[:, -1, :]
    slope = np.einsum("l,nlf->nf", t, X - mean[:, None, :]) / denom
    return np.concatenate([mean, std, last, slope], axis=1).astype(np.float32)


def stat_feature_names(feature_names: list[str]) -> list[str]:
    return [f"{f}_{s}" for s in STAT_NAMES for f in feature_names]


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Download the C-MAPSS dataset.")
    ap.add_argument("--root", default="data/raw/CMAPSSData")
    args = ap.parse_args()
    root = download(args.root)
    for s in SUBSETS:
        tr, te = load_frame(s, "train", root), load_frame(s, "test", root)
        print(
            f"{s}: train {len(tr)} rows / {tr['unit'].nunique()} units, "
            f"test {len(te)} rows / {te['unit'].nunique()} units"
        )


if __name__ == "__main__":
    main()
