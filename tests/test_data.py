"""Data pipeline tests on synthetic frames — no dataset download required."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from conformal_rul.data import (
    COLUMNS,
    SENSOR_COLS,
    SETTING_COLS,
    Preprocessor,
    add_train_rul,
    download,
    last_windows,
    sliding_windows,
    split_units,
    window_stats,
)

REGIME_TRIPLES = [
    (0, 0, 100), (10, 0.25, 100), (20, 0.7, 100),
    (25, 0.62, 60), (35, 0.84, 60), (42, 0.84, 40),
]


def toy_frame(lengths: dict[int, int], n_regimes: int = 1, seed: int = 0) -> pd.DataFrame:
    """Units with given lengths; s_01 constant, the rest trend + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in lengths.items():
        for cycle in range(1, n + 1):
            setting = REGIME_TRIPLES[rng.integers(n_regimes)]
            sensors = [1.0] + [
                cycle / n + rng.normal(0, 0.05) + (setting[0] if j % 2 else 0)
                for j in range(2, 22)
            ]
            rows.append([unit, cycle, *setting, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_train_rul_counts_down_and_caps():
    df = add_train_rul(toy_frame({1: 200}), cap=125)
    assert df["rul"].iloc[-1] == 0
    assert df["rul"].iloc[0] == 125  # capped: raw would be 199
    assert df["rul"].iloc[100] == 200 - 101


def test_constant_sensor_dropped_and_roundtrip():
    df = toy_frame({1: 60, 2: 80})
    pre = Preprocessor.fit(df, "FD001", window=30, rul_cap=125)
    assert "s_01" not in pre.feature_names
    assert pre.n_regimes == 1
    feat, regime = pre.transform(df)
    pre2 = Preprocessor.from_json(pre.to_json())
    feat2, regime2 = pre2.transform(df)
    np.testing.assert_allclose(feat, feat2, rtol=1e-6)
    np.testing.assert_array_equal(regime, regime2)


def test_multi_regime_normalization_removes_regime_offsets():
    df = toy_frame({1: 300, 2: 300}, n_regimes=6, seed=1)
    pre = Preprocessor.fit(df, "FD002", window=30, rul_cap=125)
    assert pre.n_regimes == 6
    feat, regime = pre.transform(df)
    assert len(np.unique(regime)) == 6
    # After per-regime z-scoring, per-regime means of every feature are ~0.
    for r in range(6):
        assert np.abs(feat[regime == r].mean(0)).max() < 0.2


def test_split_units_disjoint_and_complete():
    units = np.repeat(np.arange(50), 3)
    parts = split_units(units, val_frac=0.2, cal_frac=0.2, seed=0)
    all_ids = np.concatenate(list(parts.values()))
    assert len(all_ids) == 50
    assert len(np.unique(all_ids)) == 50
    assert len(parts["val"]) == 10 and len(parts["cal"]) == 10


def test_sliding_windows_shapes_and_padding():
    feat = np.arange(40 * 3, dtype=float).reshape(40, 3)
    short = np.arange(20 * 3, dtype=float).reshape(20, 3) + 1000
    f = np.concatenate([feat, short])
    units = np.array([1] * 40 + [2] * 20)
    y = np.arange(60, dtype=float)
    regime = np.zeros(60, dtype=int)
    X, yy, rr, uu = sliding_windows(f, regime, y, units, window=30)
    assert X.shape == (11 + 1, 30, 3)  # unit1: 40-30+1 full; unit2: 1 padded
    assert (uu == 2).sum() == 1
    padded = X[uu == 2][0]
    np.testing.assert_array_equal(padded[0], padded[9])  # first row repeated
    assert yy[uu == 2][0] == y[-1]


def test_last_windows_one_per_unit():
    f = np.random.default_rng(0).normal(size=(50, 4))
    units = np.array([1] * 35 + [2] * 15)
    X, rr, uu = last_windows(f, np.zeros(50, int), units, window=30)
    assert X.shape == (2, 30, 4)
    np.testing.assert_allclose(X[0], f[5:35])  # last 30 rows of unit 1


def test_window_stats_values():
    X = np.zeros((1, 4, 2), dtype=np.float32)
    X[0, :, 0] = [1, 2, 3, 4]  # slope 1, mean 2.5, last 4
    stats = window_stats(X)
    assert stats.shape == (1, 8)
    mean, _std, last, slope = stats[0, 0], stats[0, 2], stats[0, 4], stats[0, 6]
    assert mean == pytest.approx(2.5)
    assert last == pytest.approx(4.0)
    assert slope == pytest.approx(1.0)


def test_download_extracts_nested_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for s in ("FD001", "FD002", "FD003", "FD004"):
            for kind in ("train", "test", "RUL"):
                z.writestr(f"{kind}_{s}.txt", "1 1 0 0 100 " + " ".join(["1"] * 21))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("6. Turbofan/CMAPSSData.zip", inner.getvalue())
    src = tmp_path / "archive.zip"
    src.write_bytes(outer.getvalue())

    root = download(tmp_path / "out", url=src.resolve().as_uri())
    assert (root / "train_FD001.txt").exists()
    assert (root / "RUL_FD004.txt").exists()


def test_frame_columns_contract():
    assert len(COLUMNS) == 26
    assert len(SENSOR_COLS) == 21
    assert len(SETTING_COLS) == 3
