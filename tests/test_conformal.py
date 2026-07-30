"""Conformal layer tests: the coverage guarantee is checked empirically on
synthetic data where exchangeability holds by construction."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from conformal_rul.config import QUANTILES
from conformal_rul.conformal import (
    ConformalRUL,
    finite_sample_quantile,
    rul_band,
)


def synthetic(n: int, seed: int, sigma_scale: float = 0.8):
    """Heteroscedastic ground truth with a deliberately miscalibrated
    quantile model (sigma_scale < 1 makes raw quantiles under-cover, so the
    conformal layer must widen them to reach nominal coverage)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    sigma = 1 + x / 20
    y = x + rng.normal(0, sigma)
    pred = np.empty((n, 1 + len(QUANTILES)), dtype=np.float64)
    pred[:, 0] = x
    for j, q in enumerate(QUANTILES):
        pred[:, 1 + j] = x + NormalDist().inv_cdf(q) * sigma_scale * sigma
    regime = (x > 50).astype(int)
    return pred, y, regime


@pytest.fixture(scope="module")
def fitted():
    cal_pred, cal_y, cal_regime = synthetic(2000, seed=0)
    test_pred, test_y, test_regime = synthetic(4000, seed=1)
    cal = ConformalRUL.fit(cal_pred, cal_y, cal_regime, rul_cap=200, method="cqr")
    return cal, test_pred, test_y, test_regime


def test_marginal_coverage_hits_nominal(fitted):
    cal, pred, y, regime = fitted
    for cov in (80, 90, 95):
        lo, hi = cal.interval(pred, regime, cov, "marginal")
        emp = ((y >= lo) & (y <= hi)).mean()
        # n_test = 4000 -> binomial sd < 0.007; allow ~3 sd plus the
        # finite-sample over-coverage inherent to the corrected quantile.
        assert cov / 100 - 0.02 <= emp <= cov / 100 + 0.03, (cov, emp)


def test_cqr_widens_miscalibrated_quantiles(fitted):
    cal, pred, y, regime = fitted
    assert cal.qhats[90]["marginal"] > 0  # raw quantiles under-covered


def test_cqr_intervals_adapt_to_heteroscedasticity(fitted):
    cal, pred, y, regime = fitted
    lo, hi = cal.interval(pred, regime, 90, "marginal")
    width = hi - lo
    easy = pred[:, 0] < 30
    hard = pred[:, 0] > 70
    assert width[hard].mean() > 1.5 * width[easy].mean()


def test_split_conformal_coverage():
    cal_pred, cal_y, cal_regime = synthetic(2000, seed=2)
    test_pred, test_y, test_regime = synthetic(4000, seed=3)
    cal = ConformalRUL.fit(cal_pred, cal_y, cal_regime, rul_cap=200, method="split")
    lo, hi = cal.interval(test_pred, test_regime, 90, "marginal")
    emp = ((test_y >= lo) & (test_y <= hi)).mean()
    assert 0.88 <= emp <= 0.93
    # Split intervals are symmetric around the point head by construction —
    # away from the [0, cap] clipping boundary.
    interior = (lo > 0) & (hi < 200)
    np.testing.assert_allclose(
        (test_pred[:, 0] - lo)[interior], (hi - test_pred[:, 0])[interior], atol=1e-9
    )


def test_regime_mondrian_fixes_group_undercoverage(fitted):
    cal, pred, y, regime = fitted
    # The hard regime (x > 50, larger sigma) must reach nominal coverage
    # under regime-Mondrian calibration.
    lo, hi = cal.interval(pred, regime, 90, "regime")
    hard = regime == 1
    emp = ((y >= lo) & (y <= hi))[hard].mean()
    assert emp >= 0.875


def test_small_group_falls_back_to_marginal():
    cal_pred, cal_y, _ = synthetic(500, seed=4)
    tiny_regime = np.zeros(500, dtype=int)
    tiny_regime[:10] = 1  # 10 < MIN_GROUP
    cal = ConformalRUL.fit(cal_pred, cal_y, tiny_regime, rul_cap=200)
    for cov in (80, 90, 95):
        assert cal.qhats[cov]["regime"][1] == cal.qhats[cov]["marginal"]


def test_finite_sample_quantile_edges():
    assert math.isinf(finite_sample_quantile(np.array([]), 0.1))
    assert math.isinf(finite_sample_quantile(np.arange(4), 0.05))  # n too small
    # n=10, alpha=0.1 -> ceil(11*0.9)/10 = 1.0 -> max of scores
    assert finite_sample_quantile(np.arange(1, 11), 0.1) == 10


def test_bands_and_clipping():
    assert list(rul_band(np.array([5.0, 50.0, 100.0, 300.0]), 125)) == [0, 1, 2, 2]
    cal_pred, cal_y, cal_regime = synthetic(1000, seed=5)
    cal = ConformalRUL.fit(cal_pred, cal_y, cal_regime, rul_cap=125)
    lo, hi = cal.interval(cal_pred, cal_regime, 95, "band")
    assert lo.min() >= 0 and hi.max() <= 125


def test_json_roundtrip(fitted):
    cal, pred, y, regime = fitted
    cal2 = ConformalRUL.from_json(cal.to_json())
    for tax in ("marginal", "regime", "band"):
        lo1, hi1 = cal.interval(pred, regime, 90, tax)
        lo2, hi2 = cal2.interval(pred, regime, 90, tax)
        np.testing.assert_allclose(lo1, lo2)
        np.testing.assert_allclose(hi1, hi2)
