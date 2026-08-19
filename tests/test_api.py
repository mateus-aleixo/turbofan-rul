"""API contract tests against the committed model registry (models/).

Skipped when serving artifacts are absent (fresh clone before training) —
CI runs against the committed registry, so these execute there.
"""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from fastapi.testclient import TestClient  # noqa: E402

from conformal_rul.data import SENSOR_COLS, SETTING_COLS  # noqa: E402
from conformal_rul.serve.app import MODEL_ROOT, app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODEL_ROOT / "FD001" / "best.json").exists(),
    reason="serving artifacts not built (run train/conformal/export first)",
)

client = TestClient(app)


def cycles(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {s: 0.0 for s in SETTING_COLS}
        row.update({s: float(rng.normal()) for s in SENSOR_COLS})
        rows.append(row)
    return rows


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_lists_all_subsets():
    r = client.get("/models")
    assert r.status_code == 200
    subsets = {m["subset"] for m in r.json()}
    assert "FD001" in subsets
    for m in r.json():
        assert m["test_rmse"] > 0 and m["n_features"] > 0


def test_predict_contract():
    r = client.post("/predict", json={"subset": "FD001", "cycles": cycles(40)})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["rul_cycles"] <= 125
    assert body["interval"]["lower"] <= body["interval"]["upper"]
    assert body["interval"]["coverage_nominal"] == 90
    assert body["risk_band"] in ("critical", "warning", "healthy")
    assert body["padded"] is False and body["n_cycles_used"] == 30


def test_short_history_pads():
    r = client.post("/predict", json={"subset": "FD001", "cycles": cycles(5)})
    assert r.status_code == 200
    assert r.json()["padded"] is True and r.json()["n_cycles_used"] == 5


def test_wider_coverage_wider_interval():
    body = {"subset": "FD001", "cycles": cycles(40)}
    w = {}
    for cov in (80, 95):
        r = client.post("/predict", json={**body, "coverage": cov}).json()
        w[cov] = r["interval"]["upper"] - r["interval"]["lower"]
    assert w[95] >= w[80]


def test_validation_errors():
    assert client.post("/predict", json={"subset": "FD009", "cycles": cycles(3)}).status_code == 422
    assert client.post("/predict", json={"subset": "FD001", "cycles": []}).status_code == 422
    missing = [{"setting_1": 0.0}]
    assert client.post("/predict", json={"subset": "FD001", "cycles": missing}).status_code == 422


def test_interval_reports_measured_coverage_not_just_the_request():
    """`coverage_nominal` is what the caller asked for; `coverage_measured` is what
    that setting achieved on held-out data. Returning only the first states a
    target as though it were a result."""
    body = {"subset": "FD001", "cycles": cycles(35), "coverage": 90}

    iv = client.post("/predict", json=body).json()["interval"]
    assert iv["coverage_nominal"] == 90
    assert iv["coverage_measured"] is not None
    assert 0.0 <= iv["coverage_measured"] <= 1.0
    assert iv["mean_width"] > 0
    assert iv["n_calibration"] > 0
    assert iv["taxonomy"] == "band"          # the default, and the best calibrated


def test_measured_coverage_depends_on_the_taxonomy():
    """The caller picks the Mondrian grouping and the groupings do not cover
    equally: on FD001 at nominal 90, band lands near 0.89 and marginal near 0.82.
    Reporting one number for both would hide a real under-coverage."""
    base = {"subset": "FD001", "cycles": cycles(35), "coverage": 90}

    band = client.post("/predict", json={**base, "taxonomy": "band"}).json()["interval"]
    marginal = client.post("/predict", json={**base, "taxonomy": "marginal"}).json()["interval"]

    assert band["taxonomy"] == "band" and marginal["taxonomy"] == "marginal"
    assert band["coverage_measured"] != marginal["coverage_measured"]
    assert band["coverage_measured"] > marginal["coverage_measured"]


def test_models_exposes_measured_coverage_without_a_prediction():
    """The honest numbers should be readable before anyone sends data."""
    for info in client.get("/models").json():
        mc = info["measured_coverage"]
        assert mc is not None
        assert set(mc) == {"80", "90", "95"}
        assert all(0.0 <= v <= 1.0 for v in mc.values())
        # Coverage is monotone in the requested level.
        assert mc["80"] <= mc["90"] <= mc["95"]
