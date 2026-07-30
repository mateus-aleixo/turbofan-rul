"""Distribution-free prediction intervals: split conformal, CQR, Mondrian.

Method notes
------------
- Split conformal (absolute residual) wraps the point head:
  interval = point ± q̂, where q̂ is the finite-sample-corrected quantile of
  calibration residuals. Coverage ≥ 1-α holds exactly under exchangeability.
- CQR (Romano, Patterson & Candès 2019) wraps a quantile pair:
  score = max(lo - y, y - hi); interval = [lo - q̂, hi + q̂]. Widths adapt to
  local difficulty, and q̂ may be negative (the raw pair was too wide).
- Mondrian calibration computes q̂ per group. Marginal coverage can hide
  systematic under-coverage in a regime or near end of life; per-group
  calibration surfaces and fixes exactly that. Groups here are (a) operating
  regime and (b) predicted-RUL band — the band is a function of the model
  output, so it is available at inference time, unlike the true RUL.
- Finite-sample correction: q̂ = ⌈(n+1)(1-α)⌉/n empirical quantile. When a
  group is too small for the guarantee at level α the quantile is infinite;
  we fall back to the marginal q̂ rather than emit a vacuous interval.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import COVERAGES, QUANTILES, RUL_BAND_EDGES, RUL_BAND_NAMES

# Quantile-pair column (into the [N, 1+Q] prediction matrix) per coverage level.
_PAIR = {
    80: (1 + QUANTILES.index(0.10), 1 + QUANTILES.index(0.90)),
    90: (1 + QUANTILES.index(0.05), 1 + QUANTILES.index(0.95)),
    95: (1 + QUANTILES.index(0.025), 1 + QUANTILES.index(0.975)),
}
MIN_GROUP = 50  # below this, a group's guarantee is judged too weak; use marginal


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """⌈(n+1)(1-α)⌉/n empirical quantile; inf when n is too small to certify."""
    n = len(scores)
    if n == 0:
        return math.inf
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return math.inf
    return float(np.quantile(scores, level, method="higher"))


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def split_scores(point: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(y - point)


def rul_band(point_pred: np.ndarray, cap: int) -> np.ndarray:
    """Risk-band index from the *predicted* RUL — usable at inference time."""
    return np.digitize(np.clip(point_pred, 0, cap), RUL_BAND_EDGES)


@dataclass
class ConformalRUL:
    """Calibrated interval layer over a fitted model's [N, 1+Q] predictions.

    Stores one q̂ per (coverage, taxonomy, group); JSON-serializable so the
    serving image reconstructs it without any ML dependency.
    """

    method: str  # "cqr" | "split"
    rul_cap: int
    # {coverage: {"marginal": q, "regime": {gid: q}, "band": {bid: q}}}
    qhats: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        cal_pred: np.ndarray,
        cal_y: np.ndarray,
        cal_regime: np.ndarray,
        rul_cap: int,
        method: str = "cqr",
    ) -> ConformalRUL:
        self = cls(method=method, rul_cap=rul_cap)
        bands = rul_band(cal_pred[:, 0], rul_cap)
        for cov in COVERAGES:
            alpha = 1 - cov / 100
            scores = self._scores(cal_pred, cal_y, cov)
            marginal = finite_sample_quantile(scores, alpha)
            entry = {"marginal": marginal, "regime": {}, "band": {}}
            for name, groups in (("regime", cal_regime), ("band", bands)):
                for g in np.unique(groups):
                    m = groups == g
                    q = (
                        finite_sample_quantile(scores[m], alpha)
                        if m.sum() >= MIN_GROUP
                        else math.inf
                    )
                    entry[name][int(g)] = marginal if math.isinf(q) else q
            self.qhats[cov] = entry
        return self

    def _scores(self, pred: np.ndarray, y: np.ndarray, cov: int) -> np.ndarray:
        if self.method == "split":
            return split_scores(pred[:, 0], y)
        lo_c, hi_c = _PAIR[cov]
        return cqr_scores(pred[:, lo_c], pred[:, hi_c], y)

    def interval(
        self,
        pred: np.ndarray,
        regime: np.ndarray,
        coverage: int = 90,
        taxonomy: str = "band",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds, clipped to [0, cap]. taxonomy: marginal|regime|band."""
        entry = self.qhats[coverage]
        if taxonomy == "marginal":
            q = np.full(len(pred), entry["marginal"])
        elif taxonomy == "regime":
            q = np.array([entry["regime"].get(int(g), entry["marginal"]) for g in regime])
        elif taxonomy == "band":
            bands = rul_band(pred[:, 0], self.rul_cap)
            q = np.array([entry["band"].get(int(g), entry["marginal"]) for g in bands])
        else:
            raise ValueError(f"unknown taxonomy: {taxonomy}")

        if self.method == "split":
            lo, hi = pred[:, 0] - q, pred[:, 0] + q
        else:
            lo_c, hi_c = _PAIR[coverage]
            lo, hi = pred[:, lo_c] - q, pred[:, hi_c] + q
        return np.clip(lo, 0, self.rul_cap), np.clip(hi, 0, self.rul_cap)

    def report(
        self, pred: np.ndarray, y: np.ndarray, regime: np.ndarray
    ) -> dict:
        """Empirical coverage / mean width on held-out data, overall and per group."""
        out: dict = {"method": self.method, "n": int(len(y)), "coverages": {}}
        bands = rul_band(pred[:, 0], self.rul_cap)
        for cov in COVERAGES:
            block: dict = {}
            for tax in ("marginal", "regime", "band"):
                lo, hi = self.interval(pred, regime, cov, tax)
                inside = (y >= lo) & (y <= hi)
                block[tax] = {
                    "coverage": round(float(inside.mean()), 4),
                    "width": round(float((hi - lo).mean()), 2),
                }
                groups = {"regime": regime, "band": bands}.get(tax)
                if groups is not None:
                    per = {}
                    for g in np.unique(groups):
                        m = groups == g
                        label = RUL_BAND_NAMES[int(g)] if tax == "band" else f"regime_{int(g)}"
                        per[label] = {
                            "n": int(m.sum()),
                            "coverage": round(float(inside[m].mean()), 4),
                            "width": round(float((hi[m] - lo[m]).mean()), 2),
                        }
                    block[tax]["groups"] = per
            out["coverages"][cov] = block
        return out

    def to_json(self) -> str:
        return json.dumps(
            {"method": self.method, "rul_cap": self.rul_cap, "qhats": self.qhats}, indent=2
        )

    @classmethod
    def from_json(cls, text: str) -> ConformalRUL:
        d = json.loads(text)
        qhats = {
            int(cov): {
                "marginal": e["marginal"],
                "regime": {int(k): v for k, v in e["regime"].items()},
                "band": {int(k): v for k, v in e["band"].items()},
            }
            for cov, e in d["qhats"].items()
        }
        return cls(method=d["method"], rul_cap=d["rul_cap"], qhats=qhats)


def calibrate_dir(model_dir: str | Path, method: str = "cqr") -> dict:
    """Fit on the saved calibration predictions, evaluate on test, write
    conformal.json + coverage_report.json next to the model artifacts."""
    model_dir = Path(model_dir)
    z = np.load(model_dir / "preds.npz")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    cal = ConformalRUL.fit(
        z["cal_pred"], z["cal_y"], z["cal_regime"], manifest["data"]["rul_cap"], method
    )
    report = cal.report(z["test_pred"], z["test_y"], z["test_regime"])
    (model_dir / "conformal.json").write_text(cal.to_json())
    (model_dir / "coverage_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--method", choices=("cqr", "split"), default="cqr")
    args = ap.parse_args()
    for manifest in sorted(Path(args.models_root).glob("*/*/manifest.json")):
        d = manifest.parent
        report = calibrate_dir(d, args.method)
        c90 = report["coverages"][90]["band"]
        print(
            f"{d.parent.name}/{d.name}: 90% band-Mondrian coverage "
            f"{c90['coverage']:.3f}, width {c90['width']:.1f}"
        )


if __name__ == "__main__":
    main()
