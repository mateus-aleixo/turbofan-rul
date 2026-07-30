"""Render the results figures into docs/figures/ from committed artifacts.

Reproducible from a fresh clone: reads models/*/preds.npz, conformal.json and
manifests — no retraining needed. Style follows a validated palette (first
three categorical slots pass all-pairs CVD checks on the light surface).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from conformal_rul.config import SUBSETS
from conformal_rul.conformal import ConformalRUL

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"gbm": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND_FILL = "#b7d3f6"

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=10, fontweight="bold", pad=10)


def serving_model(subset: str) -> str:
    return json.loads((MODELS / subset / "best.json").read_text())["model"]


def fig_intervals(subset: str = "FD004") -> None:
    model = serving_model(subset)
    d = MODELS / subset / model
    z = np.load(d / "preds.npz")
    cal = ConformalRUL.from_json((d / "conformal.json").read_text())
    lo, hi = cal.interval(z["test_pred"], z["test_regime"], 90, "band")
    y, pred = z["test_y"], np.clip(z["test_pred"][:, 0], 0, cal.rul_cap)

    order = np.argsort(-y)
    idx = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.fill_between(idx, lo[order], hi[order], color=BAND_FILL, linewidth=0,
                    label="90 % interval (CQR, Mondrian by RUL band)")
    ax.plot(idx, pred[order], color=BLUE, linewidth=1.4, label=f"predicted ({model})")
    ax.plot(idx, y[order], color=INK, linewidth=1.1, label="true RUL")
    ax.set_xlabel(f"{subset} test engines, sorted by true RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.set_xlim(0, len(y) - 1)
    ax.set_ylim(0, cal.rul_cap * 1.04)
    ax.grid(axis="y")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right", fontsize=8.5)
    _title(ax, f"Calibrated RUL intervals on {subset} — every engine gets an honest error bar")
    fig.savefig(OUT / f"intervals_{subset.lower()}.png", dpi=150)
    plt.close(fig)


def fig_coverage() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
    x = np.arange(len(SUBSETS))
    for ax, cov in zip(axes, (80, 90, 95), strict=True):
        for tax, color, dx in (("marginal", BLUE, -0.13), ("band", ORANGE, 0.13)):
            vals = []
            for s in SUBSETS:
                rep = json.loads(
                    (MODELS / s / serving_model(s) / "coverage_report.json").read_text()
                )
                vals.append(rep["coverages"][str(cov)][tax]["coverage"])
            ax.scatter(x + dx, vals, s=45, color=color, zorder=3,
                       label=tax if cov == 80 else None)
        ax.axhline(cov / 100, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xticks(x, SUBSETS, fontsize=8)
        ax.set_ylim(0.72, 1.005)
        ax.grid(axis="y")
        ax.set_title(f"nominal {cov} %", color=INK2, fontsize=9, pad=6)
    axes[0].set_ylabel("empirical coverage")
    fig.legend(frameon=False, labelcolor=INK2, loc="lower right",
               bbox_to_anchor=(0.995, 0.13), fontsize=8.5, title="calibration",
               title_fontsize=8.5)
    fig.suptitle("Test coverage vs nominal — serving model per subset",
                 x=0.005, ha="left", color=INK, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "coverage.png", dpi=150)
    plt.close(fig)


def fig_rmse() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(len(SUBSETS))
    width = 0.24
    for i, (model, color) in enumerate(SERIES.items()):
        vals = [
            json.loads((MODELS / s / model / "manifest.json").read_text())["test"]["rmse"]
            for s in SUBSETS
        ]
        ax.bar(x + (i - 1) * (width + 0.03), vals, width, color=color, label=model, zorder=3)
    ax.set_xticks(x, SUBSETS)
    ax.set_ylabel("test RMSE (cycles)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK2, ncols=3, loc="upper left", fontsize=8.5)
    ax.set_ylim(0, 18)
    _title(ax, "Point accuracy by subset — the boosted baseline is hard to beat")
    fig.savefig(OUT / "rmse.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_intervals("FD004")
    fig_coverage()
    fig_rmse()
    print(f"figures written to {OUT}")
