"""API contract. One reading per engine cycle: 3 operating settings and the
21 raw sensor channels, exactly as they appear in the C-MAPSS column layout —
the service owns normalization and sensor selection, clients send raw data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

from conformal_rul.data import SENSOR_COLS, SETTING_COLS

CycleReading = create_model(
    "CycleReading",
    **{name: (float, ...) for name in [*SETTING_COLS, *SENSOR_COLS]},
)


class PredictRequest(BaseModel):
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    cycles: list[CycleReading] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Consecutive cycles, oldest first. The service uses the "
        "trailing window and left-pads short histories.",
    )
    coverage: Literal[80, 90, 95] = 90
    taxonomy: Literal["marginal", "regime", "band"] = Field(
        "band",
        description="Which Mondrian calibration group sets the interval width.",
    )


class Interval(BaseModel):
    lower: float
    upper: float
    coverage: int


class PredictResponse(BaseModel):
    rul_cycles: float = Field(..., description="Point estimate, capped at the training cap.")
    interval: Interval
    risk_band: Literal["critical", "warning", "healthy"]
    operating_regime: int
    subset: str
    model: str
    n_cycles_used: int
    padded: bool = Field(..., description="True if history was shorter than the model window.")


class ModelInfo(BaseModel):
    subset: str
    model: str
    window: int
    rul_cap: int
    n_features: int
    n_regimes: int
    test_rmse: float
    conformal_method: str
