"""FastAPI service over the exported ONNX models.

Torch-free by design: inference is onnxruntime, preprocessing and conformal
state are plain JSON + numpy. MODEL_ROOT points at the models/ registry
(baked into the container image). The module exposes `handler` for AWS
Lambda via Mangum and runs locally with uvicorn.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from conformal_rul import __version__
from conformal_rul.config import RUL_BAND_NAMES
from conformal_rul.conformal import ConformalRUL, rul_band
from conformal_rul.data import Preprocessor, _pad_to

from .schemas import Interval, ModelInfo, PredictRequest, PredictResponse

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))

app = FastAPI(
    title="conformal-rul",
    version=__version__,
    description="Remaining-useful-life prediction with conformal intervals "
    "on the NASA C-MAPSS benchmark.",
)


class Bundle:
    def __init__(self, subset: str):
        subset_dir = MODEL_ROOT / subset
        if not (subset_dir / "best.json").exists():
            raise FileNotFoundError(f"no serving model for {subset} under {MODEL_ROOT}")
        best = json.loads((subset_dir / "best.json").read_text())
        model_dir = subset_dir / best["model"]

        import onnxruntime as ort

        self.model_name: str = best["model"]
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.pre = Preprocessor.from_json((model_dir / "preprocessor.json").read_text())
        self.conformal = ConformalRUL.from_json((model_dir / "conformal.json").read_text())
        self.manifest = json.loads((model_dir / "manifest.json").read_text())


@lru_cache(maxsize=8)
def bundle(subset: str) -> Bundle:
    try:
        return Bundle(subset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    out = []
    for best_file in sorted(MODEL_ROOT.glob("*/best.json")):
        b = bundle(best_file.parent.name)
        out.append(
            ModelInfo(
                subset=b.pre.subset,
                model=b.model_name,
                window=b.pre.window,
                rul_cap=b.pre.rul_cap,
                n_features=b.pre.n_features,
                n_regimes=b.pre.n_regimes,
                test_rmse=b.manifest["test"]["rmse"],
                conformal_method=b.conformal.method,
            )
        )
    if not out:
        raise HTTPException(status_code=503, detail=f"no models under {MODEL_ROOT}")
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    b = bundle(req.subset)
    df = pd.DataFrame([c.model_dump() for c in req.cycles])
    feat, regime = b.pre.transform(df)

    window = b.pre.window
    padded = len(feat) < window
    feat = _pad_to(feat, window) if padded else feat[-window:]

    pred = b.session.run(None, {"windows": feat[None].astype(np.float32)})[0]
    pred[:, 1:] = np.sort(pred[:, 1:], axis=1)  # same monotonization as training

    last_regime = regime[-1:]
    lo, hi = b.conformal.interval(pred, last_regime, req.coverage, req.taxonomy)
    point = float(np.clip(pred[0, 0], 0, b.pre.rul_cap))
    band = RUL_BAND_NAMES[int(rul_band(pred[:1, 0], b.pre.rul_cap)[0])]

    return PredictResponse(
        rul_cycles=round(point, 1),
        interval=Interval(lower=round(float(lo[0]), 1), upper=round(float(hi[0]), 1),
                          coverage=req.coverage),
        risk_band=band,
        operating_regime=int(last_regime[0]),
        subset=req.subset,
        model=b.model_name,
        n_cycles_used=min(len(req.cycles), window),
        padded=padded,
    )


try:  # Lambda entrypoint; absent locally unless the serve extra is installed
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
