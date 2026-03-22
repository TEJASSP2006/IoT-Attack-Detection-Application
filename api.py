import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from realtime_detector import RealtimeDetector

_LABEL_COLUMNS = {
    "label",
    "Label",
    "attack_label",
    "Attack_label",
    "attack",
    "Attack",
    "class",
    "Class",
    "target",
    "Target",
}
_MAX_CSV_ROWS = 500

load_dotenv()

app = FastAPI(title="IoT Real-Time Attack Detector", version="1.0.0")

_DASHBOARD_PATH = Path(__file__).resolve().parent / "static" / "dashboard.html"

try:
    detector = RealtimeDetector()
except Exception:
    detector = None


class PredictRequest(BaseModel):
    record: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    if not _DASHBOARD_PATH.is_file():
        raise HTTPException(status_code=500, detail="dashboard.html missing")
    return HTMLResponse(_DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(req: PredictRequest, _auth: None = Depends(verify_api_key)) -> dict:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first (python train.py ...).",
        )
    if not req.record:
        raise HTTPException(status_code=400, detail="record cannot be empty.")
    return detector.predict(req.record)


@app.post("/predict/batch")
def predict_batch(
    req: BatchPredictRequest, _auth: None = Depends(verify_api_key)
) -> dict[str, Any]:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first (python train.py ...).",
        )
    if not req.records:
        raise HTTPException(status_code=400, detail="records cannot be empty.")
    if len(req.records) > 100:
        raise HTTPException(
            status_code=400, detail="Maximum 100 records per batch request."
        )
    results: list[dict[str, Any]] = []
    for i, rec in enumerate(req.records):
        if not isinstance(rec, dict) or not rec:
            raise HTTPException(
                status_code=400, detail=f"Invalid record at index {i}: must be non-empty object.",
            )
        results.append({"row": i, **detector.predict(rec)})
    return {"count": len(results), "results": results}


def _row_to_json_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


@app.post("/predict/batch_csv")
async def predict_batch_csv(
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first (python train.py ...).",
        )
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    dropped = [c for c in df.columns if c in _LABEL_COLUMNS]
    df = df.drop(columns=dropped, errors="ignore")
    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="No feature columns left after removing label columns.",
        )
    truncated = False
    if len(df) > _MAX_CSV_ROWS:
        df = df.iloc[:_MAX_CSV_ROWS].copy()
        truncated = True
    records = df.to_dict("records")
    results: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        clean = _row_to_json_dict(rec)
        results.append({"row": i, **detector.predict(clean)})
    return {
        "count": len(results),
        "dropped_label_columns": dropped,
        "truncated_to_max_rows": truncated,
        "max_rows": _MAX_CSV_ROWS,
        "results": results,
    }


@app.get("/stats")
def prediction_stats(_auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first (python train.py ...).",
        )
    return detector.storage.stats()


@app.get("/predictions")
def recent_predictions(limit: int = 20, _auth: None = Depends(verify_api_key)) -> dict:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first (python train.py ...).",
        )
    return {"items": detector.storage.recent(limit=min(max(limit, 1), 200))}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": detector is not None}
