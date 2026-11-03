import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import BatchPredictionRequest, BatchPredictionResponse, DayInput, PredictionResponse

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_loader.load()
    except Exception as exc:  # keep the process alive; /health and /predict report 503
        print(f"[ERROR] Model loading failed: {exc}")
    yield
    print("[INFO] Bike Sharing Demand API shutting down...")


app = FastAPI(
    title="Bike Sharing Demand API",
    description="Lab 10 - ML model serving: predicts daily bike rentals",
    version="1.0.0",
    lifespan=lifespan,
)


def _log_kwargs():
    return dict(model_name=model_loader.model_name, model_version=model_loader.model_version, run_id=model_loader.run_id)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    log_inference(input_data={}, status="failure", error="Request validation failed", request_id=uuid.uuid4().hex[:12], **_log_kwargs())
    return JSONResponse(status_code=422, content={"status": "error", "message": "Request validation failed", "details": details})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    messages = {503: "Prediction service unavailable", 500: "Internal prediction error"}
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": messages.get(exc.status_code, str(exc.detail)),
                 "details": [{"field": "request", "message": str(exc.detail)}]},
    )


@app.get("/")
def root():
    return {"message": "Bike Sharing Demand API is running", "status": "success"}


@app.get("/health")
def health():
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model or preprocessor is not loaded")
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "service": "Bike Sharing Demand API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "model_uri": model_loader.model_uri,
        "run_id": model_loader.run_id,
        "model_metrics": model_loader.metadata.get("metrics"),
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


def _predict(dicts: list[dict]) -> np.ndarray:
    frame = clean_prediction_data(pd.DataFrame(dicts))
    X = model_loader.preprocessor.transform(frame)
    return np.clip(model_loader.model.predict(X), 0, None)  # rentals cannot be negative


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DayInput):
    request_id, start = uuid.uuid4().hex[:12], time.perf_counter()
    payload = data.model_dump(mode="json")

    if not model_loader.is_ready():
        log_inference(payload, latency_ms=0.0, status="failure", error="Model or preprocessor is not available", request_id=request_id, **_log_kwargs())
        raise HTTPException(status_code=503, detail="Model or preprocessor is not available")
    try:
        value = round(float(_predict([payload])[0]), 2)
        log_inference(payload, predicted_rentals=value, latency_ms=round((time.perf_counter() - start) * 1000, 3), request_id=request_id, **_log_kwargs())
        return {"predicted_rentals": value}
    except Exception as exc:
        log_inference(payload, latency_ms=round((time.perf_counter() - start) * 1000, 3), status="failure", error=str(exc), request_id=request_id, **_log_kwargs())
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    start = time.perf_counter()
    payloads = [d.model_dump(mode="json") for d in data.days]

    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model or preprocessor is not available")
    try:
        values = [round(float(v), 2) for v in _predict(payloads)]
        latency = round((time.perf_counter() - start) * 1000, 3)
        for payload, value in zip(payloads, values):
            log_inference(payload, predicted_rentals=value, latency_ms=latency, **_log_kwargs())
        return {"count": len(values), "predictions": [{"predicted_rentals": v} for v in values]}
    except Exception as exc:
        for payload in payloads:
            log_inference(payload, status="failure", error=str(exc), **_log_kwargs())
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch prediction")
