from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse

MODEL_PATH = Path(__file__).parent / "model" / "podium_model.joblib"

model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="F1 Race Prediction API",
    description="Predicts whether an F1 driver will finish on the podium.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"message": "F1 Race Prediction API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_podium(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_df = pd.DataFrame([request.model_dump()])
    podium_probability = float(model.predict_proba(input_df)[0, 1])

    return PredictionResponse(
        driver=request.driver,
        podium_probability=round(podium_probability, 4),
        podium_prediction=podium_probability >= 0.5,
    )
