from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="F1 Race Prediction API",
    description="Predicts whether an F1 driver will finish on the podium.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "F1 Race Prediction API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_podium(request: PredictionRequest):
    # TODO: replace with real model inference once trained.
    raise NotImplementedError(f"Prediction model is not implemented yet (driver={request.driver}).")
