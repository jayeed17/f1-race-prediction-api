from pydantic import BaseModel


class PredictionRequest(BaseModel):
    driver: str
    team: str
    circuit_name: str
    grid_position: int
    qualifying_position: int


class PredictionResponse(BaseModel):
    driver: str
    podium_probability: float
    podium_prediction: bool
