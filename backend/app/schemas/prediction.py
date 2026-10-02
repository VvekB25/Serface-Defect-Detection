from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
