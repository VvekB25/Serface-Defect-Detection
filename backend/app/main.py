from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.prediction import router as prediction_router
from backend.app.core.config import CORS_ORIGINS
from backend.app.schemas.prediction import HealthResponse
from backend.app.services.prediction_service import PredictionService

prediction_service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        prediction_service.load()
    except Exception as error:
        prediction_service.load_error = error
    yield


app = FastAPI(
    title="Industrial Surface Defect Detection API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(prediction_router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if prediction_service.model_loaded else "degraded",
        model_loaded=prediction_service.model_loaded,
    )
