from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.core.config import ALLOWED_IMAGE_EXTENSIONS
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.prediction_service import (
    InvalidImageError,
    PredictionService,
)

router = APIRouter()


def get_prediction_service() -> PredictionService:
    from backend.app.main import prediction_service

    return prediction_service


@router.post("/predict", response_model=PredictionResponse)
async def predict_defect(image: UploadFile | None = File(default=None)):
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An image file is required.",
        )

    filename = image.filename or ""
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if f".{extension}" not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type. Use JPG, JPEG, or PNG.",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    service = get_prediction_service()
    if not service.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is unavailable.",
        )

    try:
        result = service.predict(contents)
    except InvalidImageError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from error

    return {
        "success": True,
        "prediction": {
            "class": result["predicted_class"],
            "confidence": result["confidence"],
        },
        "probabilities": result["probabilities"],
    }
