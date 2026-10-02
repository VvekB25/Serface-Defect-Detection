from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ml.inference.predictor import DefectPredictor

from backend.app.core.config import CLASS_NAMES_PATH, MODEL_PATH


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a readable supported image."""


class PredictionService:
    """Own the single loaded predictor used by all prediction requests."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
    ) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self._predictor: DefectPredictor | None = None
        self.load_error: Exception | None = None

    @property
    def model_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        if self._predictor is not None:
            return
        self._predictor = DefectPredictor(
            model_path=self.model_path,
            class_names_path=self.class_names_path,
        )
        self.load_error = None

    def predict(self, contents: bytes) -> dict:
        if self._predictor is None:
            raise RuntimeError("The prediction model is unavailable")

        try:
            with Image.open(BytesIO(contents)) as uploaded_image:
                uploaded_image.load()
                image = uploaded_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise InvalidImageError("Invalid or corrupted image.") from error

        return self._predictor.predict(image)
