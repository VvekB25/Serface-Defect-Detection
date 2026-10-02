from pathlib import Path

import numpy as np

from ml.config.settings import CLASS_NAMES_PATH, MODEL_PATH
from ml.inference.model_loader import load_inference_artifacts
from ml.preprocessing.image_preprocessor import ImageInput, preprocess_image


class DefectPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        artifacts = load_inference_artifacts(model_path, class_names_path)
        self.model = artifacts.model
        self.class_names = artifacts.class_names

    def predict(self, image_input: ImageInput) -> dict:
        probabilities = self.model.predict(preprocess_image(image_input), verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        if str(predicted_index) not in self.class_names:
            raise ValueError(
                f"Model output index {predicted_index} is missing from class mapping"
            )
        return {
            "predicted_class": self.class_names[str(predicted_index)],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                self.class_names[str(index)]: float(probability)
                for index, probability in enumerate(probabilities)
            },
        }
