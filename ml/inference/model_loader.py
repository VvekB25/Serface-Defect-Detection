import json
from dataclasses import dataclass
from pathlib import Path

from tensorflow.keras.models import load_model

from ml.config.settings import CLASS_NAMES_PATH, MODEL_PATH


@dataclass(frozen=True)
class InferenceArtifacts:
    model: object
    class_names: dict[str, str]


def load_trained_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run training first."
        )
    return load_model(model_path)


def load_inference_artifacts(
    model_path: str | Path = MODEL_PATH,
    class_names_path: str | Path = CLASS_NAMES_PATH,
) -> InferenceArtifacts:
    """Load the model and its output-index-to-class mapping once."""
    class_names_path = Path(class_names_path)
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found at {class_names_path}. Run training first."
        )

    model = load_trained_model(model_path)
    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, dict) or not class_names:
        raise ValueError("Class mapping must be a non-empty JSON object")

    expected_indices = {str(index) for index in range(len(class_names))}
    if set(class_names) != expected_indices:
        raise ValueError("Class mapping keys must be contiguous output indices starting at 0")
    output_count = model.output_shape[-1]
    if output_count != len(class_names):
        raise ValueError(
            f"Model has {output_count} outputs but class mapping has {len(class_names)} classes"
        )

    return InferenceArtifacts(model=model, class_names=class_names)
