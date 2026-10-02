from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_ROOT = PROJECT_ROOT / "NEU-DET"
PROCESSED_DATASET_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hybrid_defect_detection_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
)

CLASS_ALIASES = {
    "rolled-in_scale": "rolled_in_scale",
}
