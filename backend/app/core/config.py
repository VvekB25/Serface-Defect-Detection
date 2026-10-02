import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_defect_detection_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
