from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = next(
    (PROJECT_ROOT / "NEU-DET" / "validation" / "images").rglob("*.jpg")
)


def test_missing_image_is_rejected():
    with TestClient(app) as client:
        response = client.post("/api/predict")

    assert response.status_code == 400
    assert response.json()["detail"] == "An image file is required."


def test_unsupported_file_type_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            files={"image": ("sample.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_corrupt_image_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            files={"image": ("sample.jpg", b"not an image", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or corrupted image."


def test_valid_image_returns_prediction_structure():
    with TestClient(app) as client:
        with SAMPLE_IMAGE.open("rb") as image_file:
            response = client.post(
                "/api/predict",
                files={
                    "image": (
                        SAMPLE_IMAGE.name,
                        image_file,
                        "image/jpeg",
                    )
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(body["prediction"]) == {"class", "confidence"}
    assert isinstance(body["prediction"]["class"], str)
    assert 0.0 <= body["prediction"]["confidence"] <= 1.0
    assert set(body["probabilities"]) == {
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled_in_scale",
        "scratches",
    }
