# FastAPI Backend

This backend exposes the existing TensorFlow/Keras classification pipeline over HTTP. It does not define a second model, preprocessing pipeline, or prediction algorithm. Requests are passed to `ml.inference.predictor.DefectPredictor` through `PredictionService`.

## Install

From the project root:

```powershell
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Start

```powershell
python -m uvicorn backend.app.main:app --reload
```

The server runs at `http://127.0.0.1:8000`.

Interactive documentation is available at `/docs` and `/redoc`.

## Endpoints

### Health

```http
GET /api/health
```

```json
{"status": "ok", "model_loaded": true}
```

### Prediction

```http
POST /api/predict
Content-Type: multipart/form-data
```

The form field is `image`. Supported extensions are `.jpg`, `.jpeg`, and `.png`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -F "image=@path\\to\\image.jpg"
```

The response contains `success`, `prediction.class`, `prediction.confidence`, and the six class probabilities.

## Tests

```powershell
python -m pytest backend/tests -q
```

## CORS

The default development origins are `http://localhost:5173` and `http://127.0.0.1:5173`.
