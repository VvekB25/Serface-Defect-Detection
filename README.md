# Surface Defect Detection

Industrial surface defect classification on the NEU-DET dataset.

## Architecture

```text
React + Vite frontend
        |
        v
FastAPI backend
        |
        v
Python ML inference
        |
        v
TensorFlow/Keras classification model
```

The model classifies six defect types: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, and `scratches`. The trained model is `models/hybrid_defect_detection_model.keras`, with labels in `models/class_names.json`. Validation accuracy is 68.06%.

## Structure

- `ml/`: reusable training, preprocessing, model, evaluation, and inference modules
- `backend/`: FastAPI service exposing `/api/health` and `/api/predict`
- `frontend/`: React + Vite upload and classification UI
- `NEU-DET/`: original dataset
- `models/`: trained classifier artifacts

## Run locally

Install Python dependencies in a Python 3.10 or 3.11 virtual environment:

```powershell
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

Start the backend from the project root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

## Inference

The backend loads the saved classifier once at startup. It does not retrain the model for requests. The frontend sends an image in the `image` multipart field to `POST /api/predict` and displays the predicted class, confidence, and all six probabilities.

MongoDB history and object localization are future work and are not part of this checkpoint.
