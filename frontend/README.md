# React + Vite Frontend

This frontend sends inspection images to the existing FastAPI backend and displays the real ML classification prediction. It does not contain TensorFlow, model code, or duplicate preprocessing.

## Install and start

```powershell
cd frontend
npm install
npm run dev
```

The backend should be running at `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` in `.env.local` when another backend URL is needed; `.env.example` contains the development default.

The interface supports JPG, JPEG, and PNG drag-and-drop or file selection, local preview, loading and error states, classification results, six probability bars, and reset.
