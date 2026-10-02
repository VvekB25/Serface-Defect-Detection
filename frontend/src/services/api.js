const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function predictDefect(file) {
  const formData = new FormData();
  formData.append("image", file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/predict`, { method: "POST", body: formData });
  } catch {
    throw new Error("Unable to connect to the prediction server. Please make sure the FastAPI backend is running.");
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The prediction server returned an invalid response.");
  }
  if (!response.ok) throw new Error(payload.detail || "Prediction failed. Please try again.");
  return payload;
}
