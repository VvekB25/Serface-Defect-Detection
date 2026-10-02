import { useEffect, useRef, useState } from "react";
import { predictDefect } from "./services/api";

const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png"];
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const CLASS_LABELS = {
  crazing: "Crazing",
  inclusion: "Inclusion",
  patches: "Patches",
  pitted_surface: "Pitted surface",
  rolled_in_scale: "Rolled-in scale",
  scratches: "Scratches",
};

function validateFile(file) {
  if (!file) return "Please select an image first.";
  const lowerName = file.name.toLowerCase();
  const hasSupportedExtension = ACCEPTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
  const hasSupportedType = !file.type || ACCEPTED_TYPES.includes(file.type);
  if (!hasSupportedExtension || !hasSupportedType) return "Unsupported file. Please choose a JPG, JPEG, or PNG image.";
  return null;
}

function formatClassName(className) {
  return CLASS_LABELS[className] || className.replaceAll("_", " ");
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  function chooseFile(file) {
    const validationError = validateFile(file);
    setError(validationError || "");
    setPrediction(null);
    if (validationError) {
      setSelectedFile(null);
      setPreviewUrl("");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  async function handlePredict() {
    const validationError = validateFile(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }
    setLoading(true);
    setError("");
    setPrediction(null);
    try {
      setPrediction(await predictDefect(selectedFile));
    } catch (requestError) {
      setError(requestError.message || "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setSelectedFile(null);
    setPreviewUrl("");
    setPrediction(null);
    setError("");
    setDragActive(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const probabilityEntries = prediction
    ? Object.entries(prediction.probabilities).sort(([, first], [, second]) => second - first)
    : [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">SD</div>
        <div><p className="eyebrow">Inspection console / ML-01</p><h1>Industrial surface defect detection</h1></div>
        <div className="status-chip"><span /> CPU inference ready</div>
      </header>

      <section className="hero-copy">
        <div><p className="section-kicker">Visual quality control</p><h2>Inspect a surface image with the trained defect classifier.</h2></div>
        <p className="hero-note">Upload one inspection image to classify it across six NEU-DET defect categories.</p>
      </section>

      <section className="workspace-grid">
        <div className="panel upload-panel">
          <div className="panel-heading"><div><p className="section-kicker">01 / Input</p><h3>Inspection image</h3></div><span className="format-note">JPG · JPEG · PNG</span></div>
          <div
            className={`drop-zone ${dragActive ? "drop-zone-active" : ""} ${previewUrl ? "has-preview" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragActive(false); }}
            onDrop={(event) => { event.preventDefault(); setDragActive(false); chooseFile(event.dataTransfer.files?.[0]); }}
          >
            {previewUrl ? <img className="image-preview" src={previewUrl} alt="Selected inspection preview" /> : <><div className="upload-icon" aria-hidden="true">↥</div><p className="drop-title">Drop an image here</p><p className="drop-subtitle">or choose a file from your device</p></>}
            <input ref={fileInputRef} className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(event) => chooseFile(event.target.files?.[0])} aria-label="Choose inspection image" />
            <button className="button button-secondary choose-button" type="button" onClick={() => fileInputRef.current?.click()}>{previewUrl ? "Replace image" : "Choose image"}</button>
          </div>
          {selectedFile && <p className="file-meta">{selectedFile.name}<span>{(selectedFile.size / 1024).toFixed(0)} KB</span></p>}
          {error && <p className="error-message" role="alert">{error}</p>}
          <div className="action-row">
            <button className="button button-primary" type="button" onClick={handlePredict} disabled={loading || !selectedFile}>{loading ? "Analyzing image..." : "Detect defect"}</button>
            <button className="button button-quiet" type="button" onClick={clearAll} disabled={loading}>Clear</button>
          </div>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading"><div><p className="section-kicker">02 / Result</p><h3>Classification output</h3></div>{prediction && <span className="result-state">Complete</span>}</div>
          {prediction ? <div className="result-content">
            <div className="prediction-callout"><p className="result-label">Detected defect</p><h4>{formatClassName(prediction.prediction.class)}</h4><div className="confidence-line"><span>Confidence</span><strong>{(prediction.prediction.confidence * 100).toFixed(2)}%</strong></div></div>
            <div className="probability-section"><div className="probability-heading"><span>Class probabilities</span><span>Score</span></div>{probabilityEntries.map(([className, probability]) => <div className="probability-row" key={className}><div className="probability-label"><span>{formatClassName(className)}</span><strong>{(probability * 100).toFixed(2)}%</strong></div><div className="progress-track"><div className="progress-fill" style={{ width: `${Math.max(probability * 100, 1)}%` }} /></div></div>)}</div>
          </div> : <div className="empty-result"><div className="empty-line" /><p>Prediction results will appear here after analysis.</p><span>Waiting for an inspection image</span></div>}
        </div>
      </section>

      <footer className="footer-note"><span>MODEL</span> Hybrid CNN · 128 × 128 RGB · six classes</footer>
    </main>
  );
}

export default App;
