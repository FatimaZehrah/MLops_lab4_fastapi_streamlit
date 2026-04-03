from fastapi import FastAPI
from src.data import IrisData, IrisResponse
from src.predict import predict_species_with_confidence
import os
import json
from fastapi.responses import JSONResponse

app = FastAPI(title="Iris Classifier API")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
MODEL_DIR = os.path.join(BASE_DIR, "model")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

@app.get("/")
def home():
    return {"message": "FastAPI Iris Classifier is running successfully."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=IrisResponse)
def predict(data: IrisData):
    prediction, confidence = predict_species_with_confidence(data)
    return {"prediction": prediction, "confidence": confidence}

@app.get("/model-info")
def model_info():
    """
    Returns model metadata for UI display.
    """
    if not os.path.exists(METADATA_PATH):
        return JSONResponse(
            status_code=404,
            content={"error": "Model metadata not found", "path_checked": METADATA_PATH},
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Optional: indicate if a .pkl model file exists
    model_file_present = False
    if os.path.exists(MODEL_DIR):
        model_file_present = any(name.endswith(".pkl") for name in os.listdir(MODEL_DIR))

    metadata["model_file_present"] = model_file_present
    return metadata