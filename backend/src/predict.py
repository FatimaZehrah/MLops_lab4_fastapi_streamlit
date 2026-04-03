import joblib
import numpy as np
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "iris_model.pkl")


def predict_species_with_confidence(data):
    model = joblib.load(MODEL_PATH)

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    pred = model.predict(features)[0]

    # Probability for the predicted class
    proba = model.predict_proba(features)[0]
    confidence = float(proba[int(pred)])

    return int(pred), confidence
