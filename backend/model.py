import joblib
import numpy as np

model = joblib.load("depression_model.pkl")


def predict_stress(features: np.ndarray):
    # Prediction label
    pred = model.predict([features])[0]

    # Real probability
    prob = model.predict_proba([features])[0][pred]

    return {
        "stress_level": "Depressed" if pred == 1 else "Normal",
        "confidence": round(float(prob), 2),
    }
