import logging
import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ----------------------------------------------------------------------------
# App & logging setup
# ----------------------------------------------------------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ----------------------------------------------------------------------------
# Field definitions — single source of truth for the form (rendered in the
# template) AND for server-side validation below. Keys ("id") are the names
# the HTML inputs use; they map to FEATURE_ORDER for the model's real column
# names.
# ----------------------------------------------------------------------------

FIELD_SPECS = [
    {
        "id": "Age", "label": "Age", "type": "number", "unit": "years",
        "min": 1, "max": 120, "step": 1, "placeholder": "e.g. 54",
        "help": "Patient's age in years.",
    },
    {
        "id": "Sex", "label": "Sex", "type": "select",
        "help": "Biological sex recorded for the patient.",
        "options": [("1", "Male"), ("0", "Female")],
    },
    {
        "id": "ChestPain", "label": "Chest Pain Type", "type": "select",
        "help": "Category of chest pain reported by the patient.",
        "options": [
            ("1", "Typical angina"),
            ("2", "Atypical angina"),
            ("3", "Non-anginal pain"),
            ("4", "Asymptomatic"),
        ],
    },
    {
        "id": "BP", "label": "Resting Blood Pressure", "type": "number", "unit": "mm Hg",
        "min": 50, "max": 260, "step": 1, "placeholder": "e.g. 130",
        "help": "Resting blood pressure on admission to hospital.",
    },
    {
        "id": "Cholesterol", "label": "Serum Cholesterol", "type": "number", "unit": "mg/dl",
        "min": 80, "max": 650, "step": 1, "placeholder": "e.g. 246",
        "help": "Serum cholesterol level.",
    },
    {
        "id": "FBS", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select",
        "help": "Is fasting blood sugar greater than 120 mg/dl?",
        "options": [("1", "Yes"), ("0", "No")],
    },
    {
        "id": "EKG", "label": "Resting EKG Results", "type": "select",
        "help": "Resting electrocardiographic results.",
        "options": [
            ("0", "Normal"),
            ("1", "ST-T wave abnormality"),
            ("2", "Left ventricular hypertrophy"),
        ],
    },
    {
        "id": "MaxHR", "label": "Maximum Heart Rate", "type": "number", "unit": "bpm",
        "min": 60, "max": 230, "step": 1, "placeholder": "e.g. 150",
        "help": "Maximum heart rate achieved during exercise.",
    },
    {
        "id": "Exercise", "label": "Exercise-Induced Angina", "type": "select",
        "help": "Was angina induced by exercise?",
        "options": [("1", "Yes"), ("0", "No")],
    },
    {
        "id": "ST", "label": "ST Depression", "type": "number", "unit": "",
        "min": 0, "max": 10, "step": 0.1, "placeholder": "e.g. 1.0",
        "help": "ST depression induced by exercise, relative to rest.",
    },
    {
        "id": "Slope", "label": "Slope of Peak Exercise ST", "type": "select",
        "help": "Slope of the peak exercise ST segment.",
        "options": [("1", "Upsloping"), ("2", "Flat"), ("3", "Downsloping")],
    },
    {
        "id": "Vessels", "label": "Number of Major Vessels", "type": "select",
        "help": "Number of major vessels colored by fluoroscopy.",
        "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")],
    },
    {
        "id": "Thallium", "label": "Thallium Stress Test", "type": "select",
        "help": "Result of the thallium stress test.",
        "options": [("3", "Normal"), ("6", "Fixed defect"), ("7", "Reversible defect")],
    },
]

# Exact column order the scaler/model were fit on.
FEATURE_ORDER = [spec["id"] for spec in FIELD_SPECS]

# Fast lookup for validation.
FIELD_SPECS_BY_ID = {spec["id"]: spec for spec in FIELD_SPECS}

# Valid codes for each dropdown, used to reject anything not seen during training.
VALID_SELECT_VALUES = {
    spec["id"]: {value for value, _ in spec["options"]}
    for spec in FIELD_SPECS if spec["type"] == "select"
}

# ----------------------------------------------------------------------------
# Load model & scaler ONCE at startup (not per-request).
# ----------------------------------------------------------------------------

logger.info("Loading scaler from %s", SCALER_PATH)
scaler = joblib.load(SCALER_PATH)

logger.info("Loading Keras model from %s", MODEL_PATH)

from tensorflow.keras.models import load_model  # noqa: E402

model = load_model(MODEL_PATH)
logger.info("Model and scaler loaded successfully. Ready to serve predictions.")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when submitted form data fails validation."""


def validate_and_extract(form_data):
    """
    Validate incoming form data against FIELD_SPECS and return a list of
    floats in FEATURE_ORDER, ready for scaler.transform().

    Raises ValidationError with a human-readable message on any problem.
    """
    values = []
    for spec in FIELD_SPECS:
        field_id = spec["id"]
        raw = form_data.get(field_id, "").strip()

        if raw == "":
            raise ValidationError(f"'{spec['label']}' is required.")

        if spec["type"] == "select":
            if raw not in VALID_SELECT_VALUES[field_id]:
                raise ValidationError(f"'{spec['label']}' has an invalid selection.")
            values.append(float(raw))
        else:
            try:
                number = float(raw)
            except ValueError:
                raise ValidationError(f"'{spec['label']}' must be a number.")

            if not (spec["min"] <= number <= spec["max"]):
                raise ValidationError(
                    f"'{spec['label']}' must be between {spec['min']} and {spec['max']}."
                )
            values.append(number)

    return values


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def home():
    """Render the input form."""
    return render_template("index.html", fields=FIELD_SPECS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Validate inputs, run them through the scaler + model exactly as the
    notebook did, and return a JSON payload the front end renders without a
    full page reload.
    """
    try:
        features = validate_and_extract(request.form)

        # Shape (1, 13) — one row, feature order == FEATURE_ORDER == scaler's
        # training column order.
        data = np.array([features])
        data_scaled = scaler.transform(data)

        raw_prediction = model.predict(data_scaled, verbose=0)
        probability_presence = float(raw_prediction[0][0])

        if probability_presence > 0.5:
            label = "Heart Disease Detected"
            confidence = probability_presence
        else:
            label = "No Heart Disease Detected"
            confidence = 1 - probability_presence

        return jsonify({
            "success": True,
            "prediction": label,
            "confidence_pct": round(confidence * 100, 2),
            "probability_presence_pct": round(probability_presence * 100, 2),
        })

    except ValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception:
        logger.exception("Unexpected error while generating a prediction")
        return jsonify({
            "success": False,
            "error": "Something went wrong while processing your request. Please try again.",
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Not found."}), 404


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
