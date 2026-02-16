from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

# -------------------------------
# LOAD MODEL & ENCODERS
# -------------------------------
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

print("✅ AI Loan + Fraud Engine Loaded")

# -------------------------------
# HOME
# -------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "AI-Based Smart Loan Approval Backend Running"
    })

# -------------------------------
# PREDICTION + FRAUD DETECTION
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # -------- FRAUD RULES --------
    fraud = False
    fraud_reasons = []

    if data["criminal"] == "yes":
        fraud = True
        fraud_reasons.append("Criminal record detected")

    if data["loan"] / data["income"] > 0.6:
        fraud = True
        fraud_reasons.append("High loan-to-income ratio")

    # -------- ML PREDICTION --------
    df = pd.DataFrame([data])

    for col, encoder in encoders.items():
        if col in df.columns:
            if df[col][0] not in encoder.classes_:
                df[col] = encoder.transform([encoder.classes_[0]])
            else:
                df[col] = encoder.transform(df[col])

    prediction = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][prediction])

    approved = bool(prediction) and not fraud

    return jsonify({
        "approved": approved,
        "confidence": round(confidence, 2),
        "fraud": fraud,
        "fraud_reasons": fraud_reasons
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
