from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

# -------------------------------
# LOAD MODEL & ENCODERS
# -------------------------------
# Using try-except to prevent crash if files are missing
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    print("✅ AI Loan + Fraud Engine Loaded")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Define the exact features your model expects (update this list based on your training)
FEATURE_COLUMNS = ["purpose", "income", "loan", "employment", "criminal"]

@app.route("/")
def home():
    return jsonify({"message": "AI-Based Smart Loan Approval Backend Running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # -------- FRAUD RULES (Heuristics) --------
        fraud = False
        fraud_reasons = []

        if data.get("criminal") == "yes":
            fraud = True
            fraud_reasons.append("Criminal record detected")

        # Prevent division by zero if income is 0
        income = float(data.get("income", 1))
        loan = float(data.get("loan", 0))
        
        if (loan / income) > 0.6:
            fraud = True
            fraud_reasons.append("High loan-to-income ratio")

        # -------- ML PREDICTION --------
        # 1. Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # 2. Filter only relevant features (remove 'name', etc.)
        df = input_df[FEATURE_COLUMNS].copy()

        # 3. Apply Encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                # Handle unknown categories safely
                if val not in encoder.classes_:
                    df[col] = encoder.transform([encoder.classes_[0]])
                else:
                    df[col] = encoder.transform([val])

        # 4. Predict
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        confidence = float(probabilities[prediction])

        # Approval logic: must be ML-approved AND pass fraud rules
        approved = bool(prediction == 1) and not fraud

        return jsonify({
            "approved": approved,
            "score": round(confidence * 100, 1), # Send as percentage for frontend
            "fraud": fraud,
            "reason": " | ".join(fraud_reasons) if fraud_reasons else "Healthy fiscal profile"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"approved": False, "score": 0, "reason": "Internal Engine Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
