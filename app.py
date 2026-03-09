# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# -------------------------
# Load ML model (relative path)
# -------------------------
model_path = os.path.join('model', 'Electricity_consumption_prediction.pkl')

try:
    with open(model_path, 'rb') as f:
        modelpipeline = pickle.load(f)
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please check path and filename.")
    modelpipeline = None

# -------------------------
# Home page route
# -------------------------
@app.route("/home")
def home():
    return render_template("index.html")

# -------------------------
# Prediction route
# -------------------------
@app.route("/predict", methods=['POST'])
def predict():
    if modelpipeline is None:
        return render_template("index.html", prediction_text="Model not loaded. Check server logs.")

    # Extract form features
    try:
        features = [
            float(request.form['Global_reactive_power']),
            float(request.form['Voltage']),
            float(request.form['Sub_metering_1']),
            float(request.form['Sub_metering_2']),
            float(request.form['Sub_metering_3']),
            int(request.form['Hour']),
            int(request.form['DayOfWeek']),
            int(request.form['Month']),
            int(request.form['IsWeekend']),
            float(request.form['Rolling_3']),
            float(request.form['Rolling_5'])
        ]
    except Exception as e:
        return render_template("index.html", prediction_text=f"Invalid input: {e}")

    # Convert to array and predict
    input_data = np.array([features])
    prediction = modelpipeline.predict(input_data)
    output = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Predicted Power: {output} kW")

