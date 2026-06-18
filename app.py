from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ==========================
# LOAD DIABETES MODEL
# ==========================

diabetes_model = joblib.load(
    "models/diabetes_model.pkl"
)

diabetes_scaler = joblib.load(
    "models/diabetes_scaler.pkl"
)

# ==========================
# LOAD MIGRAINE MODEL
# ==========================

migraine_model = joblib.load(
    "models/migraine_model.pkl"
)

migraine_encoder = joblib.load(
    "models/migraine_label_encoder.pkl"
)

# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# DIABETES PAGE
# ==========================

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    age = float(request.form["age"])
    glucose = float(request.form["glucose"])
    chol = float(request.form["chol"])
    hdl = float(request.form["hdl"])
    ratio = float(request.form["ratio"])
    bmi = float(request.form["bmi"])

    data = np.array([
        [age, glucose, chol, hdl, ratio, bmi]
    ])

    data_scaled = diabetes_scaler.transform(data)

    prediction = diabetes_model.predict(data_scaled)[0]

    probs = diabetes_model.predict_proba(data_scaled)

    confidence = round(
        max(probs[0]) * 100,
        2
    )

    if prediction == 1:
        result = "Diabetic"
        color = "red"
    else:
        result = "Non-Diabetic"
        color = "green"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        color=color
    )


# ==========================
# MIGRAINE PAGE
# ==========================

@app.route("/migraine")
def migraine():
    return render_template("migraine.html")


@app.route("/predict_migraine", methods=["POST"])
def predict_migraine():

    features = [[
        float(request.form["Age"]),
        float(request.form["Duration"]),
        float(request.form["Frequency"]),
        float(request.form["Location"]),
        float(request.form["Character"]),
        float(request.form["Intensity"]),
        float(request.form["Nausea"]),
        float(request.form["Vomit"]),
        float(request.form["Phonophobia"]),
        float(request.form["Photophobia"]),
        float(request.form["Visual"]),
        float(request.form["Sensory"]),
        float(request.form["Dysphasia"]),
        float(request.form["Dysarthria"]),
        float(request.form["Vertigo"]),
        float(request.form["Tinnitus"]),
        float(request.form["Hypoacusis"]),
        float(request.form["Diplopia"]),
        float(request.form["Defect"]),
        float(request.form["Ataxia"]),
        float(request.form["Conscience"]),
        float(request.form["Paresthesia"]),
        float(request.form["DPF"])
    ]]

    prediction = migraine_model.predict(features)[0]

    probs = migraine_model.predict_proba(features)

    confidence = round(
        max(probs[0]) * 100,
        2
    )

    migraine_type = migraine_encoder.inverse_transform(
        [prediction]
    )[0]

    return render_template(
        "result.html",
        prediction=migraine_type,
        confidence=confidence,
        color="#28a745"
    )


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)