from flask import Flask, render_template, request
import pickle
import pandas as pd
import sqlite3
import os

app = Flask(__name__)

# Load ML model, scaler, and columns
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Database setup
DB_NAME = "database.db"
if not os.path.exists(DB_NAME):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            ca INTEGER,
            cp INTEGER,
            restecg INTEGER,
            slope INTEGER,
            thal INTEGER,
            prediction TEXT,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template("predict.html")  # your form page


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Convert numeric columns
        numeric_cols = ['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak','ca']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # One-hot encode categorical features
        categorical_cols = ['cp','restecg','slope','thal']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Reindex to match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Scale features
        df_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(df_scaled)[0]
        probability = round(model.predict_proba(df_scaled)[0][1] * 100, 2)
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'

        # Save to DB
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (
                age, sex, trestbps, chol, fbs, thalach, exang, oldpeak,
                ca, cp, restecg, slope, thal, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(data['age']), int(data['sex']), int(data['trestbps']), int(data['chol']),
            int(data['fbs']), int(data['thalach']), int(data['exang']), float(data['oldpeak']),
            int(data['ca']), int(data['cp']), int(data['restecg']), int(data['slope']),
            int(data['thal']), result, probability
        ))
        conn.commit()
        conn.close()

        return render_template("output.html", result=result, probability=probability)

    except Exception as e:
        return str(e)

@app.route('/admin')
def admin_dashboard():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    # Ensure records is always a list
    records = df.to_dict(orient="records") if not df.empty else []
    
    return render_template("admin_dashboard.html", records=records)
@app.route('/details')
def details():
    return render_template("details.html")  # the feature description page


if __name__ == '__main__':
    app.run(debug=True)
