import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get raw data
        gender = request.form['gender']
        senior_citizen = int(request.form['senior_citizen'])
        partner = request.form['partner']
        dependents = request.form['dependents']
        tenure = int(request.form['tenure'])
        phone_service = request.form['phone_service']
        multiple_lines = request.form['multiple_lines']
        internet_service = request.form['internet_service']
        online_security = request.form['online_security']
        online_backup = request.form['online_backup']
        device_protection = request.form['device_protection']
        tech_support = request.form['tech_support']
        streaming_tv = request.form['streaming_tv']
        streaming_movies = request.form['streaming_movies']
        contract = request.form['contract']
        paperless_billing = request.form['paperless_billing']
        payment_method = request.form['payment_method']
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])

        # Step 2: ⭐ MAP BINARY COLUMNS TO 0/1 (THIS IS THE FIX!)
        gender = 1 if gender == 'Male' else 0
        partner = 1 if partner == 'Yes' else 0
        dependents = 1 if dependents == 'Yes' else 0
        phone_service = 1 if phone_service == 'Yes' else 0
        multiple_lines = 1 if multiple_lines == 'Yes' else 0
        online_security = 1 if online_security == 'Yes' else 0
        online_backup = 1 if online_backup == 'Yes' else 0
        device_protection = 1 if device_protection == 'Yes' else 0
        tech_support = 1 if tech_support == 'Yes' else 0
        streaming_tv = 1 if streaming_tv == 'Yes' else 0
        streaming_movies = 1 if streaming_movies == 'Yes' else 0
        paperless_billing = 1 if paperless_billing == 'Yes' else 0

        # Step 3: Create DataFrame with NUMBERS (no strings!)
        customer_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        # Step 4: Transform
        transformed = preprocessor.transform(customer_data)

        # Step 5: Predict
        prob = model.predict_proba(transformed)[0][1]
        pred = model.predict(transformed)[0]

        # Risk level
        if prob > 0.7:
            risk = "High"
            risk_class = "high-risk"
        elif prob > 0.4:
            risk = "Medium"
            risk_class = "medium-risk"
        else:
            risk = "Low"
            risk_class = "low-risk"

        return render_template('result.html',
                             probability=f"{prob:.2%}",
                             prediction="Yes" if pred == 1 else "No",
                             risk=risk,
                             risk_class=risk_class)

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)