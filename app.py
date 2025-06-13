import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('credit_risk_model.pkl')  # change if you're using logistic

st.title("🏦 Credit Risk Prediction")
st.write("Provide your details to check loan eligibility and credit score")

# User inputs
Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=1)
Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=1)
Credit_History = st.selectbox("Credit History", ['1.0', '0.0'])

# Encoders
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
edu_map = {'Graduate': 0, 'Not Graduate': 1}
self_emp_map = {'Yes': 1, 'No': 0}
prop_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

# Predict button
if st.button("🔍 Predict Loan Approval"):
    Total_Income = Applicant_Income + Coapplicant_Income
    EMI = Loan_Amount / Loan_Amount_Term
    Debt_Income_Ratio = Loan_Amount / Total_Income if Total_Income != 0 else 0

    input_data = pd.DataFrame([{
        'Gender': gender_map[Gender],
        'Married': married_map[Married],
        'Dependents': dep_map[Dependents],
        'Education': edu_map[Education],
        'Self_Employed': self_emp_map[Self_Employed],
        'Property_Area': prop_map[Property_Area],
        'Loan_Amount': Loan_Amount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': float(Credit_History),
        'Total_Income': Total_Income,
        'EMI': EMI,
        'Debt_Income_Ratio': Debt_Income_Ratio
    }])
    # Reorder columns to match training
    feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                     'Loan_Amount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                     'Total_Income', 'EMI', 'Debt_Income_Ratio']
    input_data = input_data[feature_order]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= 0.65 else 0  # adjust threshold as needed
    credit_score = int(probability * 100)

    st.subheader("📊 Prediction Result")
    if prediction == 1:
       st.success("✅ Loan Approved")
       st.metric("📈 Approval Probability", f"{probability:.2f}")
       st.metric("💳 Customer Credit Score", f"{credit_score}/100")
    else:
       st.error("❌ Loan Rejected")
       st.metric("📉 Approval Probability", f"{probability:.2f}")
       st.metric("💳 Customer Credit Score", f"{credit_score}/100")

