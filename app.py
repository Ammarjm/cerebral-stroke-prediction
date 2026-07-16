from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# Load the Logistic Regression model
bundle = joblib.load("models/logistic_stroke_model.joblib")

model = bundle['model']

scaler = bundle['scaler']

label_encoders = bundle['label_encoders']

feature_columns = bundle['feature_columns']

# Set up page
st.title('Stroke Prediction App')

st.write('Enter patient information')

# Inputs from user
Gender = st.selectbox(
    'Gender', 
    label_encoders['gender'].classes_.tolist()
    )

Age = st.number_input(
    'Age', min_value=0.0, 
    max_value=120.0, 
    value=40.0
    )

Hypertension = st.selectbox(
    'Hypertension', 
    ['No', 'Yes'])

Heart_disease = st.selectbox(
    'Heart Disease', [
        'No', 'Yes'])

Ever_married = st.selectbox(
    'Ever Married', 
    label_encoders['ever_married'].classes_.tolist())

Work_type = st.selectbox(
    'Work Type', 
    label_encoders['work_type'].classes_.tolist())

Residence_type = st.selectbox(
    'Residence Type',
      label_encoders['Residence_type'].classes_.tolist())

Avg_glucose_level = st.number_input(
    'Average Glucose Level',
      min_value=0.0, 
      value=100.0)

Bmi = st.number_input(
    'BMI', 
    min_value=0.0, 
    value=25.0)

Smoking_status = st.selectbox(
    'Smoking Status', 
    label_encoders['smoking_status'].classes_.tolist())

# Make prediction
if st.button('Predict'):
    gender_num = label_encoders['gender'].transform([Gender])[0]

    married_num = label_encoders['ever_married'].transform([Ever_married])[0]

    work_num = label_encoders['work_type'].transform([Work_type])[0]

    residence_num = label_encoders['Residence_type'].transform([Residence_type])[0]

    smoking_num = label_encoders['smoking_status'].transform([Smoking_status])[0]

    hypertension_num = 1 if Hypertension == 'Yes' else 0

    heart_disease_num = 1 if Heart_disease == 'Yes' else 0

    input_data = pd.DataFrame([{
        "gender": gender_num,
        "age": Age,
        "hypertension": hypertension_num,
        "heart_disease": heart_disease_num,
        "ever_married": married_num,
        "work_type": work_num,
        "Residence_type": residence_num,
        "avg_glucose_level": Avg_glucose_level,
        "bmi": Bmi,
        "smoking_status": smoking_num
    }])
    
    # arrange the columns to match training sequence
    input_data = input_data.reindex(columns=feature_columns)

    # convert to float type
    input_data = input_data.astype(float)

    # scaling
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.write(f'Probability of stroke : {probability:.2%}')

    if prediction == 1:
        st.error('Prediction: Stroke Risk')
        
    else:
        st.success('Prediction: No Stroke Risk')