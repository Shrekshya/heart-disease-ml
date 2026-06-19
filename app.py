import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved models and scaler from notebook
with open('lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('dt_model.pkl', 'rb') as f:
    dt = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App UI
st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")
st.write("Models trained using 70/30 split. Best model: KNN (85.71% accuracy)")

st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 20, 80, 54)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 130)
chol = st.sidebar.slider("Cholesterol", 100, 570, 246)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 149)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.5, 1.0)
slope = st.sidebar.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (0-3)", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    lr_pred = lr.predict(input_scaled)[0]
    knn_pred = knn.predict(input_scaled)[0]
    dt_pred = dt.predict(input_scaled)[0]

    def result(pred):
        return "🔴 Has Heart Disease" if pred == 1 else "🟢 No Heart Disease"

    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Logistic Regression")
        st.write(result(lr_pred))
    with col2:
        st.subheader("KNN")
        st.write(result(knn_pred))
    with col3:
        st.subheader("Decision Tree")
        st.write(result(dt_pred))

    # Confidence scores
    st.subheader("Prediction Confidence")
    lr_conf = lr.predict_proba(input_scaled)[0][1]
    knn_conf = knn.predict_proba(input_scaled)[0][1]
    dt_conf = dt.predict_proba(input_scaled)[0][1]

    conf_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'KNN', 'Decision Tree'],
        'Confidence (Has Disease)': [f"{lr_conf:.2%}", f"{knn_conf:.2%}", f"{dt_conf:.2%}"]
    })
    st.table(conf_df)

    st.subheader("Model Performance Summary")
    perf_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'KNN', 'Decision Tree'],
        'Accuracy': ['80.52%', '85.71%', '84.42%'],
        'Has Disease Recall': ['87%', '87%', '96%'],
        'Best For': ['Simple Baseline', 'Overall Accuracy', 'Medical Safety']
    })
    st.table(perf_df)