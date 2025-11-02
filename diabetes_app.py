import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App title
st.title("🩺 Diabetes Disease Prediction App")
st.info("This app predicts whether a person is likely to have diabetes based on health data.")

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/mahnoormirjat11/Diabetic-disease-prediction/main/diabetes_prediction_dataset.csv")

# Show column names (for debug)
# st.write(df.columns)

# Display data
with st.expander("📊 View Dataset"):
    st.write("**Raw Data**")
    st.dataframe(df)
    st.write("**Shape:**", df.shape)

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Handle categorical columns (gender, smoking_history)
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
    age = st.sidebar.slider("Age", 1, 100, 30)
    hypertension = st.sidebar.selectbox("Hypertension", (0, 1))
    heart_disease = st.sidebar.selectbox("Heart Disease", (0, 1))
    smoking_history = st.sidebar.selectbox("Smoking History", ("never", "former", "current", "ever", "not current"))
    bmi = st.sidebar.slider("BMI", 10.0, 70.0, 25.0)
    HbA1c_level = st.sidebar.slider("HbA1c Level", 3.0, 9.0, 5.5)
    blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 50, 300, 120)
    
    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Match encoding with training data
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Display input
with st.expander("🧠 Input Data Preview"):
    st.write("**User Input (encoded)**")
    st.write(input_encoded)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# Display results
st.subheader("🎯 Prediction Result")
diabetes_result = np.array(["Not Diabetic", "Diabetic"])
st.success(f"**Prediction:** {diabetes_result[prediction][0]}")

st.subheader("📈 Prediction Probabilities")
st.dataframe(pd.DataFrame(prediction_proba, columns=["Not Diabetic", "Diabetic"]))

# Model accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.caption(f"Model Accuracy on Test Data: **{acc*100:.2f}%**")
