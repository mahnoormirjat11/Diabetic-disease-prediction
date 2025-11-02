import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# App title
st.title("🩺 Diabetes Disease Prediction App")
st.info("This app predicts whether a person is likely to have diabetes based on health data.")

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/mahnoormirjat11/Diabetic-disease-prediction/main/diabetes_prediction_dataset.csv")

# Display data
with st.expander("📊 View Dataset"):
    st.write("**Raw Data**")
    st.dataframe(df)
    st.write("**Shape:**", df.shape)

# Features & Target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
    age = st.sidebar.slider("Age", 1, 100, 30)
    hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", (0, 1))
    heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", (0, 1))
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

# Encode user input
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

with st.expander("🧠 Encoded Input Data"):
    st.write(input_encoded)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# Display prediction
st.subheader("🎯 Prediction Result")
diabetes_result = np.array(["Not Diabetic", "Diabetic"])
st.success(f"**Prediction:** {diabetes_result[prediction][0]}")

st.subheader("📈 Prediction Probabilities")
st.dataframe(pd.DataFrame(prediction_proba, columns=["Not Diabetic", "Diabetic"]))

# Accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.caption(f"✅ Model Accuracy: **{acc*100:.2f}%**")

# -------------------------------
# ✅ Feature Importance Section
# -------------------------------
st.subheader("📌 Feature Importance (Model Explanation)")

importances = clf.feature_importances_
feature_names = X.columns

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feature_names, importances)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

# -------------------------------
# ✅ Compare user input to dataset averages
# -------------------------------
st.subheader("📍 Your Input vs Dataset Average")

comparison_df = pd.DataFrame({
    "User Input": input_df.iloc[0],
    "Dataset Mean": df.drop("diabetes", axis=1).mean()
})

st.dataframe(comparison_df)
