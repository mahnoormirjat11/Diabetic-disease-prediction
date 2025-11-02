import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App title
st.title("🩺 Diabetes Disease Prediction App")
st.info("This app predicts whether a person is likely to have diabetes based on health data.")

df = pd.read_csv("https://raw.githubusercontent.com/mahnoormirjat11/Diabetic-disease-prediction/main/diabetes_prediction_dataset.csv")


# Display data
with st.expander("📊 View Dataset"):
    st.write("**Raw Data**")
    st.dataframe(df)
    st.write("**Shape:**", df.shape)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for input features
st.sidebar.header("Enter Patient Details")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
    age = st.sidebar.slider("Age", 21, 81, 33)
    
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Combine user input with full dataset for encoding consistency
df_combined = pd.concat([input_df, X], axis=0)

# Display input
with st.expander("🧠 Input Data Preview"):
    st.write("**User Input**")
    st.write(input_df)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display prediction results
st.subheader("🎯 Prediction Result")
diabetes_result = np.array(["Not Diabetic", "Diabetic"])
st.success(f"**Prediction:** {diabetes_result[prediction][0]}")

st.subheader("📈 Prediction Probabilities")
st.dataframe(pd.DataFrame(prediction_proba, columns=["Not Diabetic", "Diabetic"]))

# Model performance
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.caption(f"Model Accuracy on Test Data: **{acc*100:.2f}%**")
