# diabetes_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# 🩺 App Title and Intro
# ---------------------------------------------------------------
st.set_page_config(page_title="Diabetes Disease Prediction", layout="wide")
st.title("🩺 Advanced Diabetes Disease Prediction App (with EDA + XGBoost)")
st.info("This app predicts diabetes likelihood using advanced ML techniques and EDA.")

# ---------------------------------------------------------------
# 📥 Load Dataset
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mahnoormirjat11/Diabetic-disease-prediction/main/diabetes_prediction_dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ---------------------------------------------------------------
# 🧩 Data Overview
# ---------------------------------------------------------------
st.header("1️⃣ Dataset Overview")
with st.expander("📊 View Dataset"):
    st.write(df.head())
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))

# ---------------------------------------------------------------
# 🧹 Preprocessing
# ---------------------------------------------------------------
st.header("2️⃣ Data Preprocessing")

# Detect target column
target_col = "Outcome" if "Outcome" in df.columns else "diabetes"

# Encode categorical columns if any
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# ---------------------------------------------------------------
# 📊 Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------
st.header("3️⃣ Exploratory Data Analysis (EDA)")

with st.expander("📈 Statistical Summary"):
    st.write(df.describe())

with st.expander("📉 Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

with st.expander("📊 Distribution of Target Variable"):
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="pastel", ax=ax)
    plt.title("Diabetes vs Non-Diabetic Count")
    st.pyplot(fig)

# ---------------------------------------------------------------
# 🧠 Model Training (XGBoost)
# ---------------------------------------------------------------
st.header("4️⃣ Model Training (XGBoost Classifier)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize XGBoost
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: **{acc * 100:.2f}%**")

with st.expander("📋 Classification Report"):
    st.text(classification_report(y_test, y_pred))

with st.expander("📊 Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------------------------------------------------------
# 🧮 User Input + Prediction
# ---------------------------------------------------------------
st.header("5️⃣ Try Prediction")

st.sidebar.header("Enter Patient Details")

def user_input():
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
        "Age": age,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()
st.write("### 🧾 User Input Data", input_df)

# Preprocess input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display result
st.subheader("🎯 Prediction Result")
result_label = np.array(["Not Diabetic", "Diabetic"])
st.success(f"Prediction: **{result_label[prediction][0]}**")

# Display probabilities
st.write("### 📊 Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=["Not Diabetic", "Diabetic"])
st.dataframe(proba_df)

# ---------------------------------------------------------------
# ✅ End of App
# ---------------------------------------------------------------
st.caption("Made with ❤️ by Mahnoor — Powered by Streamlit & XGBoost")
