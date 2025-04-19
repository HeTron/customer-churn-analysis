import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("churn_data.csv")

# Clean data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode
df_encoded = df.copy()
label_cols = df_encoded.select_dtypes(include=['object']).columns
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Train model
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the feature order used during training
feature_order = X.columns.tolist()


# Streamlit UI
st.title("🔁 Customer Churn Predictor")

st.markdown("This app predicts the probability that a customer will churn based on their account info.")

# Inputs
gender = st.selectbox("Gender", df['gender'].unique())
contract = st.selectbox("Contract Type", df['Contract'].unique())
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
payment_method = st.selectbox("Payment Method", df['PaymentMethod'].unique())

# Prepare input
input_dict = {
    'gender': gender,
    'Contract': contract,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'PaperlessBilling': 'Yes',
    'TotalCharges': monthly_charges * tenure
}

# Encode input
input_df = pd.DataFrame([input_dict])
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Encode input
input_df = pd.DataFrame([input_dict])
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# 🔧 Ensure correct order
input_df = input_df.reindex(columns=feature_order)


# Predict
churn_prob = model.predict_proba(input_df)[0][1]
st.metric("Churn Probability", f"{churn_prob:.2%}")
