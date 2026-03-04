import streamlit as st
import numpy as np
import torch
from torch import nn
import pandas as pd
import pickle


# ------------------ Model Definition ------------------ #
class AnnModel(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.network(features)


# ------------------ Load Model ------------------ #
model = AnnModel()
model.load_state_dict(torch.load('ann_model_weights.pth'))
model.eval()


# ------------------ Load Encoders & Scaler ------------------ #
with open('label_encoder_geography.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# ------------------ Streamlit App ------------------ #
st.title('Customer Churn Prediction')


# User Inputs
geography = st.selectbox(
    'Geography',
    label_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 0, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

if is_active_member=='Yes':
    is_active_member = 1
else:
    is_active_member=0

# ------------------ Prepare Input Data ------------------ #
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-Hot Encode Geography
geo_encoded = label_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)


# Combine All Features
input_df = pd.concat([input_data, geo_encoded_df], axis=1)


# Scale Features
input_scaled = scaler.transform(input_df)


# Convert to Tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)


# ------------------ Prediction ------------------ #
with torch.no_grad():
    prediction = model(input_tensor)

result = prediction.item()


# ------------------ Output ------------------ #
if result > 0.35:
    st.write(result)
    st.write(f"Prediction: Likely to churn")
else:
    st.write(result)
    st.write(f"Prediction: Likely to stay)")