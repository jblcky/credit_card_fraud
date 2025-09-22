import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 0.3rem;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #FF3333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model_path = 'xgboost_fraud_detector_22-9-2025.joblib'
    if os.path.exists(model_path):
        return load(model_path)
    else:
        st.error("Model file not found. Please ensure the model is saved in the correct location.")
        return None

# Function to make prediction
def predict_fraud(features):
    model = load_model()
    if model is None:
        return None

    prediction_proba = model.predict_proba(features)[0]
    return prediction_proba[1]  # Probability of fraud

# Header
st.title("🔍 Credit Card Fraud Detection")
st.markdown("### Intelligent Transaction Analysis System")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Transaction Details")

    # Create input fields for all V1-V28 features and Amount
    input_features = {}

    # Create 4 rows of 7 columns each for V1-V28
    for i in range(0, 28, 7):
        cols = st.columns(7)
        for j in range(7):
            if i + j < 28:  # For V1-V28
                input_features[f'V{i+j+1}'] = cols[j].number_input(
                    f'V{i+j+1}',
                    value=0.0,
                    format='%f'
                )

    # Amount in a separate section
    st.markdown("### Transaction Amount")
    amount = st.number_input('Amount ($)', min_value=0.0, value=100.0, format='%f')
    input_features['Amount'] = amount

with col2:
    st.markdown("### Analysis Result")

    # Create a button to trigger prediction
    if st.button('Analyze Transaction', key='predict'):
        # Prepare features in the correct order
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        features = pd.DataFrame([input_features])

        # Get prediction
        fraud_probability = predict_fraud(features)

        if fraud_probability is not None:
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fraud_probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                },
                title = {'text': "Fraud Probability (%)"}
            ))

            st.plotly_chart(fig)

            # Display result message
            if fraud_probability > 0.8:
                st.error("⚠️ High risk of fraud detected!")
            elif fraud_probability > 0.5:
                st.warning("⚠️ Medium risk transaction")
            else:
                st.success("✅ Low risk transaction")

            # Additional details
            st.info(f"Confidence Score: {fraud_probability:.3f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #666; font-size: 14px;'>
            Advanced Fraud Detection System powered by XGBoost
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
