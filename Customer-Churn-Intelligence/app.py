import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. CUSTOM CLASS DEFINITION (Required for joblib)
# ==========================================
class RawDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='customerID'):
        self.id_col = id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.id_col in X.columns:
            X = X.drop(columns=[self.id_col])
        
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0.0)
        if 'tenure' in X.columns:
            X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in X.columns:
            X['MonthlyCharges'] = pd.to_numeric(X['MonthlyCharges'], errors='coerce').fillna(0.0)
            
        return X

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Telco Churn Intelligence", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 3. APPLE DESIGN SYSTEM INJECTION (CSS)
# ==========================================
apple_css = """
<style>
/* Import San Francisco / Inter System Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif !important;
    background-color: #000000 !important;
    color: #f5f5f7 !important;
}

/* Background & Padding */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* Apple Header Style */
.hero-title {
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    background: linear-gradient(180deg, #FFFFFF 0%, #A1A1A6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    color: #86868b !important;
    font-weight: 400 !important;
    margin-bottom: 2rem !important;
}

/* Glassmorphism Input Cards */
div[data-testid="stColumn"] {
    background: rgba(22, 22, 23, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    backdrop-filter: blur(20px) !important;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

div[data-testid="stColumn"]:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Form Section Headers */
.section-header {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #2997ff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 12px !important;
}

/* Apple Primary Button */
div.stButton > button {
    background: #0071e3 !important;
    color: #ffffff !important;
    border-radius: 980px !important;
    padding: 12px 28px !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 14px 0 rgba(0, 113, 227, 0.39) !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.01) !important;
    box-shadow: 0 6px 20px 0 rgba(0, 113, 227, 0.5) !important;
}

/* Input elements styling */
label {
    color: #a1a1a6 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Custom Output Result Card */
.result-card {
    border-radius: 20px;
    padding: 28px;
    margin-top: 20px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.result-card.low {
    background: linear-gradient(135deg, rgba(48, 209, 88, 0.12) 0%, rgba(22, 22, 23, 0.9) 100%);
    border-color: rgba(48, 209, 88, 0.3);
}

.result-card.medium {
    background: linear-gradient(135deg, rgba(255, 214, 10, 0.12) 0%, rgba(22, 22, 23, 0.9) 100%);
    border-color: rgba(255, 214, 10, 0.3);
}

.result-card.high {
    background: linear-gradient(135deg, rgba(255, 69, 58, 0.12) 0%, rgba(22, 22, 23, 0.9) 100%);
    border-color: rgba(255, 69, 58, 0.3);
}

.result-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #86868b;
    margin-bottom: 4px;
}

.result-value {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
}

.status-badge {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 980px;
    display: inline-block;
}

.status-badge.low { background: rgba(48, 209, 88, 0.2); color: #30d158; }
.status-badge.medium { background: rgba(255, 214, 10, 0.2); color: #ffd60a; }
.status-badge.high { background: rgba(255, 69, 58, 0.2); color: #ff453a; }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADING
# ==========================================
@st.cache_resource
def load_pipeline():
    return joblib.load("churn_model_pipeline.joblib")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()

# ==========================================
# 5. HERO HEADER
# ==========================================
st.markdown('<div class="hero-title">Customer Churn Intelligence.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Proactive risk analysis powered by predictive machine learning.</div>', unsafe_allow_html=True)

# ==========================================
# 6. INPUT INTERFACE (THREE COLUMNS)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Account Profile</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen Status", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

with col2:
    st.markdown('<div class="section-header">Subscriptions</div>', unsafe_allow_html=True)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

with col3:
    st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    TotalCharges = st.text_input("Total Charges ($)", value="840.0")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. INFERENCE TRIGGER & RESULT CARD
# ==========================================
if st.button("Analyze Risk Profile"):
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Probability Score
    prob = pipeline.predict_proba(input_data)[0, 1]
    prob_percentage = f"{prob * 100:.1f}%"
    
    # State Evaluation
    if prob >= 0.60:
        risk_class = "high"
        badge_text = "High Churn Risk"
        advice = "Immediate retention action required."
    elif prob >= 0.40:
        risk_class = "medium"
        badge_text = "Moderate Churn Risk"
        advice = "Consider offering contract incentives."
    else:
        risk_class = "low"
        badge_text = "Low Churn Risk"
        advice = "Customer relationship is stable."

    # Apple-style Output Card Rendering
    result_html = f"""
    <div class="result-card {risk_class}">
        <div>
            <div class="result-title">Predicted Churn Risk</div>
            <div class="result-value">{prob_percentage}</div>
            <div style="color: #a1a1a6; margin-top: 8px; font-size: 0.95rem;">{advice}</div>
        </div>
        <div>
            <span class="status-badge {risk_class}">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)