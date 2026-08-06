import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
# 3. APPLE WHITE DESIGN SYSTEM INJECTION (CSS)
# ==========================================
apple_light_css = """
<style>
/* Import San Francisco / Inter System Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif !important;
    background-color: #f5f5f7 !important;
    color: #1d1d1f !important;
}

/* Background & Padding */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1150px !important;
}

/* Apple Light Header Style */
.hero-title {
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    color: #1d1d1f !important;
    margin-bottom: 0.2rem !important;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    color: #86868b !important;
    font-weight: 400 !important;
    margin-bottom: 2rem !important;
}

/* Apple Light Glass Cards */
div[data-testid="stColumn"] {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stColumn"]:hover {
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06) !important;
}

/* Form Section Headers */
.section-header {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #0071e3 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 16px !important;
}

/* Apple Blue Primary Button */
div.stButton > button {
    background: #0071e3 !important;
    color: #ffffff !important;
    border-radius: 980px !important;
    padding: 14px 28px !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 14px 0 rgba(0, 113, 227, 0.3) !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.01) !important;
    box-shadow: 0 6px 20px 0 rgba(0, 113, 227, 0.4) !important;
}

/* Label & Subtext styling */
label {
    color: #1d1d1f !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

.hinglish-subtext {
    font-size: 0.8rem !important;
    color: #6e6e73 !important;
    margin-top: -8px !important;
    margin-bottom: 14px !important;
    font-style: italic;
}

/* Output Result Cards (Light Theme) */
.result-card {
    border-radius: 20px;
    padding: 28px;
    margin-top: 24px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}

.result-card.low {
    background: #f2faf1;
    border-color: #34c759;
}

.result-card.medium {
    background: #fffdf0;
    border-color: #ffcc00;
}

.result-card.high {
    background: #fff2f1;
    border-color: #ff3b30;
}

.result-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6e6e73;
    margin-bottom: 4px;
    font-weight: 600;
}

.result-value {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
    color: #1d1d1f;
}

.status-badge {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 980px;
    display: inline-block;
}

.status-badge.low { background: #34c759; color: #ffffff; }
.status-badge.medium { background: #ffcc00; color: #1d1d1f; }
.status-badge.high { background: #ff3b30; color: #ffffff; }
</style>
"""
st.markdown(apple_light_css, unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model_pipeline.joblib")

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

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
# 6. INPUT INTERFACE WITH HINGLISH HELPER TEXTS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Account Profile</div>', unsafe_allow_html=True)
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown('<div class="hinglish-subtext">(Customer Male hai ya Female)</div>', unsafe_allow_html=True)
    
    SeniorCitizen = st.selectbox("Senior Citizen Status", [0, 1])
    st.markdown('<div class="hinglish-subtext">(Kya customer senior citizen hai? 1 = Haan, 0 = Nahi)</div>', unsafe_allow_html=True)
    
    Partner = st.selectbox("Partner", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Kya customer married ya partner ke sath hai?)</div>', unsafe_allow_html=True)
    
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Kya customer par koi family dependent hai?)</div>', unsafe_allow_html=True)
    
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    st.markdown('<div class="hinglish-subtext">(Customer kitne mahine se hamari service use kar raha hai)</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Subscriptions</div>', unsafe_allow_html=True)
    
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Kya Calling Service active hai?)</div>', unsafe_allow_html=True)
    
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Kya ek se zyada phone lines chal rahi hain?)</div>', unsafe_allow_html=True)
    
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown('<div class="hinglish-subtext">(Kon sa internet Connection hai - DSL ya Fiber?)</div>', unsafe_allow_html=True)
    
    OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Extra security & Antivirus protection pack hai?)</div>', unsafe_allow_html=True)
    
    OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Data cloud backup service li hai?)</div>', unsafe_allow_html=True)
    
    DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Hardware ya Device Insurance liya hai?)</div>', unsafe_allow_html=True)
    
    TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(VIP/Fast Customer Care Help Option active hai?)</div>', unsafe_allow_html=True)
    
    StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(TV Channels streaming service subscription hai?)</div>', unsafe_allow_html=True)
    
    StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Movies streaming subscription active hai?)</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
    
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown('<div class="hinglish-subtext">(Plan kaisa hai - Har mahine ka ya 1-2 saal ka contract?)</div>', unsafe_allow_html=True)
    
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Bill Online email/app par aata hai ya physical paper?)</div>', unsafe_allow_html=True)
    
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    st.markdown('<div class="hinglish-subtext">(Customer payment kis tareeqe se karta hai)</div>', unsafe_allow_html=True)
    
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    st.markdown('<div class="hinglish-subtext">(Customer har mahine kitna bill pay karta hai)</div>', unsafe_allow_html=True)
    
    TotalCharges = st.text_input("Total Charges ($)", value="840.0")
    st.markdown('<div class="hinglish-subtext">(Ab tak kul kitna payment diya hai customer ne)</div>', unsafe_allow_html=True)

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
        advice = "Customer company chhod sakta hai. Immediately offer discounts or assistance!"
    elif prob >= 0.40:
        risk_class = "medium"
        badge_text = "Moderate Churn Risk"
        advice = "Customer confuse hai. Consider offering a better long-term plan."
    else:
        risk_class = "low"
        badge_text = "Low Churn Risk"
        advice = "Customer happy hai. Connection safe and stable."

    # Apple Light Output Card Rendering
    result_html = f"""
    <div class="result-card {risk_class}">
        <div>
            <div class="result-title">Predicted Churn Risk</div>
            <div class="result-value">{prob_percentage}</div>
            <div style="color: #6e6e73; margin-top: 8px; font-size: 1rem; font-weight: 500;">{advice}</div>
        </div>
        <div>
            <span class="status-badge {risk_class}">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
