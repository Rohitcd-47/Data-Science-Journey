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
# 3. FULL APPLE LIGHT THEME CSS OVERRIDE
# ==========================================
apple_light_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Force entire app canvas to Apple Light Gray */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F5F5F7 !important;
    color: #1D1D1F !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif !important;
}

/* Container Padding */
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1150px !important;
}

/* Main Hero Typography Fix */
.hero-title {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1D1D1F !important;
    margin-bottom: 0.3rem !important;
}

.hero-subtitle {
    font-size: 1.1rem !important;
    color: #6E6E73 !important;
    font-weight: 400 !important;
    margin-bottom: 2rem !important;
}

/* Glassmorphism Column Cards */
div[data-testid="stColumn"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 18px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
}

/* Section Subheaders */
.section-header {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0071E3 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 18px !important;
}

/* Force Light Styling on Streamlit Widgets (Selectbox, Inputs) */
div[data-baseweb="select"] > div, 
div[data-baseweb="base-input"] {
    background-color: #F5F5F7 !important;
    border: 1px solid #D2D2D7 !important;
    border-radius: 10px !important;
    color: #1D1D1F !important;
}

div[data-baseweb="select"] span, 
input {
    color: #1D1D1F !important;
    font-weight: 500 !important;
}

/* Dropdown Menu Popup Styling */
div[data-baseweb="popover"] div {
    background-color: #FFFFFF !important;
    color: #1D1D1F !important;
}

/* Labels and Hinglish Subtitles */
label {
    color: #1D1D1F !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

.hinglish-subtext {
    font-size: 0.78rem !important;
    color: #6E6E73 !important;
    margin-top: -6px !important;
    margin-bottom: 14px !important;
    font-style: italic;
}

/* Primary Action Button */
div.stButton > button {
    background: #0071E3 !important;
    color: #FFFFFF !important;
    border-radius: 980px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(0, 113, 227, 0.3) !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:hover {
    background: #0077ED !important;
    transform: translateY(-1px) !important;
}

/* Output Cards */
.result-card {
    border-radius: 18px;
    padding: 24px 30px;
    margin-top: 20px;
    border: 1px solid #E5E5EA;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
}

.result-card.low { background: #EAF8ED; border-color: #34C759; }
.result-card.medium { background: #FFF9E6; border-color: #FFCC00; }
.result-card.high { background: #FDECEB; border-color: #FF3B30; }

.result-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6E6E73; font-weight: 600; }
.result-value { font-size: 3rem; font-weight: 700; color: #1D1D1F; line-height: 1; margin-top: 4px; }
.status-badge { font-size: 1rem; font-weight: 600; padding: 8px 18px; border-radius: 980px; }
.status-badge.low { background: #34C759; color: #FFFFFF; }
.status-badge.medium { background: #FFCC00; color: #1D1D1F; }
.status-badge.high { background: #FF3B30; color: #FFFFFF; }
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
# 6. INPUT INTERFACE
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
    st.markdown('<div class="hinglish-subtext">(Customer kitne mahine se service use kar raha hai)</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Subscriptions</div>', unsafe_allow_html=True)
    
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Kya Calling Service active hai?)</div>', unsafe_allow_html=True)
    
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Kya ek se zyada phone lines chal rahi hain?)</div>', unsafe_allow_html=True)
    
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown('<div class="hinglish-subtext">(Kaun sa internet connection hai - DSL ya Fiber?)</div>', unsafe_allow_html=True)
    
    OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Extra security & Antivirus protection pack hai?)</div>', unsafe_allow_html=True)
    
    OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Data cloud backup service li hai?)</div>', unsafe_allow_html=True)
    
    DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Hardware ya Device Insurance liya hai?)</div>', unsafe_allow_html=True)
    
    TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(VIP/Fast Customer Care Help option active hai?)</div>', unsafe_allow_html=True)
    
    StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(TV channels streaming subscription hai?)</div>', unsafe_allow_html=True)
    
    StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Movies streaming subscription active hai?)</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
    
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown('<div class="hinglish-subtext">(Plan kaisa hai - Monthly ya 1-2 saal ka contract?)</div>', unsafe_allow_html=True)
    
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Bill online email/app par aata hai ya physical paper?)</div>', unsafe_allow_html=True)
    
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

    prob = pipeline.predict_proba(input_data)[0, 1]
    prob_percentage = f"{prob * 100:.1f}%"
    
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

    result_html = f"""
    <div class="result-card {risk_class}">
        <div>
            <div class="result-title">Predicted Churn Risk</div>
            <div class="result-value">{prob_percentage}</div>
            <div style="color: #6E6E73; margin-top: 8px; font-size: 0.95rem; font-weight: 500;">{advice}</div>
        </div>
        <div>
            <span class="status-badge {risk_class}">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
