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
# 3. PONDER DARK OBSIDIAN UI INJECTION (CSS)
# ==========================================
ponder_dark_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Force App Background to Deep Obsidian Black */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #08080A !important;
    color: #F0F0F2 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Display", sans-serif !important;
}

/* Fix Top-Right Header (GitHub Icon, Share, Options) Visibility */
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stHeader"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Main Layout Margins */
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 1150px !important;
}

/* Hero Title Typography */
.hero-title {
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #FFFFFF !important;
    margin-bottom: 0.3rem !important;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    color: #8A8F9E !important;
    font-weight: 400 !important;
    margin-bottom: 2.5rem !important;
}

/* Sleek Obsidian Glass Cards */
div[data-testid="stColumn"] {
    background: #121316 !important;
    border: 1px solid #22242B !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
}

/* Section Subheaders */
.section-header {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #38BDF8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 18px !important;
}

/* Dark Input Containers (Selectbox, Inputs, Number Boxes) */
div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div,
input[type="text"], input[type="number"] {
    background-color: #1A1C23 !important;
    border: 1px solid #2E323D !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

div[data-baseweb="select"] span, input {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Dropdown Menu Popup Items */
ul[aria-expanded="true"], div[data-baseweb="popover"] div {
    background-color: #1A1C23 !important;
    color: #FFFFFF !important;
    border-color: #2E323D !important;
}

li[aria-selected="true"] {
    background-color: #2E323D !important;
}

/* Increment/Decrement Buttons */
button[title="Increase value"], button[title="Decrease value"] {
    background-color: #2A2D37 !important;
    color: #FFFFFF !important;
    border: none !important;
}

/* Labels and Hinglish Subtitles */
label, label p {
    color: #E2E4E9 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

.hinglish-subtext {
    font-size: 0.78rem !important;
    color: #8A8F9E !important;
    margin-top: -6px !important;
    margin-bottom: 16px !important;
    font-style: italic;
}

/* Pure White High-Contrast Pill Button (Ponder AI Style) */
div.stButton > button {
    background: #FFFFFF !important;
    color: #08080A !important;
    border-radius: 980px !important;
    padding: 14px 28px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.15) !important;
    width: 100% !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button:hover {
    background: #E2E4E9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(255, 255, 255, 0.25) !important;
}

/* Output Cards (Dark Glowing Cards) */
.result-card {
    border-radius: 18px;
    padding: 28px;
    margin-top: 24px;
    border: 1px solid #22242B;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

.result-card.low { background: rgba(52, 199, 89, 0.12); border-color: #34C759; }
.result-card.medium { background: rgba(255, 204, 0, 0.12); border-color: #FFCC00; }
.result-card.high { background: rgba(255, 59, 48, 0.12); border-color: #FF3B30; }

.result-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8A8F9E; font-weight: 600; }
.result-value { font-size: 3.2rem; font-weight: 700; color: #FFFFFF; line-height: 1; margin-top: 4px; }
.status-badge { font-size: 1rem; font-weight: 600; padding: 8px 20px; border-radius: 980px; }
.status-badge.low { background: #34C759; color: #FFFFFF; }
.status-badge.medium { background: #FFCC00; color: #08080A; }
.status-badge.high { background: #FF3B30; color: #FFFFFF; }
</style>
"""
st.markdown(ponder_dark_css, unsafe_allow_html=True)

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
# 6. INPUT INTERFACE WITH HINGLISH SUBTITLES
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
    st.markdown('<div class="hinglish-subtext">(Kya customer par family dependent hai?)</div>', unsafe_allow_html=True)
    
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    st.markdown('<div class="hinglish-subtext">(Customer kitne mahine se service use kar raha hai)</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Subscriptions</div>', unsafe_allow_html=True)
    
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Kya Calling Service active hai?)</div>', unsafe_allow_html=True)
    
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Kya ek se zyada phone lines hain?)</div>', unsafe_allow_html=True)
    
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown('<div class="hinglish-subtext">(Kaun sa internet hai - DSL ya Fiber?)</div>', unsafe_allow_html=True)
    
    OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Extra security & Antivirus pack active hai?)</div>', unsafe_allow_html=True)
    
    OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Data cloud backup service li hai?)</div>', unsafe_allow_html=True)
    
    DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Hardware / Device Insurance hai?)</div>', unsafe_allow_html=True)
    
    TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(VIP Customer Care Support active hai?)</div>', unsafe_allow_html=True)
    
    StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(TV channels streaming service hai?)</div>', unsafe_allow_html=True)
    
    StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    st.markdown('<div class="hinglish-subtext">(Movies streaming subscription active hai?)</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
    
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown('<div class="hinglish-subtext">(Plan kaisa hai - Monthly ya 1-2 saal ka contract?)</div>', unsafe_allow_html=True)
    
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('<div class="hinglish-subtext">(Bill online aata hai ya physical paper par?)</div>', unsafe_allow_html=True)
    
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    st.markdown('<div class="hinglish-subtext">(Customer payment kis tarike se karta hai)</div>', unsafe_allow_html=True)
    
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    st.markdown('<div class="hinglish-subtext">(Customer har mahine kitna bill deta hai)</div>', unsafe_allow_html=True)
    
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
            <div style="color: #8A8F9E; margin-top: 8px; font-size: 0.95rem; font-weight: 500;">{advice}</div>
        </div>
        <div>
            <span class="status-badge {risk_class}">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
