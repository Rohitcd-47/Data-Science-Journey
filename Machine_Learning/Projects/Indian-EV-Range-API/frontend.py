import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Indian EV Analytics — Predictive Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Apple-Style Dark Theme (Zero Broken CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Surface */
    html, body, .stApp {
        background-color: #0A0B0E !important;
        color: #F5F5F7 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Ensure Sidebar Toggle Button Remains Visible */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        color: #F5F5F7 !important;
    }
    
    /* Elevated Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #11131C !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Headings */
    .title-main {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-main {
        color: #86868B;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Glass Cards for Documentation */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(20px);
    }
    
    .glass-badge {
        color: #2997FF;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    .glass-header {
        color: #F5F5F7;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .glass-body {
        color: #A1A1A6;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .code-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #64D2FF;
        font-family: monospace;
        padding: 8px 12px;
        font-size: 0.82rem;
        margin-top: 10px;
        display: block;
    }

    /* Primary Action Button */
    .stButton > button {
        background-color: #0071E3 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 12px rgba(0, 113, 227, 0.3) !important;
    }
    
    .stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.002);
    }

    /* Diagnostic Output Display Panels */
    .metric-glass {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    .val-blue { color: #2997FF; font-size: 2.8rem; font-weight: 700; }
    .val-high { color: #FF453A; font-size: 2.8rem; font-weight: 700; }
    .val-med  { color: #FF9F0A; font-size: 2.8rem; font-weight: 700; }
    .val-low  { color: #30D158; font-size: 2.8rem; font-weight: 700; }
    
    /* Sidebar Links */
    .social-link {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        color: #F5F5F7 !important;
        text-decoration: none !important;
        padding: 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }
    .social-link:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #2997FF;
    }
</style>
""", unsafe_allow_html=True)

# Remote API Endpoint
API_URL = "https://indian-ev-range-api.onrender.com/predict"

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("<h3 style='color:#F5F5F7; margin-bottom: 0px; font-size: 1.15rem; font-weight:700;'>INDIAN EV ANALYTICS</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#86868B; font-size:0.78rem;'>Production Machine Learning System</p>", unsafe_allow_html=True)
st.sidebar.write("---")

page = st.sidebar.radio(
    "Navigation Menu",
    ["Predict", "Documentation & Information"],
    index=0
)

st.sidebar.write("---")
st.sidebar.markdown("<p style='color:#86868B; font-size:0.72rem; font-weight:600; letter-spacing: 0.04em;'>DEVELOPER PROFILES</p>", unsafe_allow_html=True)

# GitHub & LinkedIn External Links
st.sidebar.markdown("""
<a href="https://github.com/Rohitcd-47/Data-Science-Journey/tree/main/Machine_Learning/Projects/Indian-EV-Range-API" target="_blank" class="social-link">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="#F5F5F7"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
    GitHub Repository
</a>
<a href="https://www.linkedin.com/in/rohit-dharmadhikari-1bba09255/" target="_blank" class="social-link">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="#2997FF"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.46 10.9v8.37H9.25V10.9H6.46M7.86 6.72a1.47 1.47 0 1 0 0 2.94 1.47 1.47 0 0 0 0-2.94z"/></svg>
    LinkedIn Profile
</a>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: PREDICT (DEFAULT ACTIVE SCREEN)
# ==========================================
if page == "Predict":
    st.markdown('<div class="title-main">Real-Time Range & Thermal Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-main">Configure vehicle hardware specs and ambient trip parameters to run diagnostic models.</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<h4 style='color:#F5F5F7; font-size:1.05rem; font-weight:600; margin-bottom:15px;'>Vehicle Hardware Specs</h4>", unsafe_allow_html=True)
        battery_capacity = st.slider("Battery Capacity (kWh)", 3.0, 100.0, 40.0, 0.5)
        claimed_range = st.slider("Claimed ARAI Range (km)", 50.0, 800.0, 421.0, 1.0)
        battery_chemistry = st.selectbox("Battery Chemistry Type", ["LFP", "NMC"])
        battery_soh = st.slider("Battery State of Health / SoH (%)", 50.0, 100.0, 95.0, 1.0)

    with col_b:
        st.markdown("<h4 style='color:#F5F5F7; font-size:1.05rem; font-weight:600; margin-bottom:15px;'>Trip Environment</h4>", unsafe_allow_html=True)
        battery_soc = st.slider("Current Charge / SoC (%)", 5.0, 100.0, 80.0, 1.0)
        outside_temp = st.slider("Outside Temperature (°C)", -5.0, 50.0, 42.0, 1.0)
        ac_status = st.radio("Cabin Air Conditioning (AC)", ["OFF", "ON"], index=1, horizontal=True)
        avg_speed = st.slider("Average Driving Speed (km/h)", 10.0, 140.0, 65.0, 1.0)

    st.write("")
    predict_btn = st.button("RUN PREDICTION MODEL")
    
    if predict_btn:
        payload = {
            "Battery_Capacity_kWh": battery_capacity,
            "Claimed_ARAI_Range_km": claimed_range,
            "Battery_Chemistry": battery_chemistry,
            "Initial_Battery_Percentage": battery_soc,
            "Outside_Temperature_Celsius": outside_temp,
            "Air_Conditioning_Status": 1 if ac_status == "ON" else 0,
            "Average_Speed_kmh": avg_speed,
            "Battery_Health_Percentage": battery_soh
        }
        
        with st.spinner("Executing model on Render cloud API..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    preds = result["predictions"]
                    
                    st.write("---")
                    st.markdown("<h3 style='color:#F5F5F7; font-size:1.1rem; font-weight:600;'>Diagnostics Overview</h3>", unsafe_allow_html=True)
                    
                    res_a, res_b = st.columns(2)
                    
                    with res_a:
                        st.markdown(f"""
                        <div class="metric-glass">
                            <p style="color:#86868B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">Predicted Real Range</p>
                            <div class="val-blue">{preds['predicted_real_range_km']} <span style="font-size:1.2rem; color:#86868B;">km</span></div>
                            <p style="color:#6E6E73; font-size:0.82rem; margin-top:6px;">Factory Claim: {claimed_range} km</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    risk = preds['battery_thermal_risk']
                    risk_class = "val-high" if risk == "High" else ("val-med" if risk == "Medium" else "val-low")
                    
                    with res_b:
                        st.markdown(f"""
                        <div class="metric-glass">
                            <p style="color:#86868B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">BMS Thermal Risk Level</p>
                            <div class="{risk_class}">{risk}</div>
                            <p style="color:#6E6E73; font-size:0.82rem; margin-top:6px;">Safety Diagnostics</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")


# ==========================================
# PAGE 2: DETAILED DOCUMENTATION & MANUAL
# ==========================================
else:
    st.markdown('<div class="title-main">System User Manual & Parameter Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-main">Detailed breakdown of all 8 input parameters, battery chemistries, and physics formulas in simple English.</div>', unsafe_allow_html=True)
    
    doc_a, doc_b = st.columns(2)
    
    with doc_a:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-badge">PARAMETER 01 — ENERGY STORAGE</div>
            <div class="glass-header">1. Battery Capacity (kWh)</div>
            <div class="glass-body">
                <b>Full Form:</b> Kilowatt-Hour.<br>
                <b>Simple Meaning:</b> This measures total energy stored in the battery pack—just like petrol tank size.<br>
                <b>Market Examples:</b>
                <ul>
                    <li>Electric Scooters (Ather 450X, Ola S1 Pro): 3.5 kWh to 4.0 kWh</li>
                    <li>Electric Cars (Tata Nexon EV, Punch EV): 30.0 kWh to 45.0 kWh</li>
                </ul>
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PARAMETER 02 — CERTIFIED BENCHMARK</div>
            <div class="glass-header">2. Claimed ARAI Range (km)</div>
            <div class="glass-body">
                <b>Full Form:</b> Automotive Research Association of India.<br>
                <b>Simple Meaning:</b> Factory range certified under laboratory conditions (no AC, slow constant 30 km/h speed, 25°C room temp). In practical driving, real-world range drops by 20%–40%.
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PARAMETER 03 — CELL CHEMISTRY</div>
            <div class="glass-header">3. Battery Chemistry Types</div>
            <div class="glass-body">
                <b>LFP (Lithium Iron Phosphate):</b> Safe in Indian summer heat, highly durable. Used in Tata Nexon EV & MG Comet.<br><br>
                <b>NMC (Nickel Manganese Cobalt):</b> Lightweight with high power output, but sensitive to high heat. Used in Ather, Ola, and Mahindra XUV400.
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PARAMETER 04 — HEALTH METRIC</div>
            <div class="glass-header">4. Battery Health / SoH (%)</div>
            <div class="glass-body">
                <b>Full Form:</b> State of Health.<br>
                <b>Simple Meaning:</b> Indicates battery aging. Brand-new EVs start at 100% SoH, while an EV used for 4 years degrades to ~85% SoH, holding less energy.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with doc_b:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-badge">PARAMETER 05 — TRIP STATE</div>
            <div class="glass-header">5. Current Battery Charge / SoC (%)</div>
            <div class="glass-body">
                <b>Full Form:</b> State of Charge.<br>
                <b>Simple Meaning:</b> Current percentage remaining on your dashboard gauge before starting the trip.
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PARAMETER 06 — WEATHER</div>
            <div class="glass-header">6. Outside Temperature (°C)</div>
            <div class="glass-body">
                <b>Simple Meaning:</b> Ambient weather. Batteries function best at 25°C. Extreme summer heat (40°C+) forces battery cooling fans to draw extra power.
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PARAMETER 07 — CLIMATE CONTROL</div>
            <div class="glass-header">7. Cabin Air Conditioning (AC)</div>
            <div class="glass-body">
                <b>Simple Meaning:</b> Running the AC draws continuous power directly from the main battery, reducing trip range by 10%–15%.
            </div>
        </div>

        <div class="glass-card">
            <div class="glass-badge">PHYSICS ENGINE — AERODYNAMICS</div>
            <div class="glass-header">8. Speed Penalty & Risk Equations</div>
            <div class="glass-body">
                Driving above 60 km/h increases air resistance exponentially, draining battery faster on highways.
                <span class="code-pill">Speed Penalty = 1.0 + (Speed - 60) × 0.01  [If Speed > 60 km/h]</span>
                <span class="code-pill">Risk Score = (Temp × 0.5) + (Speed × 0.3) + (AC × 10) + (15 if NMC)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)