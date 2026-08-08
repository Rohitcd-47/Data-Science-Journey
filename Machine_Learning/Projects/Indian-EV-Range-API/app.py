from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd

app = FastAPI(
    title="Indian EV Real Range & Thermal Risk API",
    description="Production REST API with Strict Data Validation Boundary Checks.",
    version="1.1.0"
)

# Load Pipelines
try:
    range_pipeline = joblib.load('ev_range_model_pipeline.joblib')
    risk_pipeline = joblib.load('ev_thermal_risk_pipeline.joblib')
except Exception as e:
    print(f"❌ Error loading serialized models: {e}")

# Strict Input Validation Schema (Fixing Edge Cases)
class EVTripRequest(BaseModel):
    Battery_Capacity_kWh: float = Field(..., gt=1.0, le=200.0, description="Battery capacity in kWh (1 to 200)")
    Claimed_ARAI_Range_km: float = Field(..., gt=10.0, le=1000.0, description="ARAI range claimed by OEM (10 to 1000 km)")
    Battery_Chemistry: Literal["LFP", "NMC"] = Field(..., description="Battery Chemistry must be strictly 'LFP' or 'NMC'")
    Initial_Battery_Percentage: float = Field(..., ge=0.0, le=100.0, description="Current SoC percentage (0 to 100)")
    Outside_Temperature_Celsius: float = Field(..., ge=-10.0, le=60.0, description="Ambient temperature (-10°C to 60°C)")
    Air_Conditioning_Status: int = Field(..., ge=0, le=1, description="1 for ON, 0 for OFF")
    Average_Speed_kmh: float = Field(..., gt=0.0, le=160.0, description="Average trip speed (1 to 160 km/h)")
    Battery_Health_Percentage: float = Field(..., ge=50.0, le=100.0, description="Battery State of Health SoH % (50 to 100)")

@app.get("/")
def home():
    return {"status": "Online", "service": "Indian EV Predictive API", "version": "1.1.0"}

@app.post("/predict")
def predict_ev_performance(data: EVTripRequest):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        predicted_range = float(range_pipeline.predict(input_data)[0])
        predicted_risk = str(risk_pipeline.predict(input_data)[0])

        return {
            "status": "success",
            "predictions": {
                "predicted_real_range_km": round(predicted_range, 1),
                "battery_thermal_risk": predicted_risk
            },
            "input_summary": data.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Runtime Error: {str(e)}")