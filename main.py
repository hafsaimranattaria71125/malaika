import os

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from model import load_artifacts, predictAns
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest, PredictResponse, RiskFactor

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Predictor",
    description="Predict exam scores and get optimization strategies",
    version="1.0.0"
)
 
# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for Streamlit Cloud)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Load model on startup

@app.on_event("startup")
async def startup_event():
    """Load ML artifacts when app starts"""
    load_artifacts()
    print("✅ Model artifacts loaded successfully")
    

# ── Health Route ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return JSONResponse(
        status_code=200,
        content={"success": True, "message": "DiabetaCheck API v2 is running."}
    )

# ── Predict Endpoint ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(features: PredictRequest):
    result=predictAns(features.model_dump())
    return PredictResponse(**result )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 7860 (for Hugging Face)
    port = int(os.getenv("PORT", 7860))
    host = "0.0.0.0"  # Required for Docker/Hugging Face
    
    uvicorn.run(app, host=host, port=port)