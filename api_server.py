"""
FastAPI Inference Server for Power Procurement
================================================
Real-time REST API for power demand forecasting and
procurement strategy recommendations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.forecaster import PowerForecaster
from src.recommender import ProcurementRecommender


# Initialize FastAPI app
app = FastAPI(
    title="Power Procurement Strategy API",
    description="ML-powered API for data center power demand forecasting and procurement recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
forecaster: Optional[PowerForecaster] = None
recommender: Optional[ProcurementRecommender] = None


# Request/Response Models
class GPUUtilizationProfile(BaseModel):
    """48-hour GPU utilization profile."""
    values: List[float] = Field(..., min_items=48, max_items=48, 
                                 description="48 hourly GPU utilization values (0-1)")


class ClusterConfig(BaseModel):
    """Data center cluster configuration."""
    num_gpus: int = Field(..., ge=1, description="Number of GPUs")
    gpu_type: str = Field(default="H100", description="GPU type: A100, H100, MI300X")
    num_nodes: int = Field(default=1, ge=1, description="Number of nodes")
    num_cores: int = Field(default=128, ge=1, description="Number of CPU cores")
    workload_type: str = Field(default="LLM_Training", 
                               description="Workload type: LLM_Training or Inference")
    
    # Utilization profile
    gpu_util_profile: List[float] = Field(..., min_items=48, max_items=48,
                                          description="48 hourly GPU utilization values")
    
    # Optional power info (if known)
    avg_power_kw: Optional[float] = Field(None, description="Average power if known")
    burst_peak_kw: Optional[float] = Field(None, description="Peak power if known")
    
    # Site constraints
    grid_capacity_mw: float = Field(default=100.0, ge=1, description="Grid capacity in MW")
    renewable_target_pct: float = Field(default=80.0, ge=0, le=100)
    
    # Market prices
    ppa_price_usd_mwh: float = Field(default=55.0, ge=0)
    spot_price_avg_usd_mwh: float = Field(default=70.0, ge=0)
    battery_cost_usd_kwh: float = Field(default=180.0, ge=0)


class PowerForecastResponse(BaseModel):
    """Power demand forecast response."""
    power_trace_kw: List[float]
    avg_power_kw: float
    peak_power_kw: float
    min_power_kw: float


class ProcurementRecommendation(BaseModel):
    """Procurement strategy recommendation."""
    recommended_mix: Dict[str, float]
    mix_percentages: Dict[str, float]
    contract_lead_time_months: int
    forecasted_cost_usd_mwh: float


class FullPredictionResponse(BaseModel):
    """Combined forecast and recommendation response."""
    power_forecast: PowerForecastResponse
    procurement: ProcurementRecommendation
    cluster_summary: Dict


# Startup/Shutdown Events
@app.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global forecaster, recommender
    
    try:
        print("🔄 Loading models...")
        
        forecaster = PowerForecaster("config.yaml")
        forecaster.load("models/forecaster")
        print("✅ Forecaster loaded")
        
        recommender = ProcurementRecommender("config.yaml")
        recommender.load("models/recommender")
        print("✅ Recommender loaded")
        
        print("🚀 API ready!")
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("   API will run in demo mode")


# Health Check
@app.get("/health")
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "models": {
            "forecaster": forecaster is not None,
            "recommender": recommender is not None
        }
    }


# Endpoints
@app.post("/forecast", response_model=PowerForecastResponse)
async def forecast_power(config: ClusterConfig):
    """
    Forecast 48-hour power demand from GPU utilization profile.
    
    Returns predicted power consumption trace.
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster model not loaded")
    
    try:
        # Prepare input
        gpu_util = np.array(config.gpu_util_profile).reshape(1, -1)
        
        # Prepare metadata
        meta = np.array([[
            config.num_gpus,
            config.num_nodes,
            config.num_cores,
            config.avg_power_kw or 0,
            config.burst_peak_kw or 0
        ]], dtype=np.float32)
        
        # Predict
        power_trace = forecaster.predict(gpu_util, meta)[0]
        
        return PowerForecastResponse(
            power_trace_kw=power_trace.tolist(),
            avg_power_kw=float(np.mean(power_trace)),
            peak_power_kw=float(np.max(power_trace)),
            min_power_kw=float(np.min(power_trace))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=ProcurementRecommendation)
async def recommend_procurement(config: ClusterConfig):
    """
    Get procurement strategy recommendation.
    
    Returns recommended PPA/Spot/Battery mix and cost forecast.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")
    
    try:
        # Build sample dict
        sample = {
            'num_gpus': config.num_gpus,
            'gpu_type': config.gpu_type,
            'num_nodes': config.num_nodes,
            'num_cores': config.num_cores,
            'workload_type': config.workload_type,
            'gpu_util_profile': config.gpu_util_profile,
            'power_trace_kw': config.gpu_util_profile,  # Will be overwritten if forecaster used
            'avg_power_kw': config.avg_power_kw or 10.0,
            'burst_peak_kw': config.burst_peak_kw or 15.0,
            'grid_capacity_mw': config.grid_capacity_mw,
            'renewable_target_pct': config.renewable_target_pct,
            'ppa_price_usd_mwh': config.ppa_price_usd_mwh,
            'spot_price_avg_usd_mwh': config.spot_price_avg_usd_mwh,
            'battery_cost_usd_kwh': config.battery_cost_usd_kwh,
            'batch_size': 32,
            'seq_length': 2048,
            'fp_precision': 'bf16',
            'rack_density_kw': 25.0
        }
        
        # Get recommendation
        result = recommender.predict_single(sample)
        
        return ProcurementRecommendation(
            recommended_mix=result['recommended_mix'],
            mix_percentages=result['mix_percentages'],
            contract_lead_time_months=result['contract_lead_time_months'],
            forecasted_cost_usd_mwh=result['forecasted_cost_usd_mwh']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=FullPredictionResponse)
async def full_prediction(config: ClusterConfig):
    """
    Get complete prediction: power forecast + procurement recommendation.
    
    This is the main endpoint that combines both models.
    """
    if forecaster is None or recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Step 1: Forecast power
        gpu_util = np.array(config.gpu_util_profile).reshape(1, -1)
        meta = np.array([[
            config.num_gpus,
            config.num_nodes,
            config.num_cores,
            config.avg_power_kw or 0,
            config.burst_peak_kw or 0
        ]], dtype=np.float32)
        
        power_trace = forecaster.predict(gpu_util, meta)[0]
        
        power_forecast = PowerForecastResponse(
            power_trace_kw=power_trace.tolist(),
            avg_power_kw=float(np.mean(power_trace)),
            peak_power_kw=float(np.max(power_trace)),
            min_power_kw=float(np.min(power_trace))
        )
        
        # Step 2: Get procurement recommendation
        sample = {
            'num_gpus': config.num_gpus,
            'gpu_type': config.gpu_type,
            'num_nodes': config.num_nodes,
            'num_cores': config.num_cores,
            'workload_type': config.workload_type,
            'gpu_util_profile': config.gpu_util_profile,
            'power_trace_kw': power_trace.tolist(),
            'avg_power_kw': power_forecast.avg_power_kw,
            'burst_peak_kw': power_forecast.peak_power_kw,
            'grid_capacity_mw': config.grid_capacity_mw,
            'renewable_target_pct': config.renewable_target_pct,
            'ppa_price_usd_mwh': config.ppa_price_usd_mwh,
            'spot_price_avg_usd_mwh': config.spot_price_avg_usd_mwh,
            'battery_cost_usd_kwh': config.battery_cost_usd_kwh,
            'batch_size': 32,
            'seq_length': 2048,
            'fp_precision': 'bf16',
            'rack_density_kw': 25.0
        }
        
        result = recommender.predict_single(sample)
        
        procurement = ProcurementRecommendation(
            recommended_mix=result['recommended_mix'],
            mix_percentages=result['mix_percentages'],
            contract_lead_time_months=result['contract_lead_time_months'],
            forecasted_cost_usd_mwh=result['forecasted_cost_usd_mwh']
        )
        
        # Build summary
        cluster_summary = {
            'gpu_count': config.num_gpus,
            'gpu_type': config.gpu_type,
            'workload': config.workload_type,
            'total_power_mw': round(power_forecast.avg_power_kw / 1000, 3),
            'renewable_target': config.renewable_target_pct
        }
        
        return FullPredictionResponse(
            power_forecast=power_forecast,
            procurement=procurement,
            cluster_summary=cluster_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample")
async def get_sample_request():
    """Get a sample request body for testing."""
    # Generate sample GPU utilization (training pattern)
    t = np.arange(48)
    util = 0.9 + 0.05 * np.random.randn(48)
    util = np.clip(util, 0.7, 0.98)
    
    return ClusterConfig(
        num_gpus=512,
        gpu_type="H100",
        num_nodes=8,
        num_cores=1024,
        workload_type="LLM_Training",
        gpu_util_profile=util.tolist(),
        grid_capacity_mw=50.0,
        renewable_target_pct=85.0,
        ppa_price_usd_mwh=52.0,
        spot_price_avg_usd_mwh=68.0,
        battery_cost_usd_kwh=175.0
    )


# Run with: uvicorn api_server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
