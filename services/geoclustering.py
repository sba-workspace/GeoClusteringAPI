from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import pickle
import numpy as np
from datetime import datetime
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path to allow importing route_generator
sys.path.append(str(Path(__file__).parent.parent))
from route_generator import GeoClustering

# Pydantic models for request/response validation
class Coordinate(BaseModel):
    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")

class PredictionRequest(BaseModel):
    coordinates: List[Coordinate] = Field(..., description="List of coordinates to cluster")

class TrainingRequest(BaseModel):
    training_coordinates: List[List[float]] = Field(..., description="Training coordinates as [lon, lat] pairs")
    max_k: Optional[int] = Field(15, description="Maximum number of clusters to test")

class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_center: Coordinate
    coordinates: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    success: bool
    timestamp: str
    total_points: int
    total_clusters: int
    clusters: List[ClusterInfo]
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    success: bool
    model_info: Dict[str, Any]
    cluster_centers: List[List[float]]
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="GeoClustering API",
    description="API for geographical coordinate clustering and prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clusterer instance
clusterer = None

def load_model_on_startup():
    """Load the trained model when the API starts, or initialize a new one if not found"""
    global clusterer
    try:
        # Try to load existing model first
        try:
            clusterer = GeoClustering()
            clusterer.load_model('production_model.pkl', 'production_metadata.json')
            print("Model loaded successfully from disk")
            return True
        except FileNotFoundError:
            print("No existing model found. Initializing with default model.")
        
        # Initialize with some default coordinates if no model exists
        default_coords = [
            [73.17, 22.30], [73.18, 22.30], [73.17, 22.31],  # Cluster 1
            [72.85, 21.17], [72.86, 21.17], [72.85, 21.18]   # Cluster 2
        ]
        clusterer = GeoClustering(cluster_method='kmeans')
        clusterer.cluster_coordinates(default_coords, auto=True, max_k=2)
        clusterer.save_model('production_model.pkl', 'production_metadata.json')
        print("Initialized new model with default data")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        clusterer = None
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_on_startup()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GeoClustering API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if clusterer is not None else "unhealthy",
        model_loaded=clusterer is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_clusters(request: PredictionRequest):
    """
    Predict clusters for given coordinates
    
    - **coordinates**: List of coordinate objects with lon/lat
    - Returns clustered coordinates grouped by cluster ID
    """
    try:
        if clusterer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check service health."
            )
        
        # Convert Pydantic models to list of tuples
        coordinates = [(coord.lon, coord.lat) for coord in request.coordinates]
        
        # Predict and format response
        result = clusterer.predict_and_format_for_backend(coordinates)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        # Convert to Pydantic response model
        clusters = []
        for cluster_data in result["clusters"]:
            cluster_info = ClusterInfo(
                cluster_id=cluster_data["cluster_id"],
                cluster_center=Coordinate(
                    lon=cluster_data["cluster_center"]["lon"],
                    lat=cluster_data["cluster_center"]["lat"]
                ),
                coordinates=cluster_data["coordinates"]
            )
            clusters.append(cluster_info)
        
        return PredictionResponse(
            success=True,
            timestamp=result["timestamp"],
            total_points=result["total_points"],
            total_clusters=result["total_clusters"],
            clusters=clusters,
            model_info=result["model_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/retrain", tags=["Model Management"])
async def retrain_model(request: TrainingRequest):
    """
    Retrain the model with new data
    
    - **training_coordinates**: List of [lon, lat] coordinate pairs
    - **max_k**: Maximum number of clusters to test (optional, default: 15)
    """
    try:
        training_coords = request.training_coordinates
        max_k = request.max_k
        
        # Validate coordinate format
        for coord in training_coords:
            if len(coord) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each training coordinate must be [lon, lat] pair"
                )
        
        # Retrain model
        global clusterer
        clusterer = GeoClustering(cluster_method='kmeans')
        cluster_labels, centers = clusterer.cluster_coordinates(
            training_coords,
            auto=True,
            max_k=max_k
        )
        
        # Save the new model
        clusterer.save_model('production_model.pkl', 'production_metadata.json')
        
        return {
            "success": True,
            "message": "Model retrained successfully",
            "n_clusters": len(centers),
            "cluster_centers": centers,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Management"])
async def get_model_info():
    """Get information about the current model"""
    try:
        if clusterer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        return ModelInfoResponse(
            success=True,
            model_info=clusterer.model_metadata,
            cluster_centers=clusterer.cluster_centers,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.post("/predict-batch", tags=["Prediction"])
async def predict_clusters_batch(coordinates: List[List[float]]):
    """
    Batch prediction endpoint for simple coordinate arrays
    
    - **coordinates**: List of [lon, lat] coordinate pairs
    - Simplified endpoint for direct coordinate arrays
    """
    try:
        if clusterer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Validate coordinate format
        for coord in coordinates:
            if len(coord) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each coordinate must be [lon, lat] pair"
                )
        
        # Convert to coordinate objects for internal processing
        coord_objects = [{"lon": coord[0], "lat": coord[1]} for coord in coordinates]
        
        # Predict using internal helper
        result = clusterer.predict_and_format_for_backend(coord_objects)

        # Transform to minimal format: list of clusters with coordinate arrays
        simplified_clusters = []
        for cluster in result["clusters"]:
            coord_pairs = [ [pt["lon"], pt["lat"]] for pt in cluster["coordinates"] ]
            simplified_clusters.append({
                "cluster_id": cluster["cluster_id"],
                "coordinates": coord_pairs
            })

        # Return minified JSON so arrays stay inline
        import json as _json
        from fastapi.responses import Response
        return Response(
            content=_json.dumps({"clusters": simplified_clusters}, separators=(',', ':')),
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Custom exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",  # Replace "main" with your filename
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

# Additional utility endpoints for monitoring
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get basic API metrics"""
    return {
        "uptime": datetime.now().isoformat(),
        "model_loaded": clusterer is not None,
        "api_version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "retrain": "/retrain",
            "model_info": "/model-info"
        }
    }