"""
Hugging Face Spaces app for GeoClustering API
Provides both a Gradio interface and FastAPI endpoints
"""

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from route_generator import GeoClustering

# ============================================================
# Load model (same as your load_model logic)
# ============================================================
clusterer = None

def load_model():
    global clusterer
    clusterer = GeoClustering()
    try:
        clusterer.load_model("production_model.pkl", "production_metadata.json")
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("📦 No existing model found, initializing...")
        default_coords = [
            [73.1812, 22.3072], [72.8311, 18.9388], [77.2090, 28.6139], [80.2707, 13.0827],
        ]
        clusterer = GeoClustering(cluster_method="kmeans")
        clusterer.cluster_coordinates(default_coords, auto=True, max_k=6)
        clusterer.save_model("production_model.pkl", "production_metadata.json")

load_model()

# ============================================================
# FastAPI setup
# ============================================================
app = FastAPI(title="GeoClustering API")

class ClusterRequest(BaseModel):
    coordinates: List[Tuple[float, float]]
    method: str = "kmeans"
    max_clusters: int = 6

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/cluster")
def api_cluster(req: ClusterRequest):
    """
    Enhanced clustering endpoint that returns structured data.
    """
    if not req.coordinates or len(req.coordinates) < 2:
        raise HTTPException(status_code=400, detail="At least 2 coordinates are required.")

    # Use the core clustering method
    labels, centers = clusterer.cluster_coordinates(
        req.coordinates,
        auto=True,
        max_k=req.max_clusters
    )

    # Process the results into a structured format
    unique_labels = set(labels)
    clusters_dict = {label: [] for label in unique_labels if label != -1}
    
    for i, label in enumerate(labels):
        if label != -1:
            # req.coordinates is List[Tuple[float, float]]
            lon, lat = req.coordinates[i]
            clusters_dict[label].append({"lon": lon, "lat": lat})

    # Format the final list of clusters
    output_clusters = []
    for label, center_coords in enumerate(centers):
        if label in clusters_dict:
            output_clusters.append({
                "cluster_id": int(label),
                "cluster_center": {"lon": center_coords[0], "lat": center_coords[1]},
                "coordinates": clusters_dict[label],
                "point_count": len(clusters_dict[label])
            })
            
    # Handle unassigned points (noise in DBSCAN)
    if -1 in unique_labels:
        noise_points = [{"lon": lon, "lat": lat} for i, (lon, lat) in enumerate(req.coordinates) if labels[i] == -1]
        output_clusters.append({
            "cluster_id": -1,
            "cluster_center": None,
            "coordinates": noise_points,
            "point_count": len(noise_points)
        })

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "total_points": len(req.coordinates),
        "total_clusters": len(centers),
        "clusters": output_clusters,
        "model_info": clusterer.model_metadata
    }

# ============================================================
# Gradio setup (your existing demo stays)
# ============================================================
with gr.Blocks(title="🗺️ GeoClustering Tool") as demo:
    gr.Markdown("# GeoClustering (UI)")
    # keep your existing Gradio tabs / components here
    # (I didn’t rewrite the long UI, you can reuse your existing code)

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
