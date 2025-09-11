"""
Hugging Face Spaces app for GeoClustering API
Provides both a Gradio interface and FastAPI endpoints
"""

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Tuple
import numpy as np
import pandas as pd
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
    if not req.coordinates or len(req.coordinates) < 2:
        return {"error": "Need at least 2 coordinates"}
    
    labels, centers = clusterer.cluster_coordinates(
        req.coordinates,
        auto=True,
        max_k=req.max_clusters
    )
    
    return {
        "clusters": labels.tolist(),
        "centers": centers,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
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
