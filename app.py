"""
Hugging Face Spaces app for GeoClustering API
Provides a Gradio interface for the geospatial clustering model
"""

import gradio as gr
import json
import numpy as np
import pandas as pd
from route_generator import GeoClustering
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any
import folium
from folium import plugins
import tempfile
import os

# Initialize the clustering model
clusterer = None

def load_model():
    """Load or initialize the clustering model"""
    global clusterer
    try:
        clusterer = GeoClustering()
        # Try to load existing model
        try:
            clusterer.load_model('production_model.pkl', 'production_metadata.json')
            print("✅ Model loaded successfully from disk")
        except FileNotFoundError:
            print("📦 No existing model found. Initializing with default data...")
            # Initialize with sample coordinates from major Indian cities
            default_coords = [
                [73.1812, 22.3072], [73.1905, 22.3095], [73.1745, 22.2995],  # Vadodara cluster
                [72.8311, 18.9388], [72.8406, 18.9480], [72.8256, 18.9350],  # Mumbai cluster
                [77.2090, 28.6139], [77.2167, 28.6195], [77.2045, 28.6089],  # Delhi cluster
                [80.2707, 13.0827], [80.2784, 13.0889], [80.2645, 13.0765],  # Chennai cluster
            ]
            clusterer = GeoClustering(cluster_method='kmeans')
            clusterer.cluster_coordinates(default_coords, auto=True, max_k=6)
            clusterer.save_model('production_model.pkl', 'production_metadata.json')
            print("✅ Initialized new model with sample Indian city data")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

def parse_coordinates(coord_text: str) -> List[Tuple[float, float]]:
    """Parse coordinates from text input"""
    try:
        lines = coord_text.strip().split('\n')
        coordinates = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try different formats
            if ',' in line:
                # Format: "lon,lat" or "lat,lon"
                parts = [float(x.strip()) for x in line.split(',')]
                if len(parts) == 2:
                    # Assume lon,lat format
                    coordinates.append((parts[0], parts[1]))
            elif ' ' in line:
                # Format: "lon lat" or "lat lon"
                parts = [float(x.strip()) for x in line.split()]
                if len(parts) == 2:
                    coordinates.append((parts[0], parts[1]))
        
        return coordinates
    except Exception as e:
        raise ValueError(f"Error parsing coordinates: {e}")

def create_cluster_map(coordinates: List[Tuple[float, float]], 
                      cluster_labels: np.ndarray, 
                      cluster_centers: List[Tuple[float, float]]) -> str:
    """Create an interactive folium map"""
    
    if not coordinates:
        return None
    
    # Calculate map center
    avg_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
    avg_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
    
    # Create map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
    
    # Color palette for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']
    
    # Add points to map
    for i, (coord, label) in enumerate(zip(coordinates, cluster_labels)):
        color = colors[label % len(colors)]
        folium.CircleMarker(
            location=[coord[1], coord[0]],  # folium expects [lat, lon]
            radius=8,
            popup=f'Point {i+1}<br>Cluster: {label}<br>Coordinates: ({coord[0]:.4f}, {coord[1]:.4f})',
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            weight=2
        ).add_to(m)
    
    # Add cluster centers
    for i, center in enumerate(cluster_centers):
        folium.Marker(
            location=[center[1], center[0]],  # folium expects [lat, lon]
            popup=f'Cluster {i} Center<br>({center[0]:.4f}, {center[1]:.4f})',
            icon=folium.Icon(color='darkred', icon='star')
        ).add_to(m)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    m.save(temp_file.name)
    
    return temp_file.name

def cluster_coordinates_interface(coord_text: str, 
                                cluster_method: str, 
                                max_clusters: int,
                                eps_km: float,
                                min_samples: int,
                                max_distance_km: float) -> Tuple[str, str, str]:
    """Main interface function for clustering"""
    
    if not clusterer:
        return "❌ Model not loaded", "", ""
    
    try:
        # Parse coordinates
        coordinates = parse_coordinates(coord_text)
        
        if len(coordinates) < 2:
            return "❌ Need at least 2 coordinates", "", ""
        
        # Update clustering method
        clusterer.cluster_method = cluster_method.lower()
        
        # Perform clustering based on method
        if cluster_method == "KMeans":
            labels, centers = clusterer.cluster_coordinates(
                coordinates, 
                auto=True, 
                max_k=max_clusters
            )
        elif cluster_method == "DBSCAN":
            labels, centers = clusterer.cluster_coordinates(
                coordinates,
                eps_km=eps_km,
                min_samples=min_samples
            )
        else:  # Distance-based
            labels, centers = clusterer.cluster_coordinates(
                coordinates,
                max_distance_km=max_distance_km
            )
        
        # Create results summary
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = sum(1 for label in labels if label == -1)
        
        summary = f"""
## 📊 Clustering Results

**Algorithm Used:** {cluster_method}
**Total Points:** {len(coordinates)}
**Number of Clusters:** {n_clusters}
**Noise Points:** {noise_points}

### Cluster Details:
"""
        
        # Add cluster details
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            cluster_points = [coordinates[i] for i, label in enumerate(labels) if label == cluster_id]
            summary += f"\n**Cluster {cluster_id}:** {len(cluster_points)} points"
            if cluster_id < len(centers):
                center = centers[cluster_id]
                summary += f" | Center: ({center[0]:.4f}, {center[1]:.4f})"
        
        # Create map
        map_html = create_cluster_map(coordinates, labels, centers)
        
        # Create results table
        results_data = []
        for i, (coord, label) in enumerate(zip(coordinates, labels)):
            results_data.append({
                'Point': i+1,
                'Longitude': f"{coord[0]:.6f}",
                'Latitude': f"{coord[1]:.6f}", 
                'Cluster': label,
                'Status': 'Noise' if label == -1 else f'Cluster {label}'
            })
        
        df = pd.DataFrame(results_data)
        results_table = df.to_string(index=False)
        
        return summary, map_html, f"```\n{results_table}\n```"
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def predict_clusters_interface(coord_text: str) -> Tuple[str, str, str]:
    """Interface for predicting clusters using existing model"""
    
    if not clusterer or clusterer.model is None:
        return "❌ No trained model available. Please cluster some data first.", "", ""
    
    try:
        # Parse coordinates
        coordinates = parse_coordinates(coord_text)
        
        if len(coordinates) < 1:
            return "❌ Need at least 1 coordinate", "", ""
        
        # Predict clusters
        labels = clusterer.predict_cluster(coordinates)
        
        # Create results
        summary = f"""
## 🎯 Prediction Results

**Model Used:** {clusterer.model_metadata.get('cluster_method', 'Unknown')}
**Total Points:** {len(coordinates)}
**Predicted Using:** {len(clusterer.cluster_centers)} cluster model

### Predictions:
"""
        
        results_data = []
        for i, (coord, label) in enumerate(zip(coordinates, labels)):
            label_int = int(label)
            results_data.append({
                'Point': i+1,
                'Longitude': f"{coord[0]:.6f}",
                'Latitude': f"{coord[1]:.6f}",
                'Predicted Cluster': label_int
            })
            summary += f"\nPoint {i+1}: Cluster {label_int}"
        
        # Create map for predictions
        map_html = create_cluster_map(coordinates, labels, clusterer.cluster_centers)
        
        df = pd.DataFrame(results_data)
        results_table = df.to_string(index=False)
        
        return summary, map_html, f"```\n{results_table}\n```"
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# Load model on startup
load_model()

# Create Gradio interface
with gr.Blocks(title="🗺️ GeoClustering - Coordinate Clustering Tool", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🗺️ GeoClustering - Coordinate Clustering Tool
    
    **Cluster geographical coordinates using machine learning algorithms**
    
    This tool helps you group geographic coordinates into clusters based on proximity. Perfect for:
    - 📦 Logistics route optimization
    - 🏙️ Urban planning analysis  
    - 📍 Location-based marketing
    - 🚗 Ride-sharing optimization
    """)
    
    with gr.Tabs():
        # Training/Clustering Tab
        with gr.TabItem("🎯 Cluster Coordinates", id="cluster"):
            gr.Markdown("""
            ### Enter Coordinates to Cluster
            
            **Format:** One coordinate per line as `longitude,latitude`
            
            **Example:**
            ```
            73.1812,22.3072
            73.1905,22.3095
            72.8311,18.9388
            77.2090,28.6139
            ```
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    coord_input = gr.Textbox(
                        label="📍 Coordinates",
                        placeholder="73.1812,22.3072\n73.1905,22.3095\n72.8311,18.9388\n77.2090,28.6139",
                        lines=10,
                        value="73.1812,22.3072\n73.1905,22.3095\n73.1745,22.2995\n72.8311,18.9388\n72.8406,18.9480\n77.2090,28.6139\n77.2167,28.6195\n80.2707,13.0827"
                    )
                    
                    cluster_method = gr.Dropdown(
                        label="🔧 Clustering Algorithm",
                        choices=["KMeans", "DBSCAN", "Distance"],
                        value="KMeans"
                    )
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        max_clusters = gr.Slider(
                            label="Max Clusters (KMeans)",
                            minimum=2,
                            maximum=20,
                            value=8,
                            step=1
                        )
                        
                        eps_km = gr.Slider(
                            label="Epsilon Distance km (DBSCAN)",
                            minimum=0.1,
                            maximum=50.0,
                            value=5.0,
                            step=0.1
                        )
                        
                        min_samples = gr.Slider(
                            label="Min Samples (DBSCAN)",
                            minimum=2,
                            maximum=20,
                            value=3,
                            step=1
                        )
                        
                        max_distance_km = gr.Slider(
                            label="Max Distance km (Distance-based)",
                            minimum=0.1,
                            maximum=100.0,
                            value=10.0,
                            step=0.1
                        )
                    
                    cluster_btn = gr.Button("🎯 Cluster Coordinates", variant="primary")
                
                with gr.Column(scale=2):
                    cluster_summary = gr.Markdown(label="📊 Results Summary")
                    cluster_map = gr.HTML(label="🗺️ Cluster Map")
                    cluster_table = gr.Markdown(label="📋 Detailed Results")
            
            cluster_btn.click(
                fn=cluster_coordinates_interface,
                inputs=[coord_input, cluster_method, max_clusters, eps_km, min_samples, max_distance_km],
                outputs=[cluster_summary, cluster_map, cluster_table]
            )
        
        # Prediction Tab
        with gr.TabItem("🔮 Predict New Points", id="predict"):
            gr.Markdown("""
            ### Predict Clusters for New Coordinates
            
            Use the trained model to predict which cluster new coordinates belong to.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    predict_input = gr.Textbox(
                        label="📍 New Coordinates",
                        placeholder="73.1850,22.3050\n72.8350,18.9400",
                        lines=6
                    )
                    
                    predict_btn = gr.Button("🔮 Predict Clusters", variant="primary")
                
                with gr.Column(scale=2):
                    predict_summary = gr.Markdown(label="🎯 Prediction Results")
                    predict_map = gr.HTML(label="🗺️ Prediction Map")
                    predict_table = gr.Markdown(label="📋 Prediction Details")
            
            predict_btn.click(
                fn=predict_clusters_interface,
                inputs=[predict_input],
                outputs=[predict_summary, predict_map, predict_table]
            )
        
        # Info Tab
        with gr.TabItem("ℹ️ About", id="about"):
            gr.Markdown("""
            ## 🗺️ About GeoClustering
            
            This application provides geospatial clustering using machine learning algorithms:
            
            ### 🔧 Algorithms Available:
            
            1. **KMeans** 
               - Partitions coordinates into K distinct clusters
               - Automatically finds optimal number of clusters
               - Best for: Evenly distributed points
            
            2. **DBSCAN**
               - Density-based clustering 
               - Automatically determines cluster count
               - Handles noise and irregular shapes
               - Best for: Varying densities and noise
            
            3. **Distance-based**
               - Groups points within specified distance
               - Simple threshold-based approach
               - Best for: Simple proximity grouping
            
            ### 📐 Distance Calculation:
            - Uses **Haversine formula** for accurate Earth surface distances
            - Accounts for Earth's curvature
            - Results in kilometers
            
            ### 🚀 Use Cases:
            - **Logistics**: Optimize delivery routes and warehouse locations
            - **Urban Planning**: Analyze population density and zoning
            - **Marketing**: Segment customers by location
            - **Emergency Services**: Plan response unit placement
            - **Transportation**: Optimize public transit routes
            
            ### 🎯 Features:
            - Real-time interactive clustering
            - Visual map representation
            - Model persistence and reuse
            - Batch processing support
            - Comprehensive error handling
            
            ---
            **Built with:** FastAPI, Scikit-learn, Folium, Gradio
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
