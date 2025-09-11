---
title: GeoClustering - Coordinate Clustering Tool
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11.9
---

# 🗺️ GeoClustering - Coordinate Clustering Tool

An interactive machine learning application for clustering geographical coordinates using KMeans, DBSCAN, and distance-based algorithms.

## 🚀 Features

- **Real-time Clustering**: Instantly cluster coordinates using advanced ML algorithms
- **Interactive Maps**: Visual representation with Folium maps
- **Multiple Algorithms**: KMeans, DBSCAN, and Distance-based clustering
- **Model Persistence**: Train once, predict multiple times
- **Batch Processing**: Handle multiple coordinates efficiently

## 🎯 Use Cases

- Logistics route optimization
- Urban planning analysis
- Location-based marketing segmentation
- Ride-sharing optimization
- Emergency response planning

## 🔧 Algorithms

1. **KMeans**: Optimal for evenly distributed points
2. **DBSCAN**: Handles varying densities and noise
3. **Distance-based**: Simple proximity grouping

## 📊 How to Use

1. **Cluster Tab**: Enter coordinates to create clusters
2. **Predict Tab**: Use trained model to predict new points
3. Adjust algorithm parameters for optimal results

## 🛠️ Technical Details

- **Distance Calculation**: Haversine formula for accurate Earth distances
- **Auto K-Selection**: Silhouette scoring for optimal cluster count
- **Interactive UI**: Gradio-powered interface
- **Real-time Processing**: Instant feedback and visualization

---

Built with FastAPI, Scikit-learn, Folium, and Gradio
