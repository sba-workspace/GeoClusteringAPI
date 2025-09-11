#!/bin/bash

# Setup script for Hugging Face Spaces deployment

echo "🚀 Setting up GeoClustering for Hugging Face Spaces..."

# Create necessary directories
mkdir -p models
mkdir -p temp

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements_hf.txt

# Initialize model if needed
echo "🤖 Initializing model..."
python -c "
from route_generator import GeoClustering
import os

try:
    clusterer = GeoClustering()
    if not os.path.exists('production_model.pkl'):
        print('Creating initial model...')
        default_coords = [
            [73.1812, 22.3072], [73.1905, 22.3095], [73.1745, 22.2995],
            [72.8311, 18.9388], [72.8406, 18.9480], [72.8256, 18.9350],
            [77.2090, 28.6139], [77.2167, 28.6195], [77.2045, 28.6089],
            [80.2707, 13.0827], [80.2784, 13.0889], [80.2645, 13.0765],
        ]
        clusterer = GeoClustering(cluster_method='kmeans')
        clusterer.cluster_coordinates(default_coords, auto=True, max_k=6)
        clusterer.save_model('production_model.pkl', 'production_metadata.json')
        print('✅ Model initialized successfully')
    else:
        print('✅ Model already exists')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "✅ Setup complete! Ready for Hugging Face Spaces deployment."
