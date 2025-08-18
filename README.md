# GeoClustering API

An ML-powered API for clustering geographical coordinates based on proximity within a city area. This project uses machine learning algorithms to group locations into clusters based on their geographical proximity, making it useful for logistics, urban planning, and location-based services.

## Features

- **Geospatial Clustering**: Uses KMeans, DBSCAN, and distance-based algorithms to cluster coordinates
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Automatic Retraining**: Model can be retrained with new data via API
- **Persistent Storage**: Trained models are saved and loaded automatically
- **Batch Processing**: Support for both individual and batch coordinate predictions
- **Health Monitoring**: Built-in health check and model information endpoints

## Use Cases

- Logistics and delivery route optimization
- Urban planning and zoning analysis
- Location-based marketing segmentation
- Ride-sharing and taxi services
- Emergency response planning

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Predict clusters for given coordinates
- `POST /predict-batch` - Batch prediction for coordinate arrays
- `POST /retrain` - Retrain the model with new data
- `GET /model-info` - Get information about the current model
- `GET /metrics` - Get basic API metrics

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd geoclustering-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python -m services.geoclustering
   ```

4. Access the API at `http://localhost:8000`

## Deployment

This API is configured for deployment on Vercel. The entry point is defined in `api/index.py`.

To deploy on Vercel:
1. Push your code to a GitHub repository
2. Connect the repository to Vercel
3. Vercel will automatically detect and deploy the FastAPI application

## API Usage Examples

### Predict Clusters

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": [
      {"lon": 73.1812, "lat": 22.3072},
      {"lon": 72.85, "lat": 21.17}
    ]
  }'
```

### Retrain Model

```bash
curl -X POST "http://localhost:8000/retrain" \
  -H "Content-Type: application/json" \
  -d '{
    "training_coordinates": [
      [73.1812, 22.3072],
      [72.85, 21.17],
      [73.18, 22.30]
    ],
    "max_k": 10
  }'
```

## Clustering Algorithms

The API supports three clustering methods:

1. **KMeans**: Partitions coordinates into K distinct clusters based on distance
2. **DBSCAN**: Groups coordinates based on density and distance thresholds
3. **Distance-based**: Creates clusters based on maximum distance between points

## Data Format

The API accepts coordinates in two formats:

1. **Object format**: `{"lon": longitude, "lat": latitude}`
2. **Array format**: `[longitude, latitude]`

## Project Structure

```
├── api/                 # Vercel entry point
├── services/            # FastAPI service implementation
├── route_generator.py   # Core clustering logic
├── data/                # Data files
├── clusters.json        # Sample clustering results
├── source.json          # Sample source coordinates
├── destination.json     # Sample destination coordinates
├── production_model.pkl # Trained model (auto-generated)
├── production_metadata.json # Model metadata (auto-generated)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Details

### Clustering Implementation

- **Haversine Distance**: Uses the Haversine formula to calculate accurate distances between coordinates on Earth
- **Automatic K Selection**: For KMeans, automatically determines optimal number of clusters using silhouette scoring
- **Model Persistence**: Trained models are saved as pickle files with JSON metadata

### Dependencies

- **FastAPI**: High-performance web framework
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing
- **Pydantic**: Data validation and serialization

## Development

To run in development mode with auto-reload:

```bash
uvicorn services.geoclustering:app --reload
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI and Scikit-learn
- Uses Haversine formula for accurate geospatial distance calculation
- Designed for deployment on Vercel serverless platform