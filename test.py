import requests
import json

# url = "http://127.0.0.1:8000" #"https://sba-btw-GeoClustering-API.hf.space/api/cluster"

# payload = {
#     "coordinates": [[73.1812, 22.3072], [72.8311, 18.9388], [77.2090, 28.6139]],
# }

# response = requests.post(url, json=payload)
# print(response.json())

url = "http://127.0.0.1:8000/api/cluster"

# The /api/cluster endpoint expects a JSON object with a "coordinates" key
payload = {
    "coordinates": [
        [73.1812, 22.3072], 
        [72.8311, 18.9388], 
        [77.2090, 28.6139],
        [73.1850, 22.3090] # Added another point for clustering
    ]
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
    print("Response body:", response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
