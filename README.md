# Edge-AI Disaster Response (Pro Version)

Turn-key, dockerised simulation of an edge AI pipeline for disaster response,
with **web UI**, **SQLite persistence**, and **NLP + CV + late fusion**.

## Services
- **nlp** (port 5001): text classification (request_help, infrastructure_damage, donation_offer, other)
- **cv** (port 5002): image tagging (fire, flood, damage, unknown) via OpenCV heuristics
- **api** (port 5000): late fusion + SQLite persistence + simple web UI

## Quick start
```bash
docker compose up --build
```
Open the **web UI** at: http://localhost:5000/

## Test via curl
```bash
# NLP
curl -s -X POST http://localhost:5001/predict -H "Content-Type: application/json"   -d '{"text":"Bridge collapsed, need help"}' | jq

# CV
curl -s -X POST http://localhost:5002/predict -F "image=@samples/images/damage1.jpg" | jq

# Aggregator
curl -s -X POST http://localhost:5000/analyse   -F "text=Large fire near refinery"   -F "image=@samples/images/fire1.jpg" | jq
```

## Upgrade ideas
- Replace NLP with DistilBERT (quantised) using ONNX Runtime.
- Replace CV heuristics with YOLOv8n (ONNX/OpenVINO), keep CPU-friendly.
- Add geocoding and map visualisation (Leaflet).
