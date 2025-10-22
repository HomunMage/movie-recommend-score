# 🎬 Movie Recommender (GPU + MLOps Pipeline)

This project demonstrates an end-to-end **AI/ML workflow** using containerized services for training, tracking, and serving a movie recommendation model.

## 🧩 Architecture Overview

dataset → trainer (PyTorch, GPU)
→ MLflow (experiment tracking)
→ BentoML (model packaging & inference on GPU)
→ backend API (FastAPI)
→ pgvector (vector storage)

All services are orchestrated via **Docker Compose**.

## 🚀 Main Components

- **backend/** – FastAPI service providing `/recommend` endpoint  
- **trainer/** – PyTorch model trainer with MLflow tracking  
- **mlflow/** – Experiment and model tracking server  
- **bento/** – BentoML service for model inference  
- **db (pgvector)** – Stores movie embeddings for similarity search  

## 💡 Example Usage

```bash
$ curl -X POST "http://localhost:8000/recommend?limit=5" \
  -H "Content-Type: application/json" \
  -d '{
    "user_ratings": {
      "ratings": {
        "14": 5,
        "12": 5,
        "64": 5,
        "345": 5
      }
    }
  }'
```

Response:
```
[{"movie_id":1025,"similarity":0.3218059241771698},{"movie_id":769,"similarity":
0.31998586654663086},{"movie_id":370,"similarity":0.3095213359309045},{"movie_id
":1265,"similarity":0.3009481430053711},{"movie_id":938,"similarity":0.300668427
36805777}]

```


## ⚙️ Key Features
* GPU-accelerated training with PyTorch
* MLflow tracking for metrics and artifacts
* pgvector-based embedding search
* BentoML inference service on GPU
* Fully containerized MLOps pipeline
