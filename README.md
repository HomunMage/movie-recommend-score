# 🎬 End-to-End MLOps Pipeline: MovieLens Recommender with MLflow + pgvector + Grafana

## Overview
This project demonstrates a **complete MLOps pipeline** for a recommendation system built on the **MovieLens 100k** dataset.  
It integrates model training, experiment tracking, embedding-based retrieval, and system monitoring — all within a fully containerized environment using **Docker Compose**.

The goal is to showcase how modern MLOps techniques can operationalize a recommender system that leverages **vector similarity search** for user–movie embeddings.

---

## 🧠 Project Concept
Traditional recommenders often stop at training a model.  
This project goes further — implementing an **end-to-end operational flow**:

1. Train user and movie embeddings using MovieLens 100k.  
2. Store embeddings in **PostgreSQL with the pgvector extension** for similarity search.  
3. Register models and track experiments using **MLflow**.  
4. Serve recommendations via a **FastAPI** inference service.  
5. Monitor performance with **Prometheus** and visualize it in **Grafana**.

This setup mirrors what a production ML pipeline looks like — only scaled down for local experimentation.

---

## 🧩 System Architecture
```

```
      ┌─────────────────────┐
      │     MLflow UI       │
      │ (tracking server)   │
      └──────────┬──────────┘
                 │
                 ▼
```

┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Training Job │ → │ PostgreSQL + │ ← │ FastAPI API  │
│  (MovieLens  │   │  pgvector DB │   │ (Recommender)│
│   Embedding) │   └──────────────┘   └──────────────┘
└──────────────┘          │
▼
┌─────────────────────┐
│ Prometheus +        │
│ Grafana Monitoring  │
└─────────────────────┘

````

---

## 🧩 Key Components

### **MLflow**
- Tracks experiments, hyperparameters, and evaluation metrics.  
- Logs trained models and embedding artifacts.  
- Maintains a model registry for reproducible deployment and version comparison.

### **PostgreSQL + pgvector**
- Stores user and movie embeddings as vectors.  
- Enables fast similarity queries such as:
  ```sql
  SELECT movie_id FROM movie_embeddings
  ORDER BY embedding <-> :user_vector LIMIT 10;
````

* Serves as a lightweight **vector database** integrated with the recommender system.

### **FastAPI Inference Service**

* Loads the latest trained model from MLflow registry.
* Generates user embeddings on request and retrieves top recommendations via pgvector queries.
* Exposes Prometheus metrics for request count and latency monitoring.

### **Prometheus & Grafana**

* Prometheus scrapes runtime metrics from the inference service.
* Grafana provides dashboards to visualize throughput, latency, and API health.
* Demonstrates production-style observability for deployed ML systems.

### **Training Component**

* Uses MovieLens 100k data to train a collaborative filtering or transformer-based embedding model.
* Logs metrics like RMSE, Recall@K, and NDCG to MLflow.
* Saves user and movie embedding matrices for storage in pgvector.

---

## 🚀 Core Workflow

1. **Data & Embedding Training**

   * Train user–item embedding model using MovieLens 100k.
   * Evaluate on a validation split.
   * Log metrics and model weights to MLflow.

2. **Model Registration**

   * Save the trained model and embedding artifacts in the MLflow registry.
   * Register model version (e.g., `recommender_v1`, `recommender_v2`) for reproducibility.

3. **Embedding Storage**

   * Store final user and movie embeddings in PostgreSQL with pgvector extension.
   * Enable semantic similarity queries directly in SQL.

4. **Serving & Recommendation**

   * The FastAPI service loads the latest model version.
   * When a user requests recommendations:

     * Generate or retrieve their embedding.
     * Query pgvector for top similar movies.
     * Return results as ranked recommendations.

5. **Monitoring & Visualization**

   * Prometheus continuously scrapes API metrics.
   * Grafana dashboard displays latency, request rate, and model inference time.

---

## ⚙️ Technology Stack

| Layer                          | Tools / Frameworks                  |
| ------------------------------ | ----------------------------------- |
| **Model Training**             | PyTorch / Hugging Face Transformers |
| **Experiment Tracking**        | MLflow                              |
| **Model Registry & Artifacts** | MLflow Model Registry               |
| **Database & Vector Search**   | PostgreSQL + pgvector               |
| **Inference API**              | FastAPI                             |
| **Monitoring & Dashboards**    | Prometheus + Grafana                |
| **Containerization**           | Docker Compose                      |
| **Dataset**                    | MovieLens 100k (user–item ratings)  |

---

## 🧩 Repository Structure

```
mlops-movielens/
 ├─ train/                # Training scripts (embeddings + MLflow tracking)
 ├─ inference/            # FastAPI app (inference + vector search)
 ├─ db/                   # SQL init for pgvector setup
 ├─ prometheus/           # Prometheus config
 ├─ grafana/              # Grafana dashboards
 ├─ docker-compose.yml    # All-in-one local orchestration
 └─ README.md
```

---

## 📊 What This Project Demonstrates

* **End-to-End MLOps Integration** — from dataset to model deployment and monitoring.
* **Vector Database Usage** — pgvector for embedding-based recommendation retrieval.
* **Model Lifecycle Management** — training, versioning, and registry via MLflow.
* **Production Observability** — complete Prometheus/Grafana integration.
* **Reproducible Infrastructure** — all components containerized and orchestrated.

---

## 💬 Author’s Note

This project represents a **realistic miniature of modern machine learning systems**:
a data-driven recommendation model trained with open data, tracked and deployed with MLOps best practices,
and fully observable through metrics and dashboards.

It is designed both as a **portfolio piece** and a **learning framework** for engineers bridging the gap between model development and production operations.

---