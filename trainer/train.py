import os
import requests
import zipfile
import io

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

from sqlalchemy import create_engine, text, inspect
from pgvector.sqlalchemy import Vector

# --- 1. Configuration & Setup ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATABASE_URL = os.getenv("DATABASE_URL")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("movielens_recommender")

# Model Hyperparameters
EMBEDDING_DIM = 32
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.005

engine = create_engine(DATABASE_URL)

# --- 2. Data Preparation ---
def prepare_database():
    """Enable pgvector extension and create tables."""
    with engine.connect() as conn:
        print("Enabling pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    inspector = inspect(engine)
    if not inspector.has_table("ratings"):
        print("Ratings table not found. Downloading and loading data...")
        # Download MovieLens 100K
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Read u.data file
        cols = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(z.open('ml-100k/u.data'), sep='\t', names=cols)
        
        # Load into postgres
        df.to_sql('ratings', engine, index=False, if_exists='replace')
        print(f"{len(df)} ratings loaded into the database.")
    else:
        print("Ratings table already exists.")

    if not inspector.has_table("movies"):
         # Create a table for movies and their embeddings
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS movies (
                    item_id INTEGER PRIMARY KEY,
                    embedding VECTOR({EMBEDDING_DIM})
                );
            """))
            conn.commit()
            print("Movies table created.")
    else:
        print("Movies table already exists.")


# --- 3. PyTorch Model and Dataset ---
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        # Initialize embeddings with small random values
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        dot_product = (user_vec * item_vec).sum(1)
        return dot_product

# --- 4. Training Loop ---
def train_model():
    with mlflow.start_run() as run:
        print("Starting MLflow Run:", run.info.run_id)
        
        # Log hyperparameters
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        
        df = pd.read_sql("SELECT * FROM ratings", engine)
        num_users = df['user_id'].max()
        num_items = df['item_id'].max()

        dataset = MovieLensDataset(df)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = MatrixFactorization(num_users, num_items, EMBEDDING_DIM)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("Starting model training...")
        for epoch in range(EPOCHS):
            total_loss = 0
            for users, items, ratings in dataloader:
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = loss_fn(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("mse_loss", avg_loss, step=epoch)

        print("Training finished.")
        
        # Log the trained model
        mlflow.pytorch.log_model(model, "recommender_model")
        
        # --- 5. Store Item Embeddings in pgvector ---
        print("Storing item embeddings in the database...")
        item_embeddings = model.item_embedding.weight.data.detach().cpu().numpy()
        
        # Create a DataFrame to bulk-insert
        item_data = []
        for item_id in range(1, num_items + 1):
            item_data.append({"item_id": item_id, "embedding": item_embeddings[item_id]})
        
        df_items = pd.DataFrame(item_data)

        with engine.connect() as conn:
            # Clear old embeddings
            conn.execute(text("DELETE FROM movies;"))
            # Bulk insert new embeddings
            df_items.to_sql('movies', conn, if_exists='append', index=False, dtype={'embedding': Vector(EMBEDDING_DIM)})
            conn.commit()
            
        print(f"{len(df_items)} item embeddings stored in pgvector.")

if __name__ == "__main__":
    prepare_database()
    train_model()