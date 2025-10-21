# train.py

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Configuration ---
DATA_DIR = "data/ml-100k"
EMBEDDING_DIM = 32
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.005
MODEL_PATH = "movielens_model.pt"
MOVIE_VECTORS_PATH = "movie_vectors.npy"

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load dataset ---
def load_movielens():
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep='\t', names=cols)
    print(f"Loaded {len(df)} ratings from MovieLens 100K.")
    return df

# --- Dataset ---
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values - 1, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values - 1, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# --- Matrix Factorization Model ---
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.uniform_(self.user_embedding.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_embedding.weight, -0.05, 0.05)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

# --- Training ---
def train_model(df):
    num_users = df['user_id'].max()
    num_items = df['item_id'].max()
    dataset = MovieLensDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MatrixFactorization(num_users, num_items, EMBEDDING_DIM).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting model training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for users, items, ratings in dataloader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print("Training complete.")
    return model, num_users, num_items

# --- Save movie vectors ---
def save_movie_vectors(model):
    movie_vectors = model.item_embedding.weight.data.cpu().numpy()
    # Normalize for cosine similarity (dot product)
    norms = np.linalg.norm(movie_vectors, axis=1, keepdims=True)
    movie_vectors = movie_vectors / (norms + 1e-8)
    
    np.save(MOVIE_VECTORS_PATH, movie_vectors)
    print(f"Saved {movie_vectors.shape[0]} normalized movie vectors to {MOVIE_VECTORS_PATH}")
    print(f"Shape: {movie_vectors.shape}")

# --- Main ---
if __name__ == "__main__":
    df = load_movielens()
    model, num_users, num_items = train_model(df)
    
    # Save full model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': EMBEDDING_DIM
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save movie vectors separately for inference
    save_movie_vectors(model)