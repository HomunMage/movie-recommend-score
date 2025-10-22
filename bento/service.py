# bento/service.py

import bentoml
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel

# --- Request/Response Models ---
class UserRatings(BaseModel):
    ratings: dict[int, float]  # {movie_id (1-indexed): rating}

class UserVectorResponse(BaseModel):
    vector: list[float]
    dimension: int

# --- Model Definition ---
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

# --- Service Definition ---
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class MovieLensEncoder:
    def __init__(self):
        # Load model checkpoint
        MODEL_PATH = "movielens_model.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        self.embedding_dim = checkpoint['embedding_dim']
        
        # Initialize and load model
        self.model = MatrixFactorization(
            checkpoint['num_users'], 
            checkpoint['num_items'], 
            self.embedding_dim
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.device = device
        print(f"Model loaded on {device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    @bentoml.api
    def encode_user(self, user_ratings: UserRatings) -> UserVectorResponse:
        """
        Encode a user into the vector space based on their ratings.
        
        Args:
            user_ratings: UserRatings object with ratings dict {movie_id: rating}
        
        Returns:
            UserVectorResponse with the normalized user vector
        """
        ratings_dict = user_ratings.ratings
        
        if not ratings_dict:
            raise ValueError("User ratings cannot be empty")
        
        # Get item embeddings
        item_vecs = self.model.item_embedding.weight.data
        
        # Convert to 0-indexed and create tensors
        items = torch.tensor(
            [m - 1 for m in ratings_dict.keys()], 
            dtype=torch.long
        ).to(self.device)
        
        ratings = torch.tensor(
            list(ratings_dict.values()), 
            dtype=torch.float32
        ).to(self.device)
        
        # Least squares solution: solve (V^T V) u = V^T r
        rated_vecs = item_vecs[items]
        VtV = rated_vecs.T @ rated_vecs + 1e-5 * torch.eye(self.embedding_dim).to(self.device)
        Vtr = rated_vecs.T @ ratings
        user_vector = torch.linalg.solve(VtV, Vtr)
        
        # Normalize for dot product similarity
        user_vector = user_vector / (user_vector.norm() + 1e-8)
        
        # Convert to list for JSON response
        vector_list = user_vector.cpu().detach().numpy().tolist()
        
        return UserVectorResponse(
            vector=vector_list,
            dimension=self.embedding_dim
        )
    
    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "embedding_dim": self.embedding_dim,
            "device": str(self.device)
        }