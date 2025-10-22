# single/inference.py

import torch
import torch.nn as nn
import numpy as np

# --- Config ---
MODEL_PATH = "movielens_model.pt"
MOVIE_VECTORS_PATH = "movie_vectors.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load movie vectors ---
print("Loading movie vectors...")
movie_vectors = np.load(MOVIE_VECTORS_PATH)
movie_vectors_tensor = torch.from_numpy(movie_vectors).float().to(device)
print(f"Loaded {movie_vectors.shape[0]} movies, embedding dim: {movie_vectors.shape[1]}")

# --- Load model info ---
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
EMBEDDING_DIM = checkpoint['embedding_dim']

# --- Define model structure ---
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

# --- Load model ---
model = MatrixFactorization(
    checkpoint['num_users'], 
    checkpoint['num_items'], 
    EMBEDDING_DIM
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# --- User Encoder: encode user ratings into vector space ---
def user_enc(user_ratings):
    """
    Encode a user into the vector space based on their ratings.
    
    Args:
        user_ratings: dict {movie_id (1-indexed): rating}
    
    Returns:
        user_vector: normalized tensor in the same space as movies
    """
    item_vecs = model.item_embedding.weight.data
    
    # Get rated movies (convert to 0-indexed)
    items = torch.tensor([m - 1 for m in user_ratings.keys()], dtype=torch.long).to(device)
    ratings = torch.tensor(list(user_ratings.values()), dtype=torch.float32).to(device)
    
    # Least squares solution: solve (V^T V) u = V^T r
    rated_vecs = item_vecs[items]
    VtV = rated_vecs.T @ rated_vecs + 1e-5 * torch.eye(EMBEDDING_DIM).to(device)
    Vtr = rated_vecs.T @ ratings
    user_vector = torch.linalg.solve(VtV, Vtr)
    
    # Normalize for dot product similarity
    user_vector = user_vector / (user_vector.norm() + 1e-8)
    
    return user_vector

# --- Movie Encoder: already stored in movie_vectors ---
def movie_enc(movie_id):
    """
    Get the vector for a specific movie.
    
    Args:
        movie_id: 1-indexed movie ID
    
    Returns:
        movie_vector: normalized tensor
    """
    return movie_vectors_tensor[movie_id - 1]

# --- Recommend movies ---
def recommend(user_vector, top_k=10, exclude_ids=None):
    """
    Recommend top-K movies for a user vector.
    
    Args:
        user_vector: user embedding tensor
        top_k: number of recommendations
        exclude_ids: set of movie IDs (1-indexed) to exclude
    
    Returns:
        list of (movie_id, score) tuples
    """
    # Compute dot product with all movies
    scores = movie_vectors_tensor @ user_vector
    
    # Exclude rated movies
    if exclude_ids:
        for mid in exclude_ids:
            scores[mid - 1] = -float('inf')
    
    # Get top-K
    top_scores, top_indices = torch.topk(scores, top_k)
    
    # Convert to 1-indexed movie IDs
    recommendations = [(idx.item() + 1, score.item()) 
                       for idx, score in zip(top_indices, top_scores)]
    
    return recommendations

# --- Example usage ---
if __name__ == "__main__":
    # Example: new user rated a few movies
    new_user_ratings = {
        14: 5,   # Good movie
        12: 5,   # Good movie
        64: 5,   # Good movie
        345: 5,  # Good movie
        313: 5,  # Good movie
        258: 5   # Good movie
    }
    
    print("\n=== User Ratings ===")
    for movie_id, rating in new_user_ratings.items():
        print(f"Movie {movie_id}: {rating} stars")
    
    print("\n=== Encoding user into vector space ===")
    user_vector = user_enc(new_user_ratings)
    print(f"User vector shape: {user_vector.shape}")
    
    print("\n=== Computing dot products with all movies ===")
    print("Finding top 10 recommendations...")
    
    recommendations = recommend(
        user_vector, 
        top_k=10, 
        exclude_ids=set(new_user_ratings.keys())
    )
    
    print("\n=== Top 10 Recommended Movies ===")
    for rank, (movie_id, distance) in enumerate(recommendations, 1):
        print(f"{rank}. Movie {movie_id:4d} | Close: {distance:.4f}")