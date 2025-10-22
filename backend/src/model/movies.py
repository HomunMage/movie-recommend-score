# src/model/movies.py

from .db import get_pool


async def get_nearest_movies(embedding: list[float], limit: int = 10):
    """
    Find the nearest movies to a given embedding vector using cosine similarity.
    
    Args:
        embedding: List of floats representing the user embedding
        limit: Number of nearest neighbors to return (default 10)
    
    Returns:
        List of dicts with movie_id and similarity score
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Convert Python list to PostgreSQL vector format
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        # Use cosine distance operator (<=>)
        # Lower distance = higher similarity
        rows = await conn.fetch(
            """
            SELECT 
                movie_id,
                1 - (embedding <=> $1::vector) as similarity
            FROM movie_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            embedding_str,
            limit
        )
        
        return [
            {
                "movie_id": row["movie_id"],
                "similarity": float(row["similarity"])
            }
            for row in rows
        ]


async def get_movie_by_id(movie_id: int):
    """
    Get movie details by movie_id from the movies table.
    
    Args:
        movie_id: The movie ID to retrieve
    
    Returns:
        Dict with movie details or None if not found
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT movie_id, title, genres, created_at
            FROM movies
            WHERE movie_id = $1
            """,
            movie_id
        )
        
        if not row:
            return None
        
        return {
            "movie_id": row["movie_id"],
            "title": row["title"],
            "genres": row["genres"],
            "created_at": row["created_at"]
        }


async def get_nearest_movies_with_details(embedding: list[float], limit: int = 10):
    """
    Find nearest movies and include their metadata (title, genres).
    
    Args:
        embedding: List of floats representing the user embedding
        limit: Number of nearest neighbors to return
    
    Returns:
        List of dicts with movie details and similarity scores
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        rows = await conn.fetch(
            """
            SELECT 
                me.movie_id,
                m.title,
                m.genres,
                1 - (me.embedding <=> $1::vector) as similarity
            FROM movie_embeddings me
            LEFT JOIN movies m ON me.movie_id = m.movie_id
            ORDER BY me.embedding <=> $1::vector
            LIMIT $2
            """,
            embedding_str,
            limit
        )
        
        return [
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "genres": row["genres"],
                "similarity": float(row["similarity"])
            }
            for row in rows
        ]