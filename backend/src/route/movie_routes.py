# src/route/movie_routes.py

from typing import List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query

from model.movies import (
    get_nearest_movies,
    get_movie_by_id,
    get_nearest_movies_with_details
)

router = APIRouter(prefix="/movies", tags=["Movies"])


# --- Pydantic models ---
class NearestMoviesRequest(BaseModel):
    embedding: List[float] = Field(
        ..., 
        description="User embedding vector (must match EMBEDDING_DIM)",
        min_items=32,
        max_items=32
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of nearest movies to return"
    )


class MovieResponse(BaseModel):
    movie_id: int
    similarity: float


class MovieDetailResponse(BaseModel):
    movie_id: int
    title: str | None
    genres: str | None
    similarity: float


# --- Routes ---
@router.post("/nearest", response_model=List[MovieResponse])
async def find_nearest_movies(request: NearestMoviesRequest):
    """
    Find the nearest movies to a given embedding vector.
    
    Returns a list of movie IDs with their similarity scores,
    ordered by similarity (highest first).
    """
    try:
        results = await get_nearest_movies(
            embedding=request.embedding,
            limit=request.limit
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding nearest movies: {str(e)}"
        )


@router.post("/nearest/details", response_model=List[MovieDetailResponse])
async def find_nearest_movies_with_details(request: NearestMoviesRequest):
    """
    Find the nearest movies with full metadata (title, genres).
    
    Returns a list of movies with details and similarity scores,
    ordered by similarity (highest first).
    """
    try:
        results = await get_nearest_movies_with_details(
            embedding=request.embedding,
            limit=request.limit
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding nearest movies: {str(e)}"
        )


@router.get("/{movie_id}")
async def get_movie(movie_id: int):
    """
    Retrieve movie details by movie ID.
    """
    try:
        movie = await get_movie_by_id(movie_id)
        if not movie:
            raise HTTPException(
                status_code=404,
                detail=f"Movie with ID {movie_id} not found"
            )
        return movie
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving movie: {str(e)}"
        )