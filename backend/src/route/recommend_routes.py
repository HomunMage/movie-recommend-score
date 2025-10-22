# src/route/recommend_routes.py

from typing import List, Dict
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import httpx
import os

from model.movies import get_nearest_movies_with_details

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


# --- Pydantic models ---
class UserRatings(BaseModel):
    ratings: Dict[str, int] = Field(
        ...,
        description="Dictionary of movie_id: rating pairs"
    )


class RecommendRequest(BaseModel):
    user_ratings: UserRatings


class RecommendationResponse(BaseModel):
    movie_id: int
    similarity: float


# --- Route ---
@router.post("", response_model=List[RecommendationResponse])
async def recommend_movies(request: RecommendRequest, limit: int = 5):
    """
    Get movie recommendations based on user ratings.
    
    This endpoint:
    1. Sends user ratings to the MLflow/Bento service to get user embedding
    2. Finds nearest movies using the embedding
    3. Returns movie details with similarity scores
    """
    # Get Bento service URL from environment
    bento_url = os.getenv("BENTO_URL", "http://bento:3000/encode_user")
    
    # Step 1: Call Bento service to get user embedding
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                bento_url,
                json={"user_ratings": request.user_ratings.dict()}
            )
            response.raise_for_status()
            embedding_data = response.json()
            user_embedding = embedding_data["vector"]
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"MLflow/Bento service unreachable: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"MLflow/Bento service error: {e.response.text}"
            )
        except (KeyError, TypeError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid response from MLflow/Bento service: {e}"
            )
    
    # Step 2: Find nearest movies with details
    try:
        recommendations = await get_nearest_movies_with_details(
            embedding=user_embedding,
            limit=limit
        )
        return recommendations
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding recommendations: {str(e)}"
        )