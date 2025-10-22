# src/model/__init__.py

"""
Model package initialization.
Provides database access and data models for companionship and persona features.
"""

from .db import init_db, get_pool, close_db
from .movies import (
    get_nearest_movies,
    get_movie_by_id,
    get_nearest_movies_with_details
)