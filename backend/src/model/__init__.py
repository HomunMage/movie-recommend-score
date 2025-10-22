# src/model/__init__.py

"""
Model package initialization.
Provides database access and data models for companionship and persona features.
"""

from .db import init_db, get_pool, close_db
