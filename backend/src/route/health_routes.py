# src/route/health_routes.py

from fastapi import APIRouter, HTTPException
from model.db import get_pool

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/db")
async def database_health():
    """Check database connection health."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")
