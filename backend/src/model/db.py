# src/model/db.py

import asyncpg
import os
import asyncio

pool = None

async def init_db(run_migrations=False):
    """
    Initialize a PostgreSQL connection pool using asyncpg.
    Optionally run migrations if `run_migrations=True`.
    """
    global pool
    if pool:
        return pool
    
    postgres_url = os.getenv("POSTGRES_URL", "postgres:5432")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "")
    postgres_db = os.getenv("POSTGRES_DB", "postgres")
    
    try:
        host, port = postgres_url.split(":")
        port = int(port)
    except ValueError:
        raise ValueError(f"Invalid POSTGRES_URL format: {postgres_url}. Expected 'host:port'.")

    for attempt in range(5):
        try:
            pool = await asyncpg.create_pool(
                user=postgres_user,
                password=postgres_password,
                database=postgres_db,
                host=host,
                port=port,
                min_size=1,
                max_size=5,
            )
            print(f"✅ Connected to PostgreSQL at {host}:{port}")
            
            if run_migrations:
                from .migration import run_migrations
                await run_migrations()
            
            return pool
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/5: Failed to connect to PostgreSQL ({e})")
            await asyncio.sleep(3)
    
    raise RuntimeError("❌ Could not connect to PostgreSQL after multiple attempts")

async def get_pool():
    """Return the global database connection pool, initializing if needed."""
    global pool
    if not pool:
        await init_db(run_migrations=False)
    return pool

async def close_db():
    """Close the database connection pool."""
    global pool
    if pool:
        await pool.close()
        pool = None
        print("✅ Database connection pool closed")
