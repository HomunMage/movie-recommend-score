# src/model/migration.py

from pathlib import Path
from .db import get_pool

async def run_migrations():
    """
    Run all SQL migration files in order from src/sql/migration/.
    """
    pool = await get_pool()
    migration_dir = Path(__file__).parent.parent / "sql" / "migration"

    if not migration_dir.exists():
        print(f"⚠️ Migration directory not found: {migration_dir}")
        return

    migration_files = sorted(migration_dir.glob("*.sql"))

    if not migration_files:
        print("ℹ️ No migration files found")
        return

    async with pool.acquire() as conn:
        for migration_file in migration_files:
            try:
                print(f"🔄 Running migration: {migration_file.name}")
                sql_content = migration_file.read_text()
                await conn.execute(sql_content)
                print(f"✅ Migration completed: {migration_file.name}")
            except Exception as e:
                print(f"❌ Migration failed ({migration_file.name}): {e}")
                raise
