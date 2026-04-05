"""
scripts/setup_db.py
────────────────────
Initialize database schema. Run once before first launch.

Usage:
    python scripts/setup_db.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from backend.db.database import init_db, check_db_connection


async def main():
    logger.info("Checking database connection...")
    ok = await check_db_connection()
    if not ok:
        logger.error(
            "Cannot connect to database. "
            "Make sure PostgreSQL is running and DATABASE_URL is correct."
        )
        sys.exit(1)

    logger.info("Creating tables...")
    await init_db()
    logger.success("Database setup complete!")


if __name__ == "__main__":
    asyncio.run(main())
