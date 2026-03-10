"""
scripts/load_historical_data.py
─────────────────────────────────
One-time script to load all UFC historical fight data into the database.

This will take 30-60 minutes to run the first time (scraping ~700 events).
Use --limit 10 to test with just 10 events first.

Usage:
    python scripts/load_historical_data.py              # full load
    python scripts/load_historical_data.py --limit 10  # test with 10 events
    python scripts/load_historical_data.py --stats-only # just rebuild snapshots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from loguru import logger
from src.database import init_db, get_session
from src.ingestion.data_loader import load_all_events, build_fighter_stats_snapshots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load UFC historical fight data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of events (useful for testing)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip scraping, just rebuild stats snapshots from existing DB data")
    args = parser.parse_args()

    logger.info("Initializing database...")
    init_db()
    session = get_session()

    try:
        if not args.stats_only:
            logger.info(f"Starting historical load {'(full)' if not args.limit else f'(limit: {args.limit})'}")
            load_all_events(session, limit=args.limit)

        logger.info("Building pre-fight stats snapshots...")
        build_fighter_stats_snapshots(session)

    except KeyboardInterrupt:
        logger.warning("Interrupted — committing what we have so far...")
        session.commit()
    finally:
        session.close()

    logger.success("Done. Run 'python scripts/check_db.py' to verify.")
