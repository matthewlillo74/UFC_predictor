"""
scripts/migrate_db.py
──────────────────────
Adds any missing columns to the existing database.
Safe to run multiple times — skips columns that already exist.

Run after any database.py schema change:
    python scripts/migrate_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from loguru import logger
from config import DATABASE_URL

db_path = DATABASE_URL.replace("sqlite:///", "")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

migrations = [
    # Fighter table
    ("fighters", "url",    "TEXT DEFAULT ''"),
    ("fighters", "wins",   "INTEGER DEFAULT 0"),
    ("fighters", "losses", "INTEGER DEFAULT 0"),
    ("fighters", "draws",  "INTEGER DEFAULT 0"),
    # Style fingerprint columns
    ("fighter_stats", "style_pressure",     "REAL DEFAULT NULL"),
    ("fighter_stats", "style_wrestling",    "REAL DEFAULT NULL"),
    ("fighter_stats", "style_striker",      "REAL DEFAULT NULL"),
    ("fighter_stats", "style_finisher",     "REAL DEFAULT NULL"),
    ("fighter_stats", "grappling_defense",  "REAL DEFAULT NULL"),
    ("fighter_stats", "recent_finish_rate", "REAL DEFAULT NULL"),
    ("fighter_stats", "momentum_score",     "REAL DEFAULT NULL"),
    ("fighter_stats", "slpm_pctile",        "REAL DEFAULT NULL"),
    ("fighter_stats", "td_avg_pctile",      "REAL DEFAULT NULL"),
]

for table, column, col_type in migrations:
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.success(f"Added: {table}.{column}")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.debug(f"Already exists: {table}.{column}")
        else:
            logger.error(f"Failed {table}.{column}: {e}")

conn.commit()
conn.close()
logger.success("Migration complete")
