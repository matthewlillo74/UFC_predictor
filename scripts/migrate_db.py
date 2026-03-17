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
    ("fighter_stats", "ufc_fights",         "INTEGER DEFAULT 0"),
    ("fighter_stats", "ufc_wins",           "INTEGER DEFAULT 0"),
    # Knockdown durability — real data from fight detail pages
    ("fighter_stats", "kd_landed_per_fight",   "REAL DEFAULT NULL"),
    ("fighter_stats", "kd_absorbed_per_fight", "REAL DEFAULT NULL"),
    ("fighter_stats", "kd_ratio",              "REAL DEFAULT NULL"),
    # Fight URL for scraping detail pages
    ("fights", "fight_url", "TEXT DEFAULT ''"),
    # Event URL for re-scraping event pages
    ("events", "url", "TEXT DEFAULT ''"),
    # Opponent style vulnerability
    ("fighter_stats", "winrate_vs_wrestlers", "REAL DEFAULT NULL"),
    ("fighter_stats", "winrate_vs_strikers",  "REAL DEFAULT NULL"),
    ("fighter_stats", "winrate_vs_pressure",  "REAL DEFAULT NULL"),
    # Cardio decay from round-level data
    ("fighter_stats", "cardio_decay",         "REAL DEFAULT NULL"),
    ("fighter_stats", "early_output_share",   "REAL DEFAULT NULL"),
    # Rolling style windows
    ("fighter_stats", "style_pressure_l3",    "REAL DEFAULT NULL"),
    ("fighter_stats", "style_wrestling_l3",   "REAL DEFAULT NULL"),
    ("fighter_stats", "style_striker_l3",     "REAL DEFAULT NULL"),
    ("fighter_stats", "style_pressure_l5",    "REAL DEFAULT NULL"),
    ("fighter_stats", "style_wrestling_l5",   "REAL DEFAULT NULL"),
    ("fighter_stats", "style_striker_l5",     "REAL DEFAULT NULL"),
    # Strike location rates
    ("fighter_stats", "head_strike_rate",     "REAL DEFAULT NULL"),
    ("fighter_stats", "body_strike_rate",     "REAL DEFAULT NULL"),
    ("fighter_stats", "leg_strike_rate",      "REAL DEFAULT NULL"),
    ("fighter_stats", "ground_strike_share",  "REAL DEFAULT NULL"),
    # Strike location columns on fight_stats
    ("fight_stats", "head_landed",     "INTEGER DEFAULT NULL"),
    ("fight_stats", "head_attempted",  "INTEGER DEFAULT NULL"),
    ("fight_stats", "body_landed",     "INTEGER DEFAULT NULL"),
    ("fight_stats", "body_attempted",  "INTEGER DEFAULT NULL"),
    ("fight_stats", "leg_landed",      "INTEGER DEFAULT NULL"),
    ("fight_stats", "leg_attempted",   "INTEGER DEFAULT NULL"),
    ("fight_stats", "distance_landed", "INTEGER DEFAULT NULL"),
    ("fight_stats", "clinch_landed",   "INTEGER DEFAULT NULL"),
    ("fight_stats", "ground_landed",   "INTEGER DEFAULT NULL"),
]

# Create round_stats table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS round_stats (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        fight_id              INTEGER NOT NULL REFERENCES fights(id),
        fighter_id            INTEGER NOT NULL REFERENCES fighters(id),
        round_num             INTEGER NOT NULL,
        knockdowns            INTEGER DEFAULT 0,
        sig_strikes_landed    INTEGER,
        sig_strikes_attempted INTEGER,
        total_strikes_landed  INTEGER,
        takedowns_landed      INTEGER,
        takedowns_attempted   INTEGER,
        submission_attempts   INTEGER,
        reversals             INTEGER,
        control_time_secs     INTEGER,
        head_landed           INTEGER,
        body_landed           INTEGER,
        leg_landed            INTEGER,
        distance_landed       INTEGER,
        clinch_landed         INTEGER,
        ground_landed         INTEGER,
        scraped_at            DATETIME,
        UNIQUE(fight_id, fighter_id, round_num)
    )
""")
logger.success("round_stats table ready")

# Create fight_stats table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS fight_stats (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        fight_id              INTEGER NOT NULL REFERENCES fights(id),
        fighter_id            INTEGER NOT NULL REFERENCES fighters(id),
        knockdowns            INTEGER DEFAULT 0,
        knockdowns_absorbed   INTEGER DEFAULT 0,
        sig_strikes_landed    INTEGER,
        sig_strikes_attempted INTEGER,
        sig_strikes_absorbed  INTEGER,
        total_strikes_landed  INTEGER,
        takedowns_landed      INTEGER,
        takedowns_attempted   INTEGER,
        submission_attempts   INTEGER,
        reversals             INTEGER,
        control_time_secs     INTEGER,
        scraped_at            DATETIME,
        UNIQUE(fight_id, fighter_id)
    )
""")
logger.success("fight_stats table ready")

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
