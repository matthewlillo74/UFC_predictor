"""
scripts/backfill_striking_stats.py
────────────────────────────────────
Backfills UFC striking/grappling stats into all historical FighterStats
snapshots that currently have NULL values.

THE PROBLEM:
  build_fighter_stats_snapshots() only tracks wins/losses/streaks.
  enrich_fighters.py fetches career stats from the UFC profile page,
  but only updates the LATEST snapshot. So 17,000 historical snapshots
  have NULL slpm, td_avg, etc., and the model sees 0 importance.

THE FIX:
  For each fighter, take their career-average stats (from any non-null snapshot)
  and propagate them backward to all their NULL historical snapshots.

  This is a career-proxy approach — not perfectly accurate for early fights,
  but vastly better than NULL. A fighter's striking style is largely consistent
  throughout their career (wrestlers don't become strikers), so career averages
  are a reasonable historical proxy.

  The correct long-term fix would be scraping per-fight stats from individual
  fight pages — that's a much larger scraping project.

Run:
    python scripts/backfill_striking_stats.py
Then retrain:
    rm data/processed/training_dataset.csv
    python scripts/train_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from loguru import logger
from sqlalchemy import text

from src.database import init_db, get_session, FighterStats, Fighter

STAT_COLS = [
    "slpm", "strike_accuracy", "sapm", "strike_defense",
    "td_avg", "td_accuracy", "td_defense", "sub_avg",
    "recent_win_rate",
]


def run():
    init_db()
    session = get_session()

    # Get all fighter IDs
    fighter_ids = [r[0] for r in session.execute(text("SELECT id FROM fighters")).fetchall()]
    logger.info(f"Backfilling stats for {len(fighter_ids)} fighters...")

    backfilled_fighters = 0
    backfilled_snapshots = 0
    skipped_no_data = 0

    for fighter_id in tqdm(fighter_ids, desc="Backfilling"):
        # Get all snapshots for this fighter, ordered by date
        snapshots = (
            session.query(FighterStats)
            .filter_by(fighter_id=fighter_id)
            .order_by(FighterStats.as_of_date)
            .all()
        )

        if not snapshots:
            continue

        # Find the best reference stats — prefer the most recent non-null snapshot
        reference = None
        for snap in reversed(snapshots):
            if snap.slpm is not None and snap.slpm > 0:
                reference = snap
                break

        if reference is None:
            # No stats at all for this fighter — nothing to backfill from
            skipped_no_data += 1
            continue

        # Build reference dict
        ref_stats = {col: getattr(reference, col) for col in STAT_COLS}

        # Backfill all snapshots that have NULL slpm
        fighter_updated = False
        for snap in snapshots:
            if snap.slpm is None or snap.slpm == 0:
                for col, val in ref_stats.items():
                    if val is not None and getattr(snap, col) is None:
                        setattr(snap, col, val)
                backfilled_snapshots += 1
                fighter_updated = True

        if fighter_updated:
            backfilled_fighters += 1

    session.commit()
    session.close()

    logger.success(f"Backfill complete:")
    logger.success(f"  Fighters updated:   {backfilled_fighters}")
    logger.success(f"  Snapshots updated:  {backfilled_snapshots}")
    logger.success(f"  Skipped (no data):  {skipped_no_data}")
    logger.info("Next step: rm data/processed/training_dataset.csv && python scripts/train_model.py")


if __name__ == "__main__":
    run()
