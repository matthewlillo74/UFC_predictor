"""
scripts/enrich_features.py
───────────────────────────
Backfills three new feature sets into the database:

1. WEIGHT CLASS on Fighter table
   — derived from fight history (most common weight class fought at)
   — fixes the broken percentile features that were collapsing to "Unknown"

2. UFC EXPERIENCE on FighterStats snapshots
   — ufc_fights: how many UFC fights before this date
   — ufc_wins: how many UFC wins before this date
   — captures the debut/veteran dynamic the model was blind to

3. CHAMPIONSHIP FLAG propagation
   — is_title_fight already exists on Fight, just need feature builder to use it

Run:
    python scripts/enrich_features.py
Then:
    python scripts/compute_styles.py   (recomputes percentiles with real weight classes)
    rm data/processed/training_dataset.csv
    python scripts/train_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from tqdm import tqdm
from loguru import logger
from sqlalchemy import text

from src.database import init_db, get_session, Fighter, FighterStats, Fight


def fix_weight_classes(session):
    """
    Derive weight_class for each Fighter from their fight history.
    Uses the most common weight class they competed at.
    """
    logger.info("Fixing weight classes from fight history...")

    fighters = session.query(Fighter).all()
    updated = 0

    for fighter in tqdm(fighters, desc="Weight classes"):
        # Get all fights for this fighter
        fights = session.query(Fight).filter(
            (Fight.fighter_a_id == fighter.id) | (Fight.fighter_b_id == fighter.id)
        ).all()

        if not fights:
            continue

        # Count weight classes, pick most common
        wc_counts = Counter(
            f.weight_class for f in fights
            if f.weight_class and f.weight_class.strip()
        )

        if not wc_counts:
            continue

        most_common = wc_counts.most_common(1)[0][0]
        if fighter.weight_class != most_common:
            fighter.weight_class = most_common
            updated += 1

    session.commit()
    logger.success(f"Weight classes fixed for {updated} fighters")


def backfill_ufc_experience(session):
    """
    For each FighterStats snapshot, count how many UFC fights
    the fighter had before that date.

    UFC fights = fights in our database (all ufcstats fights are UFC).
    This gives the model a 'debut vs veteran' signal.
    """
    logger.info("Backfilling UFC experience into snapshots...")

    # Get all fighters
    fighter_ids = [r[0] for r in session.execute(text("SELECT id FROM fighters")).fetchall()]
    updated = 0

    for fighter_id in tqdm(fighter_ids, desc="UFC experience"):
        # Get all completed UFC fights for this fighter, ordered by date
        fights = (
            session.query(Fight)
            .filter(
                (Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id),
                Fight.fight_date.isnot(None),
                Fight.winner_id.isnot(None),  # completed fights only
            )
            .order_by(Fight.fight_date)
            .all()
        )

        if not fights:
            continue

        # Get all snapshots for this fighter
        snapshots = (
            session.query(FighterStats)
            .filter_by(fighter_id=fighter_id)
            .order_by(FighterStats.as_of_date)
            .all()
        )

        for snap in snapshots:
            if not snap.as_of_date:
                continue
            # Count fights BEFORE this snapshot date
            prior_fights = [f for f in fights if f.fight_date < snap.as_of_date]
            prior_wins = [
                f for f in prior_fights
                if f.winner_id == fighter_id
            ]
            snap.ufc_fights = len(prior_fights)
            snap.ufc_wins = len(prior_wins)
            updated += 1

    session.commit()
    logger.success(f"UFC experience backfilled for {updated} snapshots")


def run():
    init_db()
    session = get_session()

    fix_weight_classes(session)
    backfill_ufc_experience(session)

    session.close()
    logger.success("Done. Next steps:")
    logger.success("  python scripts/compute_styles.py   # recompute percentiles with real weight classes")
    logger.success("  rm data/processed/training_dataset.csv")
    logger.success("  python scripts/train_model.py")


if __name__ == "__main__":
    run()
