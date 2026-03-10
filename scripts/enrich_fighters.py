"""
scripts/enrich_fighters.py
───────────────────────────
Fetches physical attributes and career stats for every fighter in the DB.

This fills in: height, reach, stance, date_of_birth, slpm, strike_accuracy,
sapm, strike_defense, td_avg, td_accuracy, td_defense, sub_avg

Run this AFTER load_historical_data.py and BEFORE train_model.py.
Takes ~45-60 min for all ~2,600 fighters (polite 1.5s delay between requests).

Usage:
    python scripts/enrich_fighters.py
    python scripts/enrich_fighters.py --limit 50   # test with 50 fighters
    python scripts/enrich_fighters.py --missing-only  # skip already-enriched
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tqdm import tqdm
from loguru import logger

from src.database import init_db, get_session, Fighter, FighterStats
from src.ingestion.fight_scraper import get_fighter, search_fighter


def enrich_all_fighters(limit: int = None, missing_only: bool = True):
    init_db()
    session = get_session()

    query = session.query(Fighter)

    if missing_only:
        # Only fetch fighters where we're missing key stats
        query = query.filter(
            (Fighter.reach_cm == None) | (Fighter.height_cm == None)
        )

    fighters = query.all()

    if limit:
        fighters = fighters[:limit]

    logger.info(f"Enriching {len(fighters)} fighters...")
    enriched = 0
    failed = 0

    for fighter in tqdm(fighters, desc="Enriching fighters"):
        try:
            # Try direct URL first (stored during load), then search
            url = None

            # Check if we have a URL stored anywhere — look in stats or search
            url = search_fighter(fighter.name)

            if not url:
                logger.debug(f"No URL found for: {fighter.name}")
                failed += 1
                continue

            data = get_fighter(url)
            if not data:
                failed += 1
                continue

            # Update fighter physical attributes
            if data.get("height_cm"):
                fighter.height_cm = data["height_cm"]
            if data.get("reach_cm"):
                fighter.reach_cm = data["reach_cm"]
            if data.get("stance"):
                fighter.stance = data["stance"]
            if data.get("date_of_birth"):
                fighter.date_of_birth = data["date_of_birth"]

            # Update the most recent FighterStats snapshot with UFC performance stats
            # This enriches the snapshot used for upcoming fight predictions
            latest_stats = (
                session.query(FighterStats)
                .filter_by(fighter_id=fighter.id)
                .order_by(FighterStats.as_of_date.desc())
                .first()
            )

            if latest_stats:
                if data.get("slpm") is not None:
                    latest_stats.slpm = data["slpm"]
                if data.get("strike_accuracy") is not None:
                    latest_stats.strike_accuracy = data["strike_accuracy"]
                if data.get("sapm") is not None:
                    latest_stats.sapm = data["sapm"]
                if data.get("strike_defense") is not None:
                    latest_stats.strike_defense = data["strike_defense"]
                if data.get("td_avg") is not None:
                    latest_stats.td_avg = data["td_avg"]
                if data.get("td_accuracy") is not None:
                    latest_stats.td_accuracy = data["td_accuracy"]
                if data.get("td_defense") is not None:
                    latest_stats.td_defense = data["td_defense"]
                if data.get("sub_avg") is not None:
                    latest_stats.sub_avg = data["sub_avg"]

            session.commit()
            enriched += 1

        except KeyboardInterrupt:
            logger.warning("Interrupted — saving progress...")
            session.commit()
            break
        except Exception as e:
            logger.debug(f"Failed to enrich {fighter.name}: {e}")
            session.rollback()
            failed += 1

    session.close()
    logger.success(f"Enrichment complete. Enriched: {enriched}, Failed: {failed}")
    logger.info("Now rebuild the training dataset: delete data/processed/training_dataset.csv and run train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-only", action="store_true", default=True)
    parser.add_argument("--all", action="store_true", help="Re-enrich all fighters even if already done")
    args = parser.parse_args()

    enrich_all_fighters(
        limit=args.limit,
        missing_only=not args.all,
    )
