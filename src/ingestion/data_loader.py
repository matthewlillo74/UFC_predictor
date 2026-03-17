"""
src/ingestion/data_loader.py
─────────────────────────────
Loads scraped fight data into the database.

This is the bridge between the scraper and the database.
It handles:
  - Deduplication (don't insert the same fighter/fight twice)
  - Name normalization (Alex Pereira vs A. Pereira vs Poatan)
  - Building pre-fight stat snapshots (the leakage-prevention layer)
  - Elo rating calculation after each fight is loaded

Run the full historical load with:
    python scripts/load_historical_data.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import Optional
from loguru import logger
from tqdm import tqdm
from sqlalchemy.orm import Session

from src.database import get_session, init_db, Fighter, Fight, Event, FighterStats, EloRating
from src.ingestion import fight_scraper
from src.features.elo_calculator import EloCalculator, update_ratings
from config import ELO_BASE_RATING, MIN_FIGHTS_REQUIRED


def normalize_name(name: str) -> str:
    """Lowercase, strip whitespace, remove punctuation for consistent matching."""
    import re
    return re.sub(r"[^\w\s]", "", name.lower().strip())


def get_or_create_fighter(session: Session, name: str, fighter_url: str = "") -> Fighter:
    """
    Get an existing fighter by name (fuzzy match) or create a new one.
    This handles the "Alex Pereira vs Poatan" naming problem.
    """
    norm = normalize_name(name)

    # Exact normalized match first
    existing = session.query(Fighter).filter_by(name_normalized=norm).first()
    if existing:
        return existing

    # URL-based match (most reliable — avoids false positives)
    if fighter_url:
        url_match = session.query(Fighter).filter_by(url=fighter_url).first()
        if url_match:
            return url_match

    # Fuzzy match — threshold 92 to avoid false positives like
    # "Piera Rodriguez" matching "Ray Rodriguez"
    all_fighters = session.query(Fighter).all()
    if all_fighters:
        from rapidfuzz import process, fuzz
        names = [(f.name_normalized, f) for f in all_fighters]
        match = process.extractOne(
            norm,
            [n[0] for n in names],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=92,
        )
        if match:
            matched_norm = match[0]
            score = match[1]
            for fn, fighter_obj in names:
                if fn == matched_norm:
                    # If both have URLs and they differ, don't match
                    if fighter_url and fighter_obj.url and fighter_url != fighter_obj.url:
                        break
                    logger.debug(f"Fuzzy matched '{name}' → '{fighter_obj.name}' ({score:.0f}%)")
                    return fighter_obj

    # Create new fighter — will fail on read-only DB (e.g. Streamlit Cloud)
    # In that case return None and let the caller skip this fight gracefully
    try:
        fighter = Fighter(name=name, name_normalized=norm, url=fighter_url or "")
        session.add(fighter)
        session.flush()  # get ID without committing
        logger.debug(f"Created new fighter: {name}")
        return fighter
    except Exception as e:
        logger.warning(f"Cannot create fighter '{name}' (read-only DB?): {e}")
        try:
            session.rollback()
        except Exception:
            pass
        return None


def enrich_fighter(session: Session, fighter: Fighter, fighter_url: str):
    """
    Fetch and store physical stats for a fighter if we don't have them yet.
    Only fetches if height/reach are missing.
    """
    if fighter.height_cm and fighter.reach_cm:
        return  # Already have data

    logger.info(f"Enriching fighter: {fighter.name}")
    data = fight_scraper.get_fighter(fighter_url)
    if not data:
        return

    fighter.height_cm = data.get("height_cm") or fighter.height_cm
    fighter.reach_cm = data.get("reach_cm") or fighter.reach_cm
    fighter.stance = data.get("stance") or fighter.stance
    fighter.date_of_birth = data.get("date_of_birth") or fighter.date_of_birth
    session.flush()


def load_all_events(session: Session, limit: Optional[int] = None):
    """
    Full historical data load.
    Scrapes all UFC events and loads fights + fighters into the DB.

    Args:
        limit: If set, only load this many events (useful for testing)
    """
    logger.info("Starting full historical data load...")

    events = fight_scraper.get_all_events()
    # Reverse so we load oldest first (most complete data, avoids upcoming events)
    events = list(reversed(events))
    if limit:
        events = events[:limit]
        logger.info(f"Limited to {limit} events for testing")

    logger.info(f"Loading {len(events)} events...")

    elo_ratings: dict[int, float] = {}  # fighter_id -> current elo

    for event_data in tqdm(events, desc="Events"):
        # Skip upcoming events with no date
        if event_data["date"] is None:
            logger.debug(f"Skipping dateless event: {event_data['name']}")
            continue

        # Skip if already loaded
        existing_event = session.query(Event).filter_by(name=event_data["name"]).first()
        if existing_event:
            logger.debug(f"Skipping already-loaded event: {event_data['name']}")
            continue

        # Create event
        event = Event(
            name=event_data["name"],
            date=event_data["date"],
            url=event_data.get("url", ""),
        )
        session.add(event)
        session.flush()

        # Load fights for this event
        fights = fight_scraper.get_event_fights(event_data["url"])
        for fight_data in fights:
            _load_fight(session, event, fight_data, elo_ratings)

        session.commit()
        logger.debug(f"Loaded: {event_data['name']} ({len(fights)} fights)")

    logger.success(f"Historical load complete. {len(events)} events loaded.")


def _load_fight(
    session: Session,
    event: Event,
    fight_data: dict,
    elo_ratings: dict,
):
    """Load a single fight and update Elo ratings."""
    import random

    # ufcstats always lists winner first — randomly swap so fighter_a isn't
    # always the winner. This is critical for an unbiased training dataset.
    if random.random() < 0.5:
        fight_data = fight_data.copy()
        fight_data["fighter_a_name"], fight_data["fighter_b_name"] = (
            fight_data["fighter_b_name"], fight_data["fighter_a_name"]
        )
        fight_data["fighter_a_url"], fight_data["fighter_b_url"] = (
            fight_data.get("fighter_b_url", ""), fight_data.get("fighter_a_url", "")
        )
        # Flip winner
        if fight_data["winner"] == "fighter_a":
            fight_data["winner"] = "fighter_b"
        elif fight_data["winner"] == "fighter_b":
            fight_data["winner"] = "fighter_a"

    # Get or create fighters
    fighter_a = get_or_create_fighter(
        session, fight_data["fighter_a_name"], fight_data.get("fighter_a_url", "")
    )
    fighter_b = get_or_create_fighter(
        session, fight_data["fighter_b_name"], fight_data.get("fighter_b_url", "")
    )

    # Skip if fight already exists
    existing = session.query(Fight).filter_by(
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        fight_date=event.date,
    ).first()
    if existing:
        return

    # Determine winner ID
    winner_id = None
    if fight_data["winner"] == "fighter_a":
        winner_id = fighter_a.id
    elif fight_data["winner"] == "fighter_b":
        winner_id = fighter_b.id

    # Create fight record
    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        fight_date=event.date,
        weight_class=fight_data.get("weight_class", ""),
        is_title_fight=fight_data.get("is_title_fight", False),
        winner_id=winner_id,
        method=fight_data.get("method"),
        finish_round=fight_data.get("finish_round"),
        finish_time=fight_data.get("finish_time"),
        fight_url=fight_data.get("fight_url", ""),
    )
    session.add(fight)
    session.flush()

    # Update Elo ratings
    rating_a = elo_ratings.get(fighter_a.id, ELO_BASE_RATING)
    rating_b = elo_ratings.get(fighter_b.id, ELO_BASE_RATING)

    winner_key = fight_data["winner"]  # "fighter_a" | "fighter_b" | "draw"
    method = fight_data.get("method", "Decision")

    new_a, new_b = update_ratings(
        rating_a, rating_b,
        winner="a" if winner_key == "fighter_a" else ("b" if winner_key == "fighter_b" else "draw"),
        method=method.lower().replace("/", "_").replace(" ", "_"),
    )

    elo_ratings[fighter_a.id] = new_a
    elo_ratings[fighter_b.id] = new_b

    # Store Elo snapshots
    session.add(EloRating(fighter_id=fighter_a.id, rating=new_a, after_fight_id=fight.id))
    session.add(EloRating(fighter_id=fighter_b.id, rating=new_b, after_fight_id=fight.id))


def build_fighter_stats_snapshots(session: Session):
    """
    After loading all historical fights, build pre-fight stat snapshots.

    This iterates every fight in chronological order and computes each
    fighter's stats AS OF the fight date — the key leakage-prevention step.

    These snapshots are stored in FighterStats and used by FeatureBuilder.
    """
    logger.info("Building fighter stats snapshots...")

    fights = session.query(Fight).order_by(Fight.fight_date).all()
    # fighter_id -> running stats dict
    running_stats: dict[int, dict] = {}

    for fight in tqdm(fights, desc="Building snapshots"):
        for fighter_id in [fight.fighter_a_id, fight.fighter_b_id]:
            stats = running_stats.get(fighter_id, _empty_stats())

            # Check if snapshot already exists for this fight date
            existing = session.query(FighterStats).filter_by(
                fighter_id=fighter_id,
                as_of_date=fight.fight_date,
            ).first()

            if not existing:
                snapshot = FighterStats(
                    fighter_id=fighter_id,
                    as_of_date=fight.fight_date,
                    wins=stats["wins"],
                    losses=stats["losses"],
                    draws=stats["draws"],
                    wins_ko_tko=stats["wins_ko_tko"],
                    wins_sub=stats["wins_sub"],
                    wins_decision=stats["wins_decision"],
                    win_rate=stats["wins"] / max(stats["wins"] + stats["losses"], 1),
                    finish_rate=(stats["wins_ko_tko"] + stats["wins_sub"]) / max(stats["wins"], 1),
                    win_streak=stats["win_streak"],
                    days_since_last_fight=stats.get("days_since_last_fight"),
                )
                session.add(snapshot)

        # Now update running stats AFTER saving the snapshot
        _update_running_stats(running_stats, fight)

    session.commit()
    logger.success("Fighter stats snapshots built.")


def _empty_stats() -> dict:
    return {
        "wins": 0, "losses": 0, "draws": 0,
        "wins_ko_tko": 0, "wins_sub": 0, "wins_decision": 0,
        "win_streak": 0,
        "last_fight_date": None,
        "days_since_last_fight": None,
        "recent_results": [],  # list of "W"/"L" for last N fights
    }


def _update_running_stats(running_stats: dict, fight: Fight):
    """Update the running stats dict for both fighters after a fight."""
    for fighter_id, opponent_id, is_winner in [
        (fight.fighter_a_id, fight.fighter_b_id, fight.winner_id == fight.fighter_a_id),
        (fight.fighter_b_id, fight.fighter_a_id, fight.winner_id == fight.fighter_b_id),
    ]:
        if fighter_id not in running_stats:
            running_stats[fighter_id] = _empty_stats()

        stats = running_stats[fighter_id]

        # Days since last fight
        if stats["last_fight_date"] and fight.fight_date:
            stats["days_since_last_fight"] = (fight.fight_date - stats["last_fight_date"]).days
        stats["last_fight_date"] = fight.fight_date

        if fight.winner_id == fighter_id:
            # Win
            stats["wins"] += 1
            method = fight.method or ""
            if "KO" in method.upper() or "TKO" in method.upper():
                stats["wins_ko_tko"] += 1
            elif "SUB" in method.upper():
                stats["wins_sub"] += 1
            else:
                stats["wins_decision"] += 1
            stats["win_streak"] = max(stats["win_streak"] + 1, 1)
            stats["recent_results"].append("W")
        elif fight.winner_id is not None:
            # Loss
            stats["losses"] += 1
            stats["win_streak"] = min(stats["win_streak"] - 1, -1)
            stats["recent_results"].append("L")
        else:
            # Draw/NC
            stats["draws"] += 1
            stats["win_streak"] = 0
            stats["recent_results"].append("D")

        # Keep only last 5 results
        stats["recent_results"] = stats["recent_results"][-5:]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of events (for testing)")
    parser.add_argument("--stats-only", action="store_true", help="Only rebuild stats snapshots")
    args = parser.parse_args()

    init_db()
    session = get_session()

    if not args.stats_only:
        load_all_events(session, limit=args.limit)

    build_fighter_stats_snapshots(session)
    session.close()
