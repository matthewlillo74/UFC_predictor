"""
scripts/recompute_elo.py
──────────────────────────
Rebuilds the entire EloRating history from scratch using experience-decayed
K-factors (see elo_calculator.decayed_k_factor) instead of the flat K=32 every
existing EloRating row was computed with.

Elo is inherently chronological — each rating builds on the fighter's prior
rating — so decaying K-factor can't be applied retroactively to existing rows;
the whole history has to be replayed in fight-date order from ELO_BASE_RATING.
This does NOT touch Fight/Fighter/FighterStats rows, only EloRating — all the
Elo-derived features (elo_diff, avg_opponent_elo_diff, elo_trend_diff,
elo_uncertainty_diff, elo_vs_peak_diff) read directly from EloRating via
EloCalculator at feature-build time, not from a FighterStats column, so no
snapshot rebuild is needed. Just retrain afterward to pick up the new values.

Usage:
    python scripts/recompute_elo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from loguru import logger

from src.database import init_db, get_session, Fight, EloRating
from src.features.elo_calculator import update_ratings
from config import ELO_BASE_RATING


def recompute_elo(session):
    logger.info("Deleting existing EloRating history...")
    deleted = session.query(EloRating).delete()
    session.commit()
    logger.info(f"Deleted {deleted} old EloRating rows")

    fights = (
        session.query(Fight)
        .filter(Fight.winner_id.isnot(None), Fight.fight_date.isnot(None))
        .order_by(Fight.fight_date)
        .all()
    )
    logger.info(f"Replaying {len(fights)} fights chronologically with decayed K-factor...")

    elo_ratings: dict[int, float] = {}
    fight_counts: dict[int, int] = {}
    new_rows = []

    for fight in tqdm(fights, desc="Recomputing Elo"):
        rating_a = elo_ratings.get(fight.fighter_a_id, ELO_BASE_RATING)
        rating_b = elo_ratings.get(fight.fighter_b_id, ELO_BASE_RATING)
        fights_a = fight_counts.get(fight.fighter_a_id, 0)
        fights_b = fight_counts.get(fight.fighter_b_id, 0)

        if fight.winner_id == fight.fighter_a_id:
            winner = "a"
        elif fight.winner_id == fight.fighter_b_id:
            winner = "b"
        else:
            winner = "draw"
        method = (fight.method or "Decision").lower().replace("/", "_").replace(" ", "_")

        new_a, new_b = update_ratings(
            rating_a, rating_b, winner=winner, method=method,
            fights_a=fights_a, fights_b=fights_b,
        )

        elo_ratings[fight.fighter_a_id] = new_a
        elo_ratings[fight.fighter_b_id] = new_b
        fight_counts[fight.fighter_a_id] = fights_a + 1
        fight_counts[fight.fighter_b_id] = fights_b + 1

        new_rows.append(EloRating(fighter_id=fight.fighter_a_id, rating=new_a, after_fight_id=fight.id))
        new_rows.append(EloRating(fighter_id=fight.fighter_b_id, rating=new_b, after_fight_id=fight.id))

    session.bulk_save_objects(new_rows)
    session.commit()
    logger.success(f"Recomputed Elo for {len(fights)} fights ({len(new_rows)} EloRating rows)")


def main():
    init_db()
    session = get_session()
    recompute_elo(session)
    session.close()
    logger.info("Next: rm data/processed/training_dataset.csv && python scripts/train_model.py "
                "to pick up the new Elo history in training features.")


if __name__ == "__main__":
    main()
