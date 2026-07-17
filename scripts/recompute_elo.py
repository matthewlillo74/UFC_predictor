"""
scripts/recompute_elo.py
──────────────────────────
Rebuilds the entire EloRating history from scratch, either with the flat
K=32 every EloRating row historically used, or with experience-decayed
K-factors (see elo_calculator.decayed_k_factor).

STATUS AS OF 2026-07-16: `main` runs FLAT K-factor deliberately — the 6-item
accuracy queue (including the Elo decay change) didn't reach statistical
significance (McNemar's test) and per-division calibration was found to
actively harm probability quality, so `main` was reverted to the
pre-queue baseline (73 features, flat Elo, no calibration) as the most
defensible known state for live use. The decayed-K logic and the rest of
the queue are preserved on branch `parked/accuracy-queue-2026-07-16` to
revisit once there's more live data. See AGENT_HANDOFF.md / SESSION_LOG.md.

Elo is inherently chronological — each rating builds on the fighter's prior
rating — so switching between flat/decayed can't be applied retroactively to
existing rows; the whole history has to be replayed in fight-date order from
ELO_BASE_RATING. This does NOT touch Fight/Fighter/FighterStats rows, only
EloRating — all the Elo-derived features (elo_diff, avg_opponent_elo_diff,
elo_trend_diff, elo_uncertainty_diff, elo_vs_peak_diff) read directly from
EloRating via EloCalculator at feature-build time, not from a FighterStats
column, so no snapshot rebuild is needed. Just retrain afterward to pick up
the new values.

Usage:
    python scripts/recompute_elo.py            # flat K (current main default)
    python scripts/recompute_elo.py --decayed   # experience-decayed K (parked branch)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from loguru import logger

from src.database import init_db, get_session, Fight, EloRating
from src.features.elo_calculator import update_ratings
from config import ELO_BASE_RATING, ELO_K_FACTOR


def recompute_elo(session, flat: bool = True):
    """
    Args:
        flat: if True (default, matches current main), use the flat
              ELO_K_FACTOR for every fighter. If False, use experience-decayed
              K (elo_calculator.decayed_k_factor) — the parked-branch behavior.
    """
    mode = "FLAT K (current main default)" if flat else "decayed K (parked branch behavior)"
    logger.info(f"Deleting existing EloRating history... [mode: {mode}]")
    deleted = session.query(EloRating).delete()
    session.commit()
    logger.info(f"Deleted {deleted} old EloRating rows")

    fights = (
        session.query(Fight)
        .filter(Fight.winner_id.isnot(None), Fight.fight_date.isnot(None))
        .order_by(Fight.fight_date)
        .all()
    )
    logger.info(f"Replaying {len(fights)} fights chronologically [{mode}]...")

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

        if flat:
            new_a, new_b = update_ratings(
                rating_a, rating_b, winner=winner, method=method, k_factor=ELO_K_FACTOR,
            )
        else:
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
    logger.success(f"Recomputed Elo for {len(fights)} fights ({len(new_rows)} EloRating rows) [{mode}]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recompute EloRating history")
    parser.add_argument("--decayed", action="store_true",
                         help="Use experience-decayed K-factor instead of the current main "
                              "default (flat) — this is the parked-branch behavior, only "
                              "meaningful if FEATURE_COLUMNS also matches that branch's feature set")
    args = parser.parse_args()

    init_db()
    session = get_session()
    recompute_elo(session, flat=not args.decayed)
    session.close()
    if args.decayed:
        logger.warning("Elo history is now DECAYED-K — this does NOT match main's current "
                        "FEATURE_COLUMNS/model config. Run without --decayed to restore the "
                        "flat-K state main actually uses, unless you're deliberately working "
                        "on the parked/accuracy-queue-2026-07-16 branch.")
    else:
        logger.info("Next: rm data/processed/training_dataset.csv && python scripts/train_model.py "
                    "to pick up the new Elo history in training features.")


if __name__ == "__main__":
    main()
