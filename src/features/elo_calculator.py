"""
src/features/elo_calculator.py
───────────────────────────────
Elo rating system for UFC fighters.
"""

import math
from datetime import datetime
from config import ELO_BASE_RATING, ELO_K_FACTOR, ELO_FINISH_BONUS


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_ratings(
    rating_a: float,
    rating_b: float,
    winner: str,
    method: str = "decision",
    k_factor: float = ELO_K_FACTOR,
) -> tuple:
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a

    if winner == "a":
        score_a, score_b = 1.0, 0.0
    elif winner == "b":
        score_a, score_b = 0.0, 1.0
    else:
        score_a, score_b = 0.5, 0.5

    finish_multiplier = 1.0
    if method in ("ko_tko", "submission") and winner in ("a", "b"):
        finish_multiplier = 1.0 + ELO_FINISH_BONUS

    new_a = rating_a + k_factor * finish_multiplier * (score_a - exp_a)
    new_b = rating_b + k_factor * finish_multiplier * (score_b - exp_b)
    return round(new_a, 2), round(new_b, 2)


class EloCalculator:
    def __init__(self, db_session):
        self.session = db_session

    def get_rating(self, fighter_id: int) -> float:
        from src.database import EloRating
        row = (
            self.session.query(EloRating)
            .filter_by(fighter_id=fighter_id)
            .order_by(EloRating.recorded_at.desc())
            .first()
        )
        return row.rating if row else ELO_BASE_RATING

    def get_rating_before(self, fighter_id: int, as_of_date: datetime) -> float:
        from src.database import EloRating, Fight
        row = (
            self.session.query(EloRating)
            .join(Fight, EloRating.after_fight_id == Fight.id)
            .filter(EloRating.fighter_id == fighter_id)
            .filter(Fight.fight_date < as_of_date)
            .order_by(Fight.fight_date.desc())
            .first()
        )
        return row.rating if row else ELO_BASE_RATING

    def get_leaderboard(self, weight_class: str = None, top_n: int = 20) -> list:
        from src.database import EloRating, Fighter
        from sqlalchemy import func

        subq = (
            self.session.query(
                EloRating.fighter_id,
                func.max(EloRating.recorded_at).label("latest")
            )
            .group_by(EloRating.fighter_id)
            .subquery()
        )

        query = (
            self.session.query(Fighter, EloRating)
            .join(EloRating, Fighter.id == EloRating.fighter_id)
            .join(subq, (EloRating.fighter_id == subq.c.fighter_id) &
                        (EloRating.recorded_at == subq.c.latest))
        )

        if weight_class:
            query = query.filter(Fighter.weight_class == weight_class)

        results = query.order_by(EloRating.rating.desc()).limit(top_n).all()
        return [
            {"rank": i + 1, "name": f.name, "weight_class": f.weight_class, "elo": round(e.rating, 1)}
            for i, (f, e) in enumerate(results)
        ]
