"""
src/features/elo_calculator.py
───────────────────────────────
Elo rating system for UFC fighters.

Two modes:
  - Normal (live predictions): queries DB per fighter per date
  - Cached (dataset building): preloads ALL elo history into memory once,
    then serves all lookups from in-memory dicts. 100x faster for bulk.
"""

import math
from datetime import datetime
from collections import defaultdict
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
        self._cache = None  # populated by preload()

    def preload(self):
        """
        Load ALL elo history into memory once.
        Call this before bulk dataset building — makes all lookups O(log n)
        instead of O(n) DB queries.

        After calling preload(), all get_* methods use in-memory data.
        """
        from src.database import EloRating, Fight

        # Load all elo ratings with fight dates
        rows = (
            self.session.query(EloRating, Fight.fight_date)
            .join(Fight, EloRating.after_fight_id == Fight.id)
            .order_by(EloRating.fighter_id, Fight.fight_date)
            .all()
        )

        # Also load fight counts per fighter (for uncertainty)
        from src.database import Fighter
        from sqlalchemy import func, or_

        fight_counts_raw = (
            self.session.query(
                Fight.fighter_a_id.label("fid"),
                Fight.fight_date,
            ).filter(Fight.winner_id.isnot(None))
            .union_all(
                self.session.query(
                    Fight.fighter_b_id.label("fid"),
                    Fight.fight_date,
                ).filter(Fight.winner_id.isnot(None))
            )
            .all()
        )

        # Build fighter_id → sorted list of (fight_date, rating)
        elo_history = defaultdict(list)
        for elo, fight_date in rows:
            elo_history[elo.fighter_id].append((fight_date, elo.rating))

        # Build fighter_id → sorted list of fight_dates
        fight_dates = defaultdict(list)
        for fid, fdate in fight_counts_raw:
            if fdate:
                fight_dates[fid].append(fdate)
        for fid in fight_dates:
            fight_dates[fid].sort()

        self._cache = {
            "elo_history": dict(elo_history),
            "fight_dates": dict(fight_dates),
        }

    def _is_cached(self):
        return self._cache is not None

    def get_rating_before(self, fighter_id: int, as_of_date: datetime) -> float:
        if self._is_cached():
            history = self._cache["elo_history"].get(fighter_id, [])
            # Find most recent rating before as_of_date
            rating = ELO_BASE_RATING
            for fight_date, r in history:
                if fight_date < as_of_date:
                    rating = r
                else:
                    break
            return rating

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

    def get_elo_trend(self, fighter_id: int, as_of_date: datetime, n_fights: int = 3) -> float:
        """Elo change over last N fights. Positive = rising, negative = declining."""
        if self._is_cached():
            history = self._cache["elo_history"].get(fighter_id, [])
            prior = [(d, r) for d, r in history if d < as_of_date]
            if len(prior) < 2:
                return 0.0
            window = prior[-min(n_fights + 1, len(prior)):]
            return round(window[-1][1] - window[0][1], 2)

        from src.database import EloRating, Fight
        rows = (
            self.session.query(EloRating)
            .join(Fight, EloRating.after_fight_id == Fight.id)
            .filter(EloRating.fighter_id == fighter_id)
            .filter(Fight.fight_date < as_of_date)
            .order_by(Fight.fight_date.desc())
            .limit(n_fights + 1)
            .all()
        )
        if len(rows) < 2:
            return 0.0
        return round(rows[0].rating - rows[-1].rating, 2)

    def get_elo_uncertainty(self, fighter_id: int, as_of_date: datetime) -> float:
        """1 / (1 + sqrt(n_fights)) — high = debut, low = veteran."""
        if self._is_cached():
            dates = self._cache["fight_dates"].get(fighter_id, [])
            n = sum(1 for d in dates if d < as_of_date)
            return round(1.0 / (1.0 + math.sqrt(n)), 4)

        from src.database import Fight
        n = (
            self.session.query(Fight)
            .filter(
                ((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id)),
                Fight.fight_date < as_of_date,
                Fight.winner_id.isnot(None)
            )
            .count()
        )
        return round(1.0 / (1.0 + math.sqrt(n)), 4)

    def get_career_peak_elo(self, fighter_id: int, as_of_date: datetime) -> float:
        """Highest Elo reached before this date."""
        if self._is_cached():
            history = self._cache["elo_history"].get(fighter_id, [])
            prior = [r for d, r in history if d < as_of_date]
            return max(prior) if prior else ELO_BASE_RATING

        from src.database import EloRating, Fight
        rows = (
            self.session.query(EloRating)
            .join(Fight, EloRating.after_fight_id == Fight.id)
            .filter(EloRating.fighter_id == fighter_id)
            .filter(Fight.fight_date < as_of_date)
            .all()
        )
        if not rows:
            return ELO_BASE_RATING
        return max(r.rating for r in rows)

    def get_rating(self, fighter_id: int) -> float:
        from src.database import EloRating
        row = (
            self.session.query(EloRating)
            .filter_by(fighter_id=fighter_id)
            .order_by(EloRating.recorded_at.desc())
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



def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
