"""
src/features/feature_builder.py
────────────────────────────────
Builds the matchup feature vector for a fight.

KEY RULE: Every feature must use stats that were available BEFORE the fight.
          Enforced by always querying FighterStats where as_of_date < fight_date.

Features are expressed as diffs (fighter_A - fighter_B) so the model
learns matchup dynamics, not just individual fighter quality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from config import FEATURE_COLUMNS, RECENT_FIGHTS_WINDOW, ELO_BASE_RATING


class FeatureBuilder:
    """
    Builds the feature vector for a single fight matchup.

    Usage:
        builder = FeatureBuilder(session)
        features = builder.build_matchup_features(
            fighter_a_id=1,
            fighter_b_id=2,
            as_of_date=datetime(2024, 3, 9)
        )
        X = builder.to_dataframe(features)
    """

    def __init__(self, db_session: Session):
        self.session = db_session

    def build_matchup_features(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: datetime,
    ) -> dict:
        """
        Build the full feature vector for a matchup.
        Returns a dict with all FEATURE_COLUMNS as keys.
        """
        from src.database import Fighter, FighterStats
        from src.features.elo_calculator import EloCalculator

        elo_calc = EloCalculator(self.session)

        # Load physical attributes
        fa = self.session.query(Fighter).get(fighter_a_id)
        fb = self.session.query(Fighter).get(fighter_b_id)

        # Load pre-fight stats snapshots
        stats_a = self._get_stats_before(fighter_a_id, as_of_date)
        stats_b = self._get_stats_before(fighter_b_id, as_of_date)

        # Load Elo ratings before fight
        elo_a = elo_calc.get_rating_before(fighter_a_id, as_of_date)
        elo_b = elo_calc.get_rating_before(fighter_b_id, as_of_date)

        # Average opponent Elo (strength of schedule)
        avg_opp_elo_a = self._avg_opponent_elo(fighter_a_id, as_of_date)
        avg_opp_elo_b = self._avg_opponent_elo(fighter_b_id, as_of_date)

        def s(attr):
            """Safely get attribute from stats object."""
            return getattr(stats_a, attr, None) if stats_a else None

        def t(attr):
            return getattr(stats_b, attr, None) if stats_b else None

        def age_at(fighter):
            if fighter and fighter.date_of_birth and as_of_date:
                return (as_of_date - fighter.date_of_birth).days / 365.25
            return None

        # ── Stance encoding ───────────────────────────────────────────────────
        # Southpaw vs Orthodox is a geometric mismatch — southpaws win at a
        # statistically higher rate vs orthodox due to foot angle + power hand
        # alignment. We encode as two asymmetric flags (not a diff) so the model
        # learns that A-southpaw-vs-B-orthodox is different from A-orthodox-vs-B-southpaw.
        # Switch treated as orthodox (mixed patterns average out).
        def to_stance(fighter):
            if not fighter or not fighter.stance:
                return "orthodox"
            raw = fighter.stance.lower().strip()
            return "southpaw" if "south" in raw else "orthodox"

        stance_a = to_stance(fa)
        stance_b = to_stance(fb)
        # Fighter A has the southpaw geometric edge
        is_southpaw_a_vs_orthodox_b = 1.0 if (stance_a == "southpaw" and stance_b == "orthodox") else 0.0
        # Fighter A is at the southpaw disadvantage (B has the edge)
        is_orthodox_a_vs_southpaw_b = 1.0 if (stance_a == "orthodox" and stance_b == "southpaw") else 0.0

        # ── Short-notice flag ─────────────────────────────────────────────────
        # Fighter on <21 days notice = less camp time, potentially rushed weight cut,
        # often a replacement. Derived from days_since_last_fight — no external data.
        SHORT_NOTICE_DAYS = 21
        days_a = s("days_since_last_fight")
        days_b = t("days_since_last_fight")
        short_notice_a = 1.0 if (days_a is not None and 0 < days_a < SHORT_NOTICE_DAYS) else 0.0
        short_notice_b = 1.0 if (days_b is not None and 0 < days_b < SHORT_NOTICE_DAYS) else 0.0

        features = {
            # Physical
            "reach_diff":  _diff(fa.reach_cm if fa else None,  fb.reach_cm if fb else None),
            "height_diff": _diff(fa.height_cm if fa else None, fb.height_cm if fb else None),
            "age_diff":    _diff(age_at(fa), age_at(fb)),
            # Striking
            "slpm_diff":         _diff(s("slpm"),            t("slpm")),
            "strike_acc_diff":   _diff(s("strike_accuracy"), t("strike_accuracy")),
            "sapm_diff":         _diff(s("sapm"),            t("sapm")),
            "strike_def_diff":   _diff(s("strike_defense"),  t("strike_defense")),
            # Grappling
            "td_avg_diff":  _diff(s("td_avg"),      t("td_avg")),
            "td_acc_diff":  _diff(s("td_accuracy"), t("td_accuracy")),
            "td_def_diff":  _diff(s("td_defense"),  t("td_defense")),
            "sub_avg_diff": _diff(s("sub_avg"),     t("sub_avg")),
            # Record / form
            "win_rate_diff":              _diff(s("win_rate"),          t("win_rate")),
            "finish_rate_diff":           _diff(s("finish_rate"),       t("finish_rate")),
            "recent_win_rate_diff":       _diff(s("recent_win_rate"),   t("recent_win_rate")),
            "days_since_last_fight_diff": _diff(s("days_since_last_fight"), t("days_since_last_fight")),
            "win_streak_diff":            _diff(s("win_streak"),        t("win_streak")),
            # Elo
            "elo_diff":             _diff(elo_a,         elo_b),
            "avg_opponent_elo_diff": _diff(avg_opp_elo_a, avg_opp_elo_b),
            # Style matchup features — continuous scores, not labels
            # Model learns degree of mismatch (e.g. pure wrestler vs pure striker)
            "style_pressure_diff":    _diff(s("style_pressure"),    t("style_pressure")),
            "style_wrestling_diff":   _diff(s("style_wrestling"),   t("style_wrestling")),
            "style_striker_diff":     _diff(s("style_striker"),     t("style_striker")),
            "style_finisher_diff":    _diff(s("style_finisher"),    t("style_finisher")),
            "grappling_defense_diff": _diff(s("grappling_defense"), t("grappling_defense")),
            # Recent form — recency-weighted, combats favorite bias
            "momentum_score_diff":      _diff(s("momentum_score"),      t("momentum_score")),
            "recent_finish_rate_diff":  _diff(s("recent_finish_rate"),  t("recent_finish_rate")),
            # Weight class percentiles — same stat means different things at HW vs FW
            "slpm_pctile_diff":   _diff(s("slpm_pctile"),   t("slpm_pctile")),
            "td_avg_pctile_diff": _diff(s("td_avg_pctile"), t("td_avg_pctile")),
            # UFC experience — debut vs veteran dynamic
            # A 10-fight UFC vet is very different from a debuting prospect
            "ufc_fights_diff": _diff(s("ufc_fights"), t("ufc_fights")),
            "ufc_wins_diff":   _diff(s("ufc_wins"),   t("ufc_wins")),
            # Fight context — title fights have different dynamics
            # Filled in by training loop; default 0 for live predictions
            "is_title_fight": 0.0,
            # Stance mismatch — southpaw geometric advantage (asymmetric, not a diff)
            "is_southpaw_a_vs_orthodox_b": is_southpaw_a_vs_orthodox_b,
            "is_orthodox_a_vs_southpaw_b": is_orthodox_a_vs_southpaw_b,
            # Short notice — derived from days_since_last_fight (<21 days)
            "fighter_a_short_notice": short_notice_a,
            "fighter_b_short_notice": short_notice_b,
            # Injury flags — not yet automated, remain 0 until news scraper added
            "sentiment_diff":        0.0,
            "fighter_a_injury_flag": 0.0,
            "fighter_b_injury_flag": 0.0,
        }

        missing = [col for col in FEATURE_COLUMNS if col not in features]
        if missing:
            logger.warning(f"Missing features for fight {fighter_a_id} vs {fighter_b_id}: {missing}")

        return features

    def _get_stats_before(self, fighter_id: int, as_of_date: datetime):
        """
        Get the most recent FighterStats snapshot before a given date.
        Returns None if no snapshot exists yet (new fighter).
        """
        from src.database import FighterStats
        return (
            self.session.query(FighterStats)
            .filter(FighterStats.fighter_id == fighter_id)
            .filter(FighterStats.as_of_date < as_of_date)
            .order_by(FighterStats.as_of_date.desc())
            .first()
        )

    def _avg_opponent_elo(self, fighter_id: int, as_of_date: datetime) -> float:
        """
        Calculate average Elo of opponents faced before a given date.
        Measures strength of schedule.
        """
        from src.database import Fight, EloRating
        from src.features.elo_calculator import EloCalculator

        elo_calc = EloCalculator(self.session)

        fights = (
            self.session.query(Fight)
            .filter(
                ((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id)),
                Fight.fight_date < as_of_date,
                Fight.winner_id.isnot(None),
            )
            .all()
        )

        if not fights:
            return ELO_BASE_RATING

        opp_elos = []
        for fight in fights:
            opp_id = fight.fighter_b_id if fight.fighter_a_id == fighter_id else fight.fighter_a_id
            opp_elo = elo_calc.get_rating_before(opp_id, fight.fight_date)
            opp_elos.append(opp_elo)

        return sum(opp_elos) / len(opp_elos)

    def to_dataframe(self, features: dict) -> pd.DataFrame:
        """Convert feature dict to single-row DataFrame in correct column order."""
        return pd.DataFrame([features])[FEATURE_COLUMNS]

    def to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array in correct column order."""
        return np.array([features.get(col, 0.0) for col in FEATURE_COLUMNS]).reshape(1, -1)


def build_training_dataset(session: Session) -> pd.DataFrame:
    """
    Build the full historical training dataset from the database.

    Iterates every completed fight in chronological order and builds
    the pre-fight feature vector + labels for each one.

    Returns a DataFrame ready for model training.
    This is the function you run once after loading historical data.
    """
    from src.database import Fight, Fighter

    logger.info("Building training dataset from historical fights...")

    fights = (
        session.query(Fight)
        .filter(Fight.winner_id.isnot(None))   # only completed fights
        .filter(Fight.method.isnot(None))
        .order_by(Fight.fight_date)
        .all()
    )

    logger.info(f"Processing {len(fights)} completed fights...")

    builder = FeatureBuilder(session)
    rows = []
    skipped = 0

    for fight in fights:
        try:
            features = builder.build_matchup_features(
                fighter_a_id=fight.fighter_a_id,
                fighter_b_id=fight.fighter_b_id,
                as_of_date=fight.fight_date,
            )

            # Fill in fight-level context features
            features["is_title_fight"] = 1.0 if fight.is_title_fight else 0.0

            # Labels
            winner = 1 if fight.winner_id == fight.fighter_a_id else 0
            method = fight.method or "Decision"
            finish_round = fight.finish_round

            row = {
                **features,
                "fight_id":    fight.id,
                "fight_date":  fight.fight_date,
                "weight_class": fight.weight_class,
                "winner":      winner,
                "method":      method,
                "finish_round": finish_round,
            }
            rows.append(row)

        except Exception as e:
            logger.debug(f"Skipped fight {fight.id}: {e}")
            skipped += 1

    df = pd.DataFrame(rows)
    logger.success(f"Dataset built: {len(df)} fights, {skipped} skipped")
    logger.info(f"Date range: {df.fight_date.min()} → {df.fight_date.max()}")
    logger.info(f"Fighter A win rate: {df.winner.mean():.1%} (should be ~50%)")

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _diff(a, b) -> float:
    """Return a - b. Returns 0.0 if either is None (handled by model as neutral)."""
    if a is None or b is None:
        return 0.0
    return float(a) - float(b)
