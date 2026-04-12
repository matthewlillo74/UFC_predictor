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
        self._fighters_cache = None       # fighter_id → Fighter
        self._stats_cache = None          # fighter_id → sorted [(as_of_date, FighterStats)]
        self._opp_elo_cache = None        # fighter_id → list of (fight_date, opp_elo)
        self._fight_history = None        # fighter_id → sorted [(fight_date, opp_id)]
        self._elo_calc = None             # shared EloCalculator (preloaded for bulk use)

    def preload(self):
        """
        Load all fighters, stats snapshots, and Elo history into memory.
        Call once before building a large training dataset — avoids thousands
        of individual DB queries and speeds up dataset building by ~50-100x.
        """
        from src.database import Fighter, FighterStats, EloRating, Fight
        from src.features.elo_calculator import EloCalculator
        from collections import defaultdict

        logger.info("Preloading feature cache...")

        # All fighters
        self._fighters_cache = {f.id: f for f in self.session.query(Fighter).all()}

        # All FighterStats — sorted by as_of_date per fighter
        stats_raw = self.session.query(FighterStats).order_by(
            FighterStats.fighter_id, FighterStats.as_of_date
        ).all()
        stats_by_fighter = defaultdict(list)
        for s in stats_raw:
            stats_by_fighter[s.fighter_id].append(s)
        self._stats_cache = dict(stats_by_fighter)

        # Opponent Elo per fight per fighter (for avg_opponent_elo)
        # Build: fighter_id → [(fight_date, opp_elo)]
        elo_by_fighter = defaultdict(list)
        elo_rows = (
            self.session.query(EloRating, Fight.fight_date, Fight.fighter_a_id, Fight.fighter_b_id)
            .join(Fight, EloRating.after_fight_id == Fight.id)
            .filter(Fight.fight_date.isnot(None))
            .all()
        )
        # Build fight_id → (date, fighter_a, fighter_b) and elo snapshots
        elo_by_fight = defaultdict(list)  # fight_id → [(fighter_id, rating)]
        for elo, fight_date, fa_id, fb_id in elo_rows:
            elo_by_fight[(fight_date, fa_id, fb_id)].append((elo.fighter_id, elo.rating, fight_date))

        # For avg_opponent_elo: fighter_id → [(fight_date, opp_rating_before_fight)]
        # We approximate opp rating as the pre-fight snapshot = their current elo at fight time
        # This is precomputed in the EloCalculator preload, so we delegate to it
        self._opp_elo_cache = {}  # will use elo_calc.preload() instead

        # Shared EloCalculator with full preload
        self._elo_calc = EloCalculator(self.session)
        self._elo_calc.preload()

        # Fight history per fighter: fighter_id → [(fight_date, opponent_id)]
        # Used for avg_opponent_elo without hitting DB per fight
        from src.database import Fight as FightModel
        all_fights = (
            self.session.query(FightModel)
            .filter(
                FightModel.fight_date.isnot(None),
                FightModel.winner_id.isnot(None),
            )
            .all()
        )
        fight_history = defaultdict(list)
        for f in all_fights:
            fight_history[f.fighter_a_id].append((f.fight_date, f.fighter_b_id))
            fight_history[f.fighter_b_id].append((f.fight_date, f.fighter_a_id))
        # Sort by date
        for fid in fight_history:
            fight_history[fid].sort(key=lambda x: x[0])
        self._fight_history = dict(fight_history)

        logger.info(f"Cache loaded: {len(self._fighters_cache)} fighters, "
                    f"{sum(len(v) for v in self._stats_cache.values())} stat snapshots, "
                    f"{len(all_fights)} fights")

    def _get_fighter(self, fighter_id: int):
        """Get fighter from cache or DB."""
        if self._fighters_cache is not None:
            return self._fighters_cache.get(fighter_id)
        return self.session.query(__import__('src.database', fromlist=['Fighter']).Fighter).get(fighter_id)

    def _get_stats_cached(self, fighter_id: int, as_of_date: datetime):
        """Get most recent FighterStats before as_of_date from cache."""
        if self._stats_cache is not None:
            snapshots = self._stats_cache.get(fighter_id, [])
            result = None
            for snap in snapshots:
                if snap.as_of_date and snap.as_of_date < as_of_date:
                    result = snap
                else:
                    break
            return result
        return self._get_stats_before(fighter_id, as_of_date)

    def _avg_opp_elo_cached(self, fighter_id: int, as_of_date: datetime) -> float:
        """Compute avg opponent Elo — fully in-memory when cache is loaded."""
        if self._fight_history is not None and self._elo_calc and self._elo_calc._is_cached():
            prior_fights = [
                (fd, opp_id)
                for fd, opp_id in self._fight_history.get(fighter_id, [])
                if fd < as_of_date
            ]
            if not prior_fights:
                return ELO_BASE_RATING
            opp_elos = [
                self._elo_calc.get_rating_before(opp_id, fight_date)
                for fight_date, opp_id in prior_fights
            ]
            return sum(opp_elos) / len(opp_elos)
        return self._avg_opponent_elo(fighter_id, as_of_date)

    def build_matchup_features(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: datetime,
    ) -> dict:
        """
        Build the full feature vector for a matchup.
        Returns a dict with all FEATURE_COLUMNS as keys.
        Uses preloaded cache if available (fast), otherwise queries DB (live predictions).
        """
        from src.database import Fighter, FighterStats
        from src.features.elo_calculator import EloCalculator

        # Use shared preloaded calculator if available, else create per-call
        elo_calc = self._elo_calc if self._elo_calc else EloCalculator(self.session)

        # Load physical attributes (from cache if available)
        fa = self._get_fighter(fighter_a_id) if self._fighters_cache else self.session.query(Fighter).get(fighter_a_id)
        fb = self._get_fighter(fighter_b_id) if self._fighters_cache else self.session.query(Fighter).get(fighter_b_id)

        # Load pre-fight stats snapshots (from cache if available)
        stats_a = self._get_stats_cached(fighter_a_id, as_of_date)
        stats_b = self._get_stats_cached(fighter_b_id, as_of_date)

        # Load Elo ratings before fight
        elo_a = elo_calc.get_rating_before(fighter_a_id, as_of_date)
        elo_b = elo_calc.get_rating_before(fighter_b_id, as_of_date)

        # Elo dynamics — trend, uncertainty, peak
        elo_trend_a    = elo_calc.get_elo_trend(fighter_a_id, as_of_date)
        elo_trend_b    = elo_calc.get_elo_trend(fighter_b_id, as_of_date)
        elo_uncert_a   = elo_calc.get_elo_uncertainty(fighter_a_id, as_of_date)
        elo_uncert_b   = elo_calc.get_elo_uncertainty(fighter_b_id, as_of_date)
        elo_peak_a     = elo_calc.get_career_peak_elo(fighter_a_id, as_of_date)
        elo_peak_b     = elo_calc.get_career_peak_elo(fighter_b_id, as_of_date)
        # Distance from peak — negative means fighter is below their best form
        elo_vs_peak_a  = elo_a - elo_peak_a
        elo_vs_peak_b  = elo_b - elo_peak_b

        # Average opponent Elo (strength of schedule)
        avg_opp_elo_a = self._avg_opp_elo_cached(fighter_a_id, as_of_date)
        avg_opp_elo_b = self._avg_opp_elo_cached(fighter_b_id, as_of_date)

        def s(attr):
            """Safely get attribute from stats object."""
            return getattr(stats_a, attr, None) if stats_a else None

        def t(attr):
            return getattr(stats_b, attr, None) if stats_b else None

        def age_at(fighter):
            if fighter and fighter.date_of_birth and as_of_date:
                return (as_of_date - fighter.date_of_birth).days / 365.25
            return None

        # ── Age curve by weight class ──────────────────────────────────────────
        # Raw age_diff treats a 30-year-old flyweight and heavyweight identically.
        # In reality, smaller fighters peak and decline earlier.
        # age_vs_peak_diff = (A's distance from their class peak) - (B's distance)
        # Negative = A is closer to their prime than B = advantage for A.
        WEIGHT_CLASS_PEAK = {
            "Strawweight": 28, "Flyweight": 29, "Bantamweight": 30,
            "Featherweight": 30, "Lightweight": 31, "Welterweight": 32,
            "Middleweight": 33, "Light Heavyweight": 33, "Heavyweight": 34,
            "Women's Strawweight": 27, "Women's Flyweight": 28,
            "Women's Bantamweight": 29, "Women's Featherweight": 30,
        }
        def age_vs_peak(fighter):
            age = age_at(fighter)
            if age is None:
                return None
            wc = (fighter.weight_class or "").strip()
            peak = WEIGHT_CLASS_PEAK.get(wc, 31)  # default 31 if unknown
            return age - peak  # 0 = at peak, positive = past peak, negative = pre-peak

        # ── Durability / damage accumulation ──────────────────────────────────
        # Fighters degrade from absorbed damage over their career.
        # We use SAPM (strikes absorbed per minute) as a career damage proxy,
        # weighted toward recent fights via recent_finish_rate (how often they
        # get finished). High SAPM + high recent finish rate = durability concern.
        def durability_score(stats_obj):
            if not stats_obj:
                return 0.0
            sapm = getattr(stats_obj, "sapm", None) or 0.0
            recent_finish = getattr(stats_obj, "recent_finish_rate", None) or 0.0
            ko_losses = getattr(stats_obj, "losses_ko_tko", None) or 0
            losses = max(getattr(stats_obj, "losses", None) or 1, 1)
            ko_loss_rate = ko_losses / losses
            # Composite: high sapm, high recent finishes, high KO loss rate = worse durability
            return round(sapm * 0.4 + recent_finish * 0.3 + ko_loss_rate * 0.3, 4)

        durability_a = durability_score(stats_a)
        durability_b = durability_score(stats_b)

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
            # Age curve by weight class — distance from divisional prime
            # 0 = at peak, positive = past peak (declining), negative = pre-peak
            "age_vs_peak_diff": _diff(age_vs_peak(fa), age_vs_peak(fb)),
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
            "elo_diff":              _diff(elo_a,           elo_b),
            "avg_opponent_elo_diff": _diff(avg_opp_elo_a,   avg_opp_elo_b),
            # Elo dynamics — trend, uncertainty, peak distance
            # These capture trajectory not captured by a single rating snapshot
            "elo_trend_diff":      _diff(elo_trend_a,   elo_trend_b),    # rising vs falling
            "elo_uncertainty_diff": _diff(elo_uncert_a, elo_uncert_b),   # debut vs veteran
            "elo_vs_peak_diff":    _diff(elo_vs_peak_a, elo_vs_peak_b),  # decline from prime
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
            # Durability — composite proxy (SAPM + KO loss rate)
            "durability_diff": _diff(durability_a, durability_b),
            # Knockdown durability — real measured data from fight detail pages
            # kd_absorbed_per_fight: how often does this fighter get knocked down?
            # kd_ratio: does this fighter knock people down more than they get knocked down?
            "kd_absorbed_per_fight_diff": _diff(s("kd_absorbed_per_fight"), t("kd_absorbed_per_fight")),
            "kd_ratio_diff":              _diff(s("kd_ratio"),              t("kd_ratio")),
            # Opponent style vulnerability matchup features
            #   how well does A do vs the kind of fighter B actually is?
            # Negative diff = A is MORE vulnerable to B's style than vice versa
            "style_vuln_wrestling_diff": _diff(
                _safe(s("winrate_vs_wrestlers")) * _safe(t("style_wrestling")),
                _safe(t("winrate_vs_wrestlers")) * _safe(s("style_wrestling")),
            ),
            "style_vuln_striker_diff": _diff(
                _safe(s("winrate_vs_strikers")) * _safe(t("style_striker")),
                _safe(t("winrate_vs_strikers")) * _safe(s("style_striker")),
            ),
            "style_vuln_pressure_diff": _diff(
                _safe(s("winrate_vs_pressure")) * _safe(t("style_pressure")),
                _safe(t("winrate_vs_pressure")) * _safe(s("style_pressure")),
            ),
            # ── Interaction features ───────────────────────────────────────────
            # XGBoost can learn interactions but needs many fights to do so reliably.
            # With ~8k fights, explicit domain-knowledge interactions add real signal.
            #
            # Takedown success probability — wrestler vs fighter who can stuff TDs
            "td_success_prob_diff": _diff(
                _safe(s("td_avg")) * (1 - _safe(t("td_defense"))),
                _safe(t("td_avg")) * (1 - _safe(s("td_defense"))),
            ),
            # Striking exchange edge — net effective striking (output × accuracy - absorbed)
            "striking_edge_diff": _diff(
                _safe(s("slpm")) * _safe(s("strike_accuracy")) - _safe(s("sapm")),
                _safe(t("slpm")) * _safe(t("strike_accuracy")) - _safe(t("sapm")),
            ),
            # Grappling dominance — td_avg × sub_avg × (1 - opp_td_def)
            "grapple_dom_diff": _diff(
                _safe(s("td_avg")) * _safe(s("sub_avg")) * (1 - _safe(t("td_defense"))),
                _safe(t("td_avg")) * _safe(t("sub_avg")) * (1 - _safe(s("td_defense"))),
            ),
            # Finish threat vs durability — finish_rate × (1 - opp_strike_defense)
            "finish_threat_diff": _diff(
                _safe(s("finish_rate")) * (1 - _safe(t("strike_defense"))),
                _safe(t("finish_rate")) * (1 - _safe(s("strike_defense"))),
            ),
            # Reach × striking accuracy — reach only matters if you use it
            "reach_strike_diff": _diff(
                _safe(fa.reach_cm if fa else None) * _safe(s("strike_accuracy")),
                _safe(fb.reach_cm if fb else None) * _safe(t("strike_accuracy")),
            ),
            # Cardio decay — from round-level data
            "cardio_decay_diff":       _diff(s("cardio_decay"),       t("cardio_decay")),
            "early_output_share_diff": _diff(s("early_output_share"), t("early_output_share")),
            # Strike location rates
            "head_strike_rate_diff":    _diff(s("head_strike_rate"),    t("head_strike_rate")),
            "leg_strike_rate_diff":     _diff(s("leg_strike_rate"),     t("leg_strike_rate")),
            "ground_strike_share_diff": _diff(s("ground_strike_share"), t("ground_strike_share")),
            # Rolling style windows — career avg vs recent trend
            "style_pressure_l3_diff":  _diff(s("style_pressure_l3"),  t("style_pressure_l3")),
            "style_wrestling_l3_diff": _diff(s("style_wrestling_l3"), t("style_wrestling_l3")),
            "style_striker_l3_diff":   _diff(s("style_striker_l3"),   t("style_striker_l3")),
            "style_pressure_l5_diff":  _diff(s("style_pressure_l5"),  t("style_pressure_l5")),
            "style_wrestling_l5_diff": _diff(s("style_wrestling_l5"), t("style_wrestling_l5")),
            "style_striker_l5_diff":   _diff(s("style_striker_l5"),   t("style_striker_l5")),
            # Injury flags — not yet automated, remain 0 until news scraper added
            "sentiment_diff":        0.0,
            "fighter_a_injury_flag": 0.0,
            "fighter_b_injury_flag": 0.0,
            # ── Method-specific rates ─────────────────────────────────────────────
            # finish_rate_diff only captures total finishing ability. These separate
            # HOW fighters win and lose — critical for method/prop prediction.
            # A submission specialist vs a KO artist is fundamentally different from
            # two generic finishers. These features directly feed the method model.
            "ko_rate_diff":          _diff(
                _safe(s("wins_ko_tko")) / max(_safe(s("wins")) or 1, 1),
                _safe(t("wins_ko_tko")) / max(_safe(t("wins")) or 1, 1),
            ),
            "sub_rate_diff":         _diff(
                _safe(s("wins_sub")) / max(_safe(s("wins")) or 1, 1),
                _safe(t("wins_sub")) / max(_safe(t("wins")) or 1, 1),
            ),
            "decision_rate_diff":    _diff(
                _safe(s("wins_decision")) / max(_safe(s("wins")) or 1, 1),
                _safe(t("wins_decision")) / max(_safe(t("wins")) or 1, 1),
            ),
            # How often does a fighter get KO'd when they lose?
            # High value = fragile chin = opponent has KO path even if they lose overall
            "ko_vulnerability_diff": _diff(
                _safe(s("losses_ko_tko")) / max(_safe(s("losses")) or 1, 1),
                _safe(t("losses_ko_tko")) / max(_safe(t("losses")) or 1, 1),
            ),
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
    # Preload all data into memory — avoids ~100k individual DB queries
    # Brings dataset build time from hours down to ~10 minutes
    builder.preload()
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


def _safe(x) -> float:
    """Convert None to 0.0 for use in multiplication. Never raises."""
    return float(x) if x is not None else 0.0
