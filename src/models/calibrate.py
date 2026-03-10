"""
src/models/calibrate.py
────────────────────────
Probability calibration for the UFC prediction model.

Raw XGBoost probabilities are often overconfident — a 70% prediction
might actually win only 62% of the time. Calibration fixes this.

We use isotonic regression calibration (better than Platt scaling for
non-linear probability distributions like MMA outcomes).

Also contains the underdog detection logic — identifies fights where
the model meaningfully disagrees with the market.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from config import MODELS_DIR


class ProbabilityCalibrator:
    """
    Wraps the XGBoost winner model with isotonic regression calibration.

    After calibration:
    - A 60% prediction should win ~60% of the time
    - Underdog probabilities are less suppressed
    - Value bets are more reliably identified
    """

    def __init__(self):
        self.calibrator = None
        self._is_fitted = False

    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray):
        """
        Fit the calibrator on held-out validation data.

        Args:
            raw_probs:       Raw model probabilities (n_samples,)
            actual_outcomes: Actual results 0/1 (n_samples,)
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, actual_outcomes)
        self._is_fitted = True
        logger.success(f"Calibrator fitted on {len(raw_probs)} samples")

    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration to a single probability."""
        if not self._is_fitted:
            return raw_prob
        return float(self.calibrator.predict([raw_prob])[0])

    def calibrate_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to an array of probabilities."""
        if not self._is_fitted:
            return raw_probs
        return self.calibrator.predict(raw_probs)

    def save(self):
        path = MODELS_DIR / "calibrator.pkl"
        joblib.dump(self.calibrator, path)
        logger.success(f"Calibrator saved to {path}")

    def load(self):
        path = MODELS_DIR / "calibrator.pkl"
        if path.exists():
            self.calibrator = joblib.load(path)
            self._is_fitted = True
            logger.success("Calibrator loaded")
        else:
            logger.warning("No calibrator found — using raw probabilities")

    def calibration_report(self, raw_probs: np.ndarray, actuals: np.ndarray) -> pd.DataFrame:
        """
        Show how well calibrated the model is.
        Good calibration: predicted prob ≈ actual win rate in each bucket.
        """
        calibrated = self.calibrate_batch(raw_probs)

        rows = []
        for label, probs in [("Raw", raw_probs), ("Calibrated", calibrated)]:
            fraction_pos, mean_pred = calibration_curve(actuals, probs, n_bins=8)
            for fp, mp in zip(fraction_pos, mean_pred):
                rows.append({"Type": label, "Predicted": mp, "Actual": fp})

        return pd.DataFrame(rows)


# ── Underdog & Value Detection ────────────────────────────────────────────────

def find_value_bets(predictions: list[dict], min_edge: float = 0.06) -> list[dict]:
    """
    Given a list of fight predictions with odds, find value bets.

    A value bet = model probability significantly exceeds market probability.
    The higher the edge, the stronger the value signal.

    Returns sorted list of value bets, best edge first.
    """
    value_bets = []

    for pred in predictions:
        odds = pred.get("odds_data")
        if not odds:
            continue

        prob_a = pred["prob_fighter_a"]
        prob_b = pred["prob_fighter_b"]
        fair_a = odds.get("fair_prob_a", 0.5)
        fair_b = odds.get("fair_prob_b", 0.5)

        edge_a = prob_a - fair_a
        edge_b = prob_b - fair_b

        for fighter, model_prob, market_prob, edge, odds_val in [
            (pred["fighter_a"], prob_a, fair_a, edge_a, odds.get("odds_a")),
            (pred["fighter_b"], prob_b, fair_b, edge_b, odds.get("odds_b")),
        ]:
            if edge >= min_edge:
                kelly = _kelly_fraction(model_prob, odds_val)
                value_bets.append({
                    "fighter":      fighter,
                    "opponent":     pred["fighter_b"] if fighter == pred["fighter_a"] else pred["fighter_a"],
                    "model_prob":   round(model_prob, 4),
                    "market_prob":  round(market_prob, 4),
                    "edge":         round(edge, 4),
                    "odds":         odds_val,
                    "kelly_pct":    round(kelly * 100, 1),
                    "is_underdog":  model_prob < 0.5,
                    "weight_class": pred.get("weight_class", ""),
                })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def find_upset_candidates(predictions: list[dict], min_upset_score: float = 0.10) -> list[dict]:
    """
    Find fights where the model likes the underdog significantly more than the market.

    These are the most interesting bets — market has an underdog at +250 but the
    model thinks they win 45% of the time instead of the implied 28%.
    """
    upsets = []

    for pred in predictions:
        odds = pred.get("odds_data")
        prob_a = pred["prob_fighter_a"]
        prob_b = pred["prob_fighter_b"]

        if odds:
            fair_a = odds.get("fair_prob_a", 0.5)
            fair_b = 1 - fair_a
            # Underdog is whoever the market has below 50%
            if fair_a < 0.5:
                upset_score = prob_a - fair_a
                underdog = pred["fighter_a"]
                underdog_model_prob = prob_a
                underdog_market_prob = fair_a
                underdog_odds = odds.get("odds_a")
            else:
                upset_score = prob_b - fair_b
                underdog = pred["fighter_b"]
                underdog_model_prob = prob_b
                underdog_market_prob = fair_b
                underdog_odds = odds.get("odds_b")

            if upset_score >= min_upset_score:
                upsets.append({
                    "underdog":      underdog,
                    "favorite":      pred["fighter_b"] if underdog == pred["fighter_a"] else pred["fighter_a"],
                    "upset_score":   round(upset_score, 4),
                    "model_prob":    round(underdog_model_prob, 4),
                    "market_prob":   round(underdog_market_prob, 4),
                    "underdog_odds": underdog_odds,
                    "weight_class":  pred.get("weight_class", ""),
                })

    upsets.sort(key=lambda x: x["upset_score"], reverse=True)
    return upsets


def _kelly_fraction(win_prob: float, american_odds: int) -> float:
    """
    Kelly Criterion: optimal bet sizing given edge.
    Returns fraction of bankroll to bet.
    Cap at 10% — never go full Kelly in practice.
    """
    if not american_odds or win_prob <= 0:
        return 0.0
    if american_odds > 0:
        b = american_odds / 100
    else:
        b = 100 / abs(american_odds)

    q = 1 - win_prob
    kelly = (b * win_prob - q) / b
    return max(0.0, min(kelly, 0.10))  # cap at 10%
