"""
src/betting/value_detector.py
──────────────────────────────
Compares model probabilities to sportsbook odds to find value bets.

A value bet exists when the model believes a fighter's true win probability
is meaningfully higher than what the sportsbook is implying.

Example:
    Sportsbook: Fighter B +200  (implied prob: 33%)
    Model:      Fighter B 48%
    Edge:       +15%  ← value bet

Also detects:
  - Upset candidates (underdog model prob >> market prob)
  - Line movement (opening vs closing odds shift)
  - Reverse line movement (sharp money signal)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import Optional
from loguru import logger


# ── Odds Math ─────────────────────────────────────────────────────────────────

def american_to_prob(american_odds: int) -> float:
    """
    Convert American moneyline odds to implied win probability.

    Examples:
        -200  →  0.667  (heavy favorite)
        +150  →  0.400  (underdog)
        -110  →  0.524  (slight favorite / juice)
    """
    if american_odds < 0:
        return (-american_odds) / (-american_odds + 100)
    else:
        return 100 / (american_odds + 100)


def prob_to_american(prob: float) -> int:
    """Convert win probability back to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove the sportsbook's vig (juice) from implied probabilities.
    Raw implied probs sum to >1.0 because of the house edge.
    This normalizes them to sum to exactly 1.0.

    Example:
        Raw:        A=0.556, B=0.476  (sum=1.032, 3.2% vig)
        No-vig:     A=0.539, B=0.461
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


# ── Value Detection ───────────────────────────────────────────────────────────

def detect_value(
    model_prob: float,
    market_prob: float,
    min_edge: float = 0.05,
) -> dict:
    """
    Compare model probability to market-implied probability.

    Args:
        model_prob:   Model's estimated win probability for a fighter
        market_prob:  Sportsbook's no-vig implied probability
        min_edge:     Minimum edge to flag as value (default 5%)

    Returns dict with edge size, direction, and value flag.
    """
    edge = model_prob - market_prob
    return {
        "model_prob":  round(model_prob, 4),
        "market_prob": round(market_prob, 4),
        "edge":        round(edge, 4),
        "edge_pct":    round(edge * 100, 2),
        "is_value":    edge >= min_edge,
        "direction":   "overvalued" if edge > 0 else "undervalued",
    }


def upset_score(
    model_prob_underdog: float,
    market_prob_underdog: float,
) -> float:
    """
    Score indicating how much the model disagrees with the market on the underdog.
    Higher = stronger upset signal.

    Positive = model thinks underdog is more likely to win than market does.
    """
    return round(model_prob_underdog - market_prob_underdog, 4)


def detect_line_movement(
    opening_odds_a: int,
    current_odds_a: int,
) -> dict:
    """
    Detect how much the line has moved since opening.

    Large movement suggests new information (injury, sharp money, public steam).
    """
    opening_prob = american_to_prob(opening_odds_a)
    current_prob = american_to_prob(current_odds_a)
    movement = current_prob - opening_prob

    return {
        "opening_odds":  opening_odds_a,
        "current_odds":  current_odds_a,
        "opening_prob":  round(opening_prob, 4),
        "current_prob":  round(current_prob, 4),
        "movement":      round(movement, 4),
        "moved_toward_a": movement > 0,
        "significant":   abs(movement) >= 0.05,  # 5% shift is notable
    }


def detect_reverse_line_movement(
    opening_odds_a: int,
    current_odds_a: int,
    pct_bets_on_a: Optional[float] = None,
) -> dict:
    """
    Detect reverse line movement — a sharp money signal.

    Reverse line movement = line moves AGAINST the side getting most bets.
    This means professional bettors are on the other side.

    Example:
        70% of public bets on Fighter A
        But line moves to make Fighter A more expensive (AGAINST public)
        → Sharp money on Fighter B

    Args:
        opening_odds_a:   Opening American odds for fighter A
        current_odds_a:   Current American odds for fighter A
        pct_bets_on_a:    % of public betting tickets on A (0.0-1.0), if available
    """
    movement = detect_line_movement(opening_odds_a, current_odds_a)
    line_moved_toward_a = movement["moved_toward_a"]

    rlm_detected = False
    explanation = ""

    if pct_bets_on_a is not None:
        public_favors_a = pct_bets_on_a > 0.55
        # RLM = public on A but line moved away from A (toward B)
        if public_favors_a and not line_moved_toward_a:
            rlm_detected = True
            explanation = f"Public {pct_bets_on_a:.0%} on A but line moved toward B — sharp money on B"
        elif not public_favors_a and line_moved_toward_a:
            rlm_detected = True
            explanation = f"Public {1-pct_bets_on_a:.0%} on B but line moved toward A — sharp money on A"
    else:
        # Without public data, flag any significant movement as potentially notable
        if movement["significant"]:
            explanation = f"Significant line movement: {movement['movement']:+.1%}"

    return {
        **movement,
        "rlm_detected":  rlm_detected,
        "explanation":   explanation,
        "sharp_side":    "B" if (rlm_detected and line_moved_toward_a is False) else
                         "A" if (rlm_detected and line_moved_toward_a) else "unknown",
    }


# ── Full Fight Value Analysis ─────────────────────────────────────────────────

def analyze_fight_value(
    fighter_a_name: str,
    fighter_b_name: str,
    model_prob_a: float,
    odds_a: int,
    odds_b: int,
    opening_odds_a: Optional[int] = None,
    opening_odds_b: Optional[int] = None,
    pct_bets_on_a: Optional[float] = None,
    min_edge: float = 0.05,
) -> dict:
    """
    Full value analysis for a fight matchup.

    Combines model probabilities with market odds to produce a complete
    betting intelligence report for one fight.

    Args:
        fighter_a_name:  Name of fighter A
        fighter_b_name:  Name of fighter B
        model_prob_a:    Model's win probability for fighter A (0-1)
        odds_a:          Current American moneyline odds for fighter A
        odds_b:          Current American moneyline odds for fighter B
        opening_odds_a:  Opening line for fighter A (optional)
        opening_odds_b:  Opening line for fighter B (optional)
        pct_bets_on_a:   % of public bets on fighter A (optional)
        min_edge:        Minimum edge threshold to flag value (default 5%)

    Returns:
        Complete analysis dict with value flags, edges, and recommendations
    """
    model_prob_b = 1.0 - model_prob_a

    # Market implied probs (raw, with vig)
    raw_prob_a = american_to_prob(odds_a)
    raw_prob_b = american_to_prob(odds_b)

    # Remove vig for fair comparison
    fair_prob_a, fair_prob_b = remove_vig(raw_prob_a, raw_prob_b)

    # Value analysis
    value_a = detect_value(model_prob_a, fair_prob_a, min_edge)
    value_b = detect_value(model_prob_b, fair_prob_b, min_edge)

    # Line movement
    line_movement = None
    rlm = None
    if opening_odds_a is not None:
        line_movement = detect_line_movement(opening_odds_a, odds_a)
        rlm = detect_reverse_line_movement(opening_odds_a, odds_a, pct_bets_on_a)

    # Determine best value pick
    best_value = None
    if value_a["is_value"] and value_b["is_value"]:
        best_value = fighter_a_name if value_a["edge"] > value_b["edge"] else fighter_b_name
    elif value_a["is_value"]:
        best_value = fighter_a_name
    elif value_b["is_value"]:
        best_value = fighter_b_name

    # Upset detection — underdog is whoever has lower model prob
    if model_prob_a < model_prob_b:
        upset = upset_score(model_prob_a, fair_prob_a)
        upset_fighter = fighter_a_name
    else:
        upset = upset_score(model_prob_b, fair_prob_b)
        upset_fighter = fighter_b_name

    return {
        "fighter_a": fighter_a_name,
        "fighter_b": fighter_b_name,
        # Model
        "model_prob_a":  round(model_prob_a, 4),
        "model_prob_b":  round(model_prob_b, 4),
        # Market
        "odds_a":        odds_a,
        "odds_b":        odds_b,
        "fair_prob_a":   round(fair_prob_a, 4),
        "fair_prob_b":   round(fair_prob_b, 4),
        "vig":           round((raw_prob_a + raw_prob_b - 1.0) * 100, 2),
        # Value
        "value_a":       value_a,
        "value_b":       value_b,
        "best_value_pick": best_value,
        # Upset
        "upset_score":   upset,
        "upset_fighter": upset_fighter,
        "upset_alert":   abs(upset) >= 0.10,
        # Line movement
        "line_movement": line_movement,
        "rlm":           rlm,
        # Summary flag
        "has_value":     best_value is not None,
    }
