"""
src/betting/parlay_builder.py
──────────────────────────────
Data-backed parlay construction from fight predictions.

Philosophy:
  - Never just take the highest probability legs — that gives you chalk parlays
    with terrible payouts and zero edge
  - Score each leg on BOTH confidence AND market edge
  - Build multiple risk tiers: safe, value, and shot parlays
  - Compute true expected value so you know if the parlay is mathematically worth it

Parlay EV formula:
  Combined prob = product of all leg probs
  Parlay odds = product of all leg decimal odds
  EV = (combined_prob * parlay_odds) - 1
  Positive EV = worth considering
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ParlayLeg:
    fighter:        str
    opponent:       str
    model_prob:     float
    market_prob:    float
    american_odds:  Optional[int]
    edge:           float           # model_prob - market_prob
    weight_class:   str
    is_underdog:    bool

    @property
    def decimal_odds(self) -> float:
        if not self.american_odds:
            return 1 / self.market_prob if self.market_prob else 2.0
        if self.american_odds > 0:
            return (self.american_odds / 100) + 1
        return (100 / abs(self.american_odds)) + 1

    @property
    def leg_score(self) -> float:
        """
        Score each potential leg on confidence + edge.
        High confidence with positive edge = best leg.
        High confidence with negative edge = avoid (you're buying chalk at bad price).
        """
        confidence_score = self.model_prob           # 0.5 to 1.0
        edge_score = max(self.edge, -0.05)           # penalize negative edge but don't kill it
        return (confidence_score * 0.6) + (edge_score * 0.4 + 0.05) * 0.4


@dataclass
class Parlay:
    legs:           list[ParlayLeg]
    tier:           str             # "safe" | "value" | "shot"
    description:    str

    @property
    def combined_model_prob(self) -> float:
        p = 1.0
        for leg in self.legs:
            p *= leg.model_prob
        return p

    @property
    def combined_market_prob(self) -> float:
        p = 1.0
        for leg in self.legs:
            p *= leg.market_prob
        return p

    @property
    def true_decimal_odds(self) -> float:
        """What the parlay actually pays at a sportsbook (product of decimal odds)."""
        odds = 1.0
        for leg in self.legs:
            odds *= leg.decimal_odds
        return odds

    @property
    def true_american_odds(self) -> int:
        d = self.true_decimal_odds
        if d >= 2.0:
            return int((d - 1) * 100)
        return int(-100 / (d - 1))

    @property
    def expected_value(self) -> float:
        """
        EV per $1 bet.
        Positive = profitable long run.
        e.g. 0.15 means you expect to profit $0.15 per $1 wagered.
        """
        return (self.combined_model_prob * self.true_decimal_odds) - 1

    @property
    def market_ev(self) -> float:
        """EV using market probs — should be negative (that's the vig)."""
        return (self.combined_market_prob * self.true_decimal_odds) - 1

    @property
    def edge_vs_market(self) -> float:
        """How much better our EV is than the market's EV."""
        return self.expected_value - self.market_ev

    def summary(self) -> str:
        legs_str = "\n".join(
            f"  {'🐶 ' if leg.is_underdog else '  '}{leg.fighter} ({leg.model_prob:.0%} model / "
            f"{'+'if leg.american_odds and leg.american_odds>0 else ''}{leg.american_odds or 'N/A'} odds / "
            f"{leg.edge:+.0%} edge)"
            for leg in self.legs
        )
        ev_str = f"+{self.expected_value:.1%}" if self.expected_value > 0 else f"{self.expected_value:.1%}"
        return (
            f"{'='*52}\n"
            f"  {self.tier.upper()} PARLAY — {len(self.legs)} legs\n"
            f"  {self.description}\n"
            f"{'='*52}\n"
            f"{legs_str}\n"
            f"  ─────────────────────────────────────────────\n"
            f"  Combined model prob:  {self.combined_model_prob:.1%}\n"
            f"  Parlay odds:          +{self.true_american_odds}\n"
            f"  Expected value:       {ev_str} per $1\n"
            f"  Edge vs market:       {self.edge_vs_market:+.1%}\n"
        )


def american_to_decimal(american: int) -> float:
    if american > 0:
        return (american / 100) + 1
    return (100 / abs(american)) + 1


def build_candidate_legs(predictions: list[dict], min_model_prob: float = 0.52) -> list[ParlayLeg]:
    """
    Build all valid parlay legs from fight predictions.
    Only include legs where the model has some conviction (>52%).
    """
    legs = []
    for pred in predictions:
        odds_data = pred.get("odds_data")

        for fighter, opponent, model_prob, market_prob, american_odds in [
            (
                pred["fighter_a"], pred["fighter_b"],
                pred["prob_fighter_a"], odds_data.get("fair_prob_a", 0.5) if odds_data else 0.5,
                odds_data.get("odds_a") if odds_data else None,
            ),
            (
                pred["fighter_b"], pred["fighter_a"],
                pred["prob_fighter_b"], odds_data.get("fair_prob_b", 0.5) if odds_data else 0.5,
                odds_data.get("odds_b") if odds_data else None,
            ),
        ]:
            if model_prob < min_model_prob:
                continue
            # Only take one side per fight — the one the model prefers
            if fighter == pred.get("predicted_winner") or model_prob > 0.5:
                edge = model_prob - market_prob
                legs.append(ParlayLeg(
                    fighter=fighter,
                    opponent=opponent,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    american_odds=american_odds,
                    edge=edge,
                    weight_class=pred.get("weight_class", ""),
                    is_underdog=model_prob < 0.5 or (american_odds and american_odds > 0),
                ))
                break  # one leg per fight

    # Sort by leg score descending
    legs.sort(key=lambda l: l.leg_score, reverse=True)
    return legs


def build_parlays(predictions: list[dict]) -> dict[str, list[Parlay]]:
    """
    Build recommended parlays across three tiers.

    Returns dict with keys: 'safe', 'value', 'shot', 'super'
    """
    all_legs = build_candidate_legs(predictions, min_model_prob=0.52)

    if len(all_legs) < 2:
        return {}

    # ── Tier 1: Safe parlay (3 legs, highest confidence + positive edge) ──────
    safe_legs = [l for l in all_legs if l.model_prob >= 0.60 and l.edge >= 0.0][:3]
    safe_parlays = []
    if len(safe_legs) >= 3:
        p = Parlay(
            legs=safe_legs[:3],
            tier="safe",
            description="High-confidence picks with positive model edge"
        )
        safe_parlays.append(p)

    # ── Tier 2: Value parlay (4-5 legs, balanced confidence + edge) ──────────
    value_legs = [l for l in all_legs if l.model_prob >= 0.55]
    value_parlays = []

    best_value = None
    best_ev = -999
    for n in [4, 5]:
        if len(value_legs) < n:
            continue
        for combo in combinations(value_legs[:8], n):
            p = Parlay(
                legs=list(combo),
                tier="value",
                description=f"{n}-leg value parlay — model edge on every leg"
            )
            if p.expected_value > best_ev and p.combined_model_prob > 0.08:
                best_ev = p.expected_value
                best_value = p

    if best_value:
        value_parlays.append(best_value)

    # ── Tier 3: Shot parlay (4-5 legs, mix in an underdog for payout) ────────
    shot_parlays = []
    underdogs = [l for l in all_legs if l.is_underdog and l.edge >= 0.05]
    favorites = [l for l in all_legs if not l.is_underdog and l.model_prob >= 0.60]

    if underdogs and len(favorites) >= 3:
        shot_legs = favorites[:3] + underdogs[:1]
        shot_legs.sort(key=lambda l: l.leg_score, reverse=True)
        p = Parlay(
            legs=shot_legs[:4],
            tier="shot",
            description="3 strong favorites + 1 underdog with model edge — enhanced payout"
        )
        shot_parlays.append(p)

        if len(underdogs) >= 2 and len(favorites) >= 3:
            shot_legs2 = favorites[:3] + underdogs[:2]
            p2 = Parlay(
                legs=shot_legs2[:5],
                tier="shot",
                description="3 strong favorites + 2 model-backed underdogs — big payout potential"
            )
            shot_parlays.append(p2)

    # ── Tier 4: Super parlay (8-12 legs, all model picks, lottery ticket) ────
    super_legs = [l for l in all_legs if l.model_prob >= 0.52]
    super_parlays = []
    for n in [8, 10, 12]:
        if len(super_legs) >= n:
            p = Parlay(
                legs=super_legs[:n],
                tier="super",
                description=f"{n}-leg super parlay — all model picks, lottery ticket"
            )
            super_parlays.append(p)
            break

    return {
        "safe":  safe_parlays,
        "value": value_parlays,
        "shot":  shot_parlays,
        "super": super_parlays,
    }
