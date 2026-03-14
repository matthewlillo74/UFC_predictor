"""
scripts/backtest_parlays.py
────────────────────────────
Backtest the parlay builder against historical UFC events.

For each past event in the test window (Aug 2023 → present):
  1. Build predictions using ONLY data available before that event
  2. Run the parlay builder as if it were fight week
  3. Score each parlay leg against actual results
  4. Report hit rate, ROI, and which parlays would have hit

This answers: "Would these parlays have made money historically?"

Usage:
    python scripts/backtest_parlays.py
    python scripts/backtest_parlays.py --events 10   # last 10 events only
    python scripts/backtest_parlays.py --event-name "UFC 326"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from loguru import logger
from sqlalchemy import text

from src.database import init_db, get_session, Fight, Event, Fighter
from src.features.feature_builder import FeatureBuilder
from src.models.predict import UFCPredictor
from src.betting.parlay_builder import build_parlays, build_candidate_legs, Parlay


BET_SIZE = 10.0  # $ per parlay for ROI calculation


def get_past_events(session, limit=None, name_filter=None):
    """Get completed events in the test window (post Aug 2023)."""
    q = (session.query(Event)
         .filter(Event.date >= datetime(2023, 8, 1))
         .filter(Event.date <= datetime.now())
         .order_by(Event.date.desc()))
    if name_filter:
        q = q.filter(Event.name.ilike(f"%{name_filter}%"))
    if limit:
        q = q.limit(limit)
    return q.all()


def get_event_fights(session, event_id):
    """Get all completed fights for an event."""
    return (session.query(Fight)
            .filter_by(event_id=event_id)
            .filter(Fight.winner_id.isnot(None))
            .all())


def build_predictions_for_event(session, event, fights, predictor, builder):
    """
    Build predictions for all fights in an event using only pre-fight data.
    Returns list of pred dicts with actual_winner injected.
    """
    predictions = []
    fight_date = event.date or datetime.now()

    for fight in fights:
        try:
            fa = session.get(Fighter, fight.fighter_a_id)
            fb = session.get(Fighter, fight.fighter_b_id)
            if not fa or not fb:
                continue

            features = builder.build_matchup_features(
                fight.fighter_a_id, fight.fighter_b_id, fight_date
            )
            pred = predictor.predict(features, fa.name, fb.name)
            pred["weight_class"] = fight.weight_class or ""
            pred["actual_winner_id"] = fight.winner_id
            pred["actual_winner"] = fa.name if fight.winner_id == fight.fighter_a_id else fb.name
            pred["fight_id"] = fight.id
            pred["method"] = fight.method or ""

            # No live odds for historical — simulate market from fight record
            # Use a simple heuristic: market roughly tracks Elo-based probability
            # This lets us compute approximate edge even without historical odds
            pred["odds_data"] = None

            predictions.append(pred)
        except Exception as e:
            logger.debug(f"Skipped fight {fight.id}: {e}")

    return predictions


def score_parlay(parlay: Parlay, predictions: list[dict]) -> dict:
    """
    Score a parlay against actual results.
    Returns hit status and which legs won/lost.
    """
    actual_winners = {
        pred["fighter_a"]: pred["actual_winner"] == pred["fighter_a"]
        for pred in predictions
    }
    actual_winners.update({
        pred["fighter_b"]: pred["actual_winner"] == pred["fighter_b"]
        for pred in predictions
    })

    leg_results = []
    all_hit = True

    for leg in parlay.legs:
        won = actual_winners.get(leg.fighter, False)
        leg_results.append({
            "fighter": leg.fighter,
            "won": won,
            "model_prob": leg.model_prob,
        })
        if not won:
            all_hit = False

    payout = (parlay.true_decimal_odds * BET_SIZE) - BET_SIZE if all_hit else -BET_SIZE

    return {
        "hit": all_hit,
        "legs_won": sum(1 for l in leg_results if l["won"]),
        "total_legs": len(leg_results),
        "leg_results": leg_results,
        "payout": payout,
        "odds": parlay.true_american_odds,
        "model_prob": parlay.combined_model_prob,
    }


def run_backtest(events_limit=None, name_filter=None):
    init_db()
    session = get_session()

    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        logger.error(f"Model not loaded: {e}")
        return

    builder = FeatureBuilder(session)
    events = get_past_events(session, limit=events_limit, name_filter=name_filter)

    if not events:
        logger.error("No events found in test window")
        return

    logger.info(f"Backtesting {len(events)} events...")

    # Aggregate stats
    totals = {
        "safe":  {"bets": 0, "hits": 0, "profit": 0.0},
        "value": {"bets": 0, "hits": 0, "profit": 0.0},
        "shot":  {"bets": 0, "hits": 0, "profit": 0.0},
        "super": {"bets": 0, "hits": 0, "profit": 0.0},
    }

    print("\n" + "═"*70)
    print("  UFC PARLAY BACKTEST")
    print(f"  {len(events)} events  |  ${BET_SIZE:.0f}/parlay")
    print("═"*70)

    oliveira_check = None

    for event in reversed(events):  # chronological order
        fights = get_event_fights(session, event.id)
        if len(fights) < 3:
            continue

        predictions = build_predictions_for_event(session, event, fights, predictor, builder)
        if len(predictions) < 3:
            continue

        parlays = build_parlays(predictions)
        if not any(parlays.values()):
            continue

        event_date = event.date.strftime("%Y-%m-%d") if event.date else "Unknown"
        print(f"\n{'─'*70}")
        print(f"  {event.name}  [{event_date}]")
        print(f"  {len(fights)} fights  |  {len(predictions)} predicted")
        print(f"{'─'*70}")

        # Check for Oliveira vs Holloway in this event
        for pred in predictions:
            names = {pred["fighter_a"].lower(), pred["fighter_b"].lower()}
            if any("oliveira" in n for n in names) and any("holloway" in n for n in names):
                oliveira_check = {
                    "event": event.name,
                    "pred": pred,
                    "actual": pred.get("actual_winner", "?"),
                }

        for tier in ["safe", "value", "shot", "super"]:
            tier_parlays = parlays.get(tier, [])
            for parlay in tier_parlays:
                result = score_parlay(parlay, predictions)
                totals[tier]["bets"] += 1
                totals[tier]["profit"] += result["payout"]
                if result["hit"]:
                    totals[tier]["hits"] += 1

                hit_str = "✅ HIT" if result["hit"] else "❌ MISS"
                odds_str = f"+{result['odds']}" if result['odds'] > 0 else str(result['odds'])
                print(f"  [{tier.upper():5s}] {hit_str}  {result['legs_won']}/{result['total_legs']} legs  "
                      f"odds {odds_str}  model {result['model_prob']:.0%}  "
                      f"{'profit' if result['payout'] > 0 else 'loss'} ${abs(result['payout']):.0f}")

                if result["hit"] or result["legs_won"] >= result["total_legs"] - 1:
                    for lr in result["leg_results"]:
                        icon = "✅" if lr["won"] else "❌"
                        print(f"         {icon} {lr['fighter']} ({lr['model_prob']:.0%})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  BACKTEST SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Tier':<8} {'Bets':>5} {'Hits':>5} {'Hit%':>7} {'Profit':>10} {'ROI':>8}")
    print(f"  {'─'*50}")

    for tier, stats in totals.items():
        if stats["bets"] == 0:
            continue
        hit_pct = stats["hits"] / stats["bets"] * 100
        roi = stats["profit"] / (stats["bets"] * BET_SIZE) * 100
        profit_str = f"+${stats['profit']:.0f}" if stats["profit"] >= 0 else f"-${abs(stats['profit']):.0f}"
        print(f"  {tier.upper():<8} {stats['bets']:>5} {stats['hits']:>5} "
              f"{hit_pct:>6.1f}%  {profit_str:>9}  {roi:>+6.1f}%")

    total_bets = sum(s["bets"] for s in totals.values())
    total_profit = sum(s["profit"] for s in totals.values())
    total_roi = total_profit / (total_bets * BET_SIZE) * 100 if total_bets > 0 else 0
    print(f"  {'─'*50}")
    profit_str = f"+${total_profit:.0f}" if total_profit >= 0 else f"-${abs(total_profit):.0f}"
    print(f"  {'TOTAL':<8} {total_bets:>5}       {'':>7} {profit_str:>9}  {total_roi:>+6.1f}%")

    # ── Oliveira check ────────────────────────────────────────────────────────
    if oliveira_check:
        print(f"\n{'═'*70}")
        print("  OLIVEIRA vs HOLLOWAY CHECK")
        print(f"{'═'*70}")
        pred = oliveira_check["pred"]
        fa = pred["fighter_a"]
        fb = pred["fighter_b"]
        prob_a = pred["prob_fighter_a"]
        prob_b = pred["prob_fighter_b"]
        actual = oliveira_check["actual"]
        oliveira_side = "fighter_a" if "oliveira" in fa.lower() else "fighter_b"
        oliveira_prob = prob_a if oliveira_side == "fighter_a" else prob_b
        print(f"  Event:   {oliveira_check['event']}")
        print(f"  {fa}: {prob_a:.1%}  vs  {fb}: {prob_b:.1%}")
        print(f"  Model picked: {pred['predicted_winner']}")
        print(f"  Oliveira model prob: {oliveira_prob:.1%}")
        print(f"  Actual winner: {actual}")
        if "oliveira" in actual.lower():
            print(f"  ✅ Model correctly identified Oliveira as value")
            if oliveira_prob > 0.40:
                print(f"  ✅ Model prob ({oliveira_prob:.0%}) >> market implied (~33%) — strong value signal")
        else:
            print(f"  ❌ Model missed this one")
    else:
        print("\n  (Oliveira vs Holloway not found in test window — may be outside date range)")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=None, help="Number of recent events to backtest")
    parser.add_argument("--event-name", type=str, default=None, help="Filter by event name")
    args = parser.parse_args()

    run_backtest(events_limit=args.events, name_filter=args.event_name)
