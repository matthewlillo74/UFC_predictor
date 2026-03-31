"""
scripts/log_live_results.py
────────────────────────────
Logs actual fight results against model predictions for live accuracy tracking.

This is the most valuable accuracy signal — unlike backtests (which use fights
the model trained on), live results are pure out-of-sample ground truth.

Tracks ALL prediction dimensions:
  - Winner prediction accuracy + confidence calibration
  - Method prediction (KO/TKO vs Submission vs Decision)
  - Round O/U accuracy (model said under 2.5, did it go early?)
  - Value edge tracking (do high-edge picks actually outperform?)
  - Market agreement analysis (when model disagrees with market, who wins?)
  - Flat $100 P&L simulation
  - Pattern analysis in misses

Usage:
    # After event results are in (run post-event pipeline first):
    python scripts/log_live_results.py --event "UFC Fight Night: Evloev vs. Murphy"

    # Full report:
    python scripts/log_live_results.py --report

    # Last N events only:
    python scripts/log_live_results.py --report --events 5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.database import init_db, get_session, Prediction, Fight, Fighter, Event, BettingOdds
from config import PREDICTIONS_DIR


LIVE_LOG_PATH = PREDICTIONS_DIR / "live_accuracy.csv"

LOG_COLUMNS = [
    # Identity
    "event", "fight_date", "fighter_a", "fighter_b", "weight_class",
    # Winner
    "predicted_winner", "confidence", "actual_winner", "winner_correct",
    # Market
    "market_implied_prob", "model_edge", "odds_favorite", "model_agreed_with_market",
    # Method
    "method_predicted", "method_prob", "method_actual", "method_correct",
    # Round
    "round_line", "prob_under", "actual_finish_round", "went_early", "round_correct",
    # Combined
    "full_prediction_correct",
    # P&L
    "flat_bet_return",
]


def prob_to_payout(prob: float, stake: float = 100) -> float:
    """Net profit if bet wins at fair odds implied by prob."""
    if prob <= 0 or prob >= 1:
        return 0
    return stake * (1 - prob) / prob


def get_or_create_log() -> list[dict]:
    if LIVE_LOG_PATH.exists():
        with open(LIVE_LOG_PATH, newline="") as f:
            return list(csv.DictReader(f))
    return []


def save_log(rows: list[dict]):
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.success(f"Saved {len(rows)} fights to {LIVE_LOG_PATH}")


def score_event(session, event_name: str):
    """Score all predictions for an event against actual results."""
    event = session.query(Event).filter(
        Event.name.ilike(f"%{event_name}%")
    ).first()

    if not event:
        logger.error(f"Event not found: '{event_name}'")
        recent = session.query(Event).order_by(Event.date.desc()).limit(10).all()
        for e in recent:
            logger.info(f"  {e.name} ({e.date})")
        return

    logger.info(f"Scoring: {event.name}")

    fights = session.query(Fight).filter(
        Fight.event_id == event.id,
        Fight.winner_id.isnot(None)
    ).all()

    if not fights:
        logger.warning("No completed fights — run post-event pipeline first")
        return

    log = get_or_create_log()
    existing_keys = {(r["event"], r["fighter_a"], r["fighter_b"]) for r in log}
    new_rows = []
    scored = 0

    for fight in fights:
        fa = session.get(Fighter, fight.fighter_a_id)
        fb = session.get(Fighter, fight.fighter_b_id)
        if not fa or not fb:
            continue

        # Primary lookup: by fight_id (works when pipeline ran correctly)
        pred = session.query(Prediction).filter_by(fight_id=fight.id).first()

        # Fallback: find prediction by fighter IDs across any fight row
        # This handles the case where the pipeline scraped the event twice,
        # creating new fight IDs after predictions were already stored
        # Fallback: find prediction by fighter IDs across any fight row
        # This handles the case where the pipeline scraped the event twice,
        # creating new fight IDs after predictions were already stored
        if not pred:
            # Find all fights with these two fighters
            sibling_fights = session.query(Fight).filter(
                Fight.fighter_a_id == fight.fighter_a_id,
                Fight.fighter_b_id == fight.fighter_b_id,
            ).all()
            sibling_fights += session.query(Fight).filter(
                Fight.fighter_a_id == fight.fighter_b_id,
                Fight.fighter_b_id == fight.fighter_a_id,
            ).all()
            
            for sibling in sibling_fights:
                pred = session.query(Prediction).filter_by(fight_id=sibling.id).first()
                if pred:
                    logger.debug(f"Matched prediction via fighter IDs for {fa.name} vs {fb.name}")
                    break
            
        # Catch-all if pred is STILL None after the primary check and the fallback
        if not pred:
            logger.debug(f"No prediction found for {fa.name} vs {fb.name} — skipping")
            continue

        actual_winner = session.get(Fighter, fight.winner_id)
        if not pred.predicted_winner_id:
            logger.debug(f"Prediction for {fa.name} vs {fb.name} has no predicted_winner_id — skipping")
            continue

        predicted_winner = session.get(Fighter, pred.predicted_winner_id)

        key = (event.name, fa.name, fb.name)
        if key in existing_keys:
            continue

        # ── Winner ────────────────────────────────────────────────────────
        winner_correct = predicted_winner.id == actual_winner.id

        # ── Market & P&L ──────────────────────────────────────────────────
        odds_row = session.query(BettingOdds).filter_by(fight_id=fight.id).first()
        market_implied_prob = 0.5
        model_edge = 0.0
        odds_favorite = ""
        model_agreed = ""
        flat_bet_return = 0.0

        if odds_row:
            mp_a = odds_row.implied_prob_a or 0.5
            mp_b = odds_row.implied_prob_b or 0.5
            odds_favorite = fa.name if mp_a > 0.5 else fb.name
            model_agreed = "1" if ((pred.prob_fighter_a or 0.5) > 0.5) == (mp_a > 0.5) else "0"

            if predicted_winner.id == fa.id:
                market_implied_prob = mp_a
                model_edge = (pred.prob_fighter_a or 0.5) - mp_a
            else:
                market_implied_prob = mp_b
                model_edge = (pred.prob_fighter_b or 0.5) - mp_b

            flat_bet_return = prob_to_payout(market_implied_prob, 100) if winner_correct else -100.0

        # ── Method ────────────────────────────────────────────────────────
        method_probs = {
            "KO_TKO":     pred.prob_ko_tko or 0,
            "Submission": pred.prob_submission or 0,
            "Decision":   pred.prob_decision or 0,
        }
        method_predicted = max(method_probs, key=method_probs.get)
        method_prob = method_probs[method_predicted]
        method_actual = fight.method or ""
        method_correct = method_predicted == method_actual

        # ── Round O/U ─────────────────────────────────────────────────────
        is_title = fight.is_title_fight
        finish_round = fight.finish_round

        if is_title:
            round_line = "under_3_5"
            prob_under = pred.prob_under_3_5 or 0.0
            went_early = 1 if (finish_round and finish_round <= 3 and
                               method_actual in ("KO_TKO", "Submission")) else 0
        else:
            round_line = "under_2_5"
            prob_under = pred.prob_under_2_5 or 0.0
            went_early = 1 if (finish_round and finish_round <= 2 and
                               method_actual in ("KO_TKO", "Submission")) else 0

        model_said_under = prob_under > 0.5
        round_correct = (model_said_under and went_early == 1) or \
                        (not model_said_under and went_early == 0)

        full_correct = winner_correct and method_correct and round_correct

        # ── Update DB ─────────────────────────────────────────────────────
        pred.was_correct = winner_correct
        pred.method_correct = method_correct
        pred.round_correct = round_correct
        pred.actual_winner_id = actual_winner.id

        # ── Log row ───────────────────────────────────────────────────────
        new_rows.append({
            "event":                   event.name,
            "fight_date":              fight.fight_date.date() if fight.fight_date else "",
            "fighter_a":               fa.name,
            "fighter_b":               fb.name,
            "weight_class":            fight.weight_class or "",
            "predicted_winner":        predicted_winner.name,
            "confidence":              f"{pred.confidence_score:.3f}" if pred.confidence_score else "",
            "actual_winner":           actual_winner.name,
            "winner_correct":          "1" if winner_correct else "0",
            "market_implied_prob":     f"{market_implied_prob:.3f}",
            "model_edge":              f"{model_edge:+.3f}",
            "odds_favorite":           odds_favorite,
            "model_agreed_with_market": model_agreed,
            "method_predicted":        method_predicted,
            "method_prob":             f"{method_prob:.3f}",
            "method_actual":           method_actual,
            "method_correct":          "1" if method_correct else "0",
            "round_line":              round_line,
            "prob_under":              f"{prob_under:.3f}",
            "actual_finish_round":     str(finish_round) if finish_round else "5",
            "went_early":              str(went_early),
            "round_correct":           "1" if round_correct else "0",
            "full_prediction_correct": "1" if full_correct else "0",
            "flat_bet_return":         f"{flat_bet_return:.2f}",
        })
        existing_keys.add(key)

        w = "✅" if winner_correct else "❌"
        m = "✅" if method_correct else "❌"
        r = "✅" if round_correct else "❌"
        logger.info(
            f"  {w}W {m}M {r}R  "
            f"{fa.name} vs {fb.name}  → "
            f"predicted {predicted_winner.name} ({pred.confidence_score:.0%}), "
            f"actual {actual_winner.name} by {method_actual} R{finish_round or '?'}"
        )
        scored += 1

    session.commit()
    log.extend(new_rows)
    save_log(log)
    logger.success(f"Scored {scored} fights for {event.name}")


def print_report(session, last_n_events: int = None):
    log = get_or_create_log()
    if not log:
        logger.warning("No live results yet. Run: python scripts/log_live_results.py --event 'Event Name'")
        return

    df = pd.DataFrame(log)
    for col in ["winner_correct", "method_correct", "round_correct",
                "full_prediction_correct", "went_early"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["confidence"]      = pd.to_numeric(df["confidence"], errors="coerce")
    df["model_edge"]      = pd.to_numeric(df["model_edge"], errors="coerce")
    df["flat_bet_return"] = pd.to_numeric(df["flat_bet_return"], errors="coerce")
    df["prob_under"]      = pd.to_numeric(df["prob_under"], errors="coerce")

    if last_n_events:
        events = df["event"].unique()[-last_n_events:]
        df = df[df["event"].isin(events)]

    total = len(df)
    if total == 0:
        return

    w_acc = df["winner_correct"].mean()
    m_acc = df["method_correct"].mean()
    r_acc = df["round_correct"].mean()
    f_acc = df["full_prediction_correct"].mean()
    pnl   = df["flat_bet_return"].sum()
    n_ev  = df["event"].nunique()

    print("\n" + "═" * 64)
    print("  LIVE PREDICTION ACCURACY REPORT")
    print(f"  {total} fights · {n_ev} events")
    print("═" * 64)
    print(f"  Winner accuracy:       {w_acc:.1%}  ({df['winner_correct'].sum()}/{total})")
    print(f"  Method accuracy:       {m_acc:.1%}  ({df['method_correct'].sum()}/{total})")
    print(f"  Round O/U accuracy:    {r_acc:.1%}  ({df['round_correct'].sum()}/{total})")
    print(f"  Full (W+M+R) accuracy: {f_acc:.1%}  ({df['full_prediction_correct'].sum()}/{total})")
    print(f"  Flat $100 P&L:         ${pnl:+.0f}")
    print()

    # ── Confidence calibration ────────────────────────────────────────────
    print("  WINNER CALIBRATION BY CONFIDENCE")
    print(f"  {'Confidence':10s}  {'Fights':6s}  {'Hit%':7s}  {'vs Model':8s}  Status")
    print("  " + "─" * 54)
    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["50-55%","55-60%","60-65%","65-70%","70-75%","75-80%","80-90%","90%+"]
    df["conf_bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    for row in (df.groupby("conf_bucket", observed=True)
                .agg(fights=("winner_correct","count"), accuracy=("winner_correct","mean"))
                .reset_index().to_dict("records")):
        n = int(row["fights"])
        if n == 0:
            continue
        actual = row["accuracy"]
        bucket = str(row["conf_bucket"])
        try:
            lo = float(bucket.split("-")[0].rstrip("%")) / 100
            hi_raw = bucket.split("-")[1].rstrip("%+").rstrip("%")
            hi = float(hi_raw) / 100 if hi_raw.replace(".","").isdigit() else lo + 0.05
            expected = (lo + hi) / 2
        except Exception:
            expected = actual
        gap = actual - expected
        status = "✅ calibrated" if abs(gap) < 0.05 else ("📈 underconfident" if gap > 0 else "⚠️  overconfident")
        print(f"  {bucket:10s}  {n:6d}  {actual:6.1%}  {gap:+7.1%}   {status}")
    print()

    # ── Method breakdown ──────────────────────────────────────────────────
    print("  METHOD PREDICTION ACCURACY")
    print("  " + "─" * 40)
    for method in ["KO_TKO", "Submission", "Decision"]:
        s = df[df["method_predicted"] == method]
        if len(s) > 0:
            print(f"  Predicted {method:12s}: {s['method_correct'].mean():.1%}  ({len(s)} calls)")
    print()
    print("  ACTUAL METHOD DISTRIBUTION")
    for method, count in df["method_actual"].value_counts().items():
        bar = "█" * int(count / total * 25)
        print(f"  {method:15s}: {count/total:.1%}  ({count})  {bar}")
    print()

    # ── Round O/U ─────────────────────────────────────────────────────────
    print("  ROUND O/U ACCURACY")
    print("  " + "─" * 40)
    under_calls = df[df["prob_under"] > 0.5]
    over_calls  = df[df["prob_under"] <= 0.5]
    if len(under_calls) > 0:
        print(f"  Predicted UNDER: {under_calls['round_correct'].mean():.1%}  ({len(under_calls)} fights)")
    if len(over_calls) > 0:
        print(f"  Predicted OVER:  {over_calls['round_correct'].mean():.1%}  ({len(over_calls)} fights)")
    print(f"  Actual early finish rate: {df['went_early'].mean():.1%}  (model expects ~45%)")
    print()

    # ── Value edge performance ────────────────────────────────────────────
    has_edge = df[df["model_edge"].notna() & (df["model_edge"] != 0)]
    if len(has_edge) > 0:
        print("  VALUE EDGE PERFORMANCE")
        print("  (Does higher model edge = higher accuracy?)")
        print("  " + "─" * 50)
        edge_bins   = [-1, -0.10, 0, 0.10, 0.20, 1]
        edge_labels = ["<-10% edge","-10-0% edge","0-10% edge","10-20% edge","20%+ edge"]
        has_edge = has_edge.copy()
        has_edge["edge_bucket"] = pd.cut(has_edge["model_edge"], bins=edge_bins,
                                         labels=edge_labels, right=False)
        for row in (has_edge.groupby("edge_bucket", observed=True)
                    .agg(fights=("winner_correct","count"),
                         accuracy=("winner_correct","mean"),
                         pnl=("flat_bet_return","sum"))
                    .reset_index().to_dict("records")):
            n = int(row["fights"])
            if n == 0:
                continue
            print(f"  {str(row['edge_bucket']):14s}: {row['accuracy']:.1%} acc  "
                  f"${row['pnl']:+.0f} P&L  ({n} fights)")
        print()

        agree    = df[df["model_agreed_with_market"] == "1"]
        disagree = df[df["model_agreed_with_market"] == "0"]
        if len(agree) > 0:
            print(f"  Model agreed with market:    {agree['winner_correct'].mean():.1%}  ({len(agree)} fights)")
        if len(disagree) > 0:
            print(f"  Model DISAGREED with market: {disagree['winner_correct'].mean():.1%}  ({len(disagree)} fights)  ← key signal")
        print()

    # ── Weight class breakdown ─────────────────────────────────────────────
    print("  ACCURACY BY WEIGHT CLASS")
    print("  " + "─" * 58)
    print(f"  {'Division':28s}  {'Fights':6s}  {'Win%':6s}  {'Meth%':6s}  {'Rnd%':6s}  {'P&L':>8s}")
    print("  " + "─" * 58)
    wc_stats = (df.groupby("weight_class")
                .agg(fights=("winner_correct","count"),
                     w=("winner_correct","mean"),
                     m=("method_correct","mean"),
                     r=("round_correct","mean"),
                     pnl=("flat_bet_return","sum"))
                .reset_index()
                .sort_values("w", ascending=False))
    for row in wc_stats.to_dict("records"):
        n = int(row["fights"])
        if n == 0:
            continue
        trend = "📈" if row["w"] >= w_acc else "📉"
        print(f"  {trend} {row['weight_class']:26s}  {n:6d}  "
              f"{row['w']:5.1%}  {row['m']:5.1%}  {row['r']:5.1%}  ${row['pnl']:+7.0f}")
    print()

    # ── Favorite vs underdog ───────────────────────────────────────────────
    has_mkt = df[df["model_edge"].notna() & (df["market_implied_prob"].notna())]
    if len(has_mkt) > 0:
        has_mkt = has_mkt.copy()
        has_mkt["mkt_prob_f"] = pd.to_numeric(has_mkt["market_implied_prob"], errors="coerce")
        favs = has_mkt[has_mkt["mkt_prob_f"] >= 0.5]
        dogs = has_mkt[has_mkt["mkt_prob_f"] < 0.5]
        print("  FAVORITE vs UNDERDOG (market-defined)")
        print("  " + "─" * 50)
        if len(favs) > 0:
            print(f"  Betting market favorites: {favs['winner_correct'].mean():.1%}  "
                  f"({len(favs)} fights)  ${favs['flat_bet_return'].sum():+.0f} P&L")
        if len(dogs) > 0:
            print(f"  Betting market underdogs: {dogs['winner_correct'].mean():.1%}  "
                  f"({len(dogs)} fights)  ${dogs['flat_bet_return'].sum():+.0f} P&L")

        # Model-defined: when model picked the market underdog
        model_on_dog = has_mkt[
            ((has_mkt["mkt_prob_f"] < 0.5) & (has_mkt["predicted_winner"] == has_mkt["fighter_a"])) |
            ((has_mkt["mkt_prob_f"] >= 0.5) & (has_mkt["predicted_winner"] == has_mkt["fighter_b"]))
        ]
        model_on_fav = has_mkt[~has_mkt.index.isin(model_on_dog.index)]
        if len(model_on_fav) > 0:
            print(f"  Model picked market favorite:  {model_on_fav['winner_correct'].mean():.1%}  ({len(model_on_fav)} fights)")
        if len(model_on_dog) > 0:
            print(f"  Model picked market underdog:  {model_on_dog['winner_correct'].mean():.1%}  ({len(model_on_dog)} fights)  ← contrarian picks")
        print()

    # ── Fight type breakdown ───────────────────────────────────────────────
    print("  FIGHT TYPE BREAKDOWN")
    print("  " + "─" * 40)
    # Title fights vs regular
    if "is_title_fight" in df.columns:
        titles = df[df["is_title_fight"] == "1"]
        regular = df[df["is_title_fight"] != "1"]
        if len(titles) > 0:
            print(f"  Title fights:    {titles['winner_correct'].mean():.1%}  ({len(titles)} fights)")
        if len(regular) > 0:
            print(f"  Regular fights:  {regular['winner_correct'].mean():.1%}  ({len(regular)} fights)")

    # Finish fights vs decisions
    decisions = df[df["method_actual"] == "Decision"]
    finishes  = df[df["method_actual"].isin(["KO_TKO", "Submission"])]
    if len(decisions) > 0:
        print(f"  Decision fights: {decisions['winner_correct'].mean():.1%}  ({len(decisions)} fights)  "
              f"— method acc {decisions['method_correct'].mean():.1%}")
    if len(finishes) > 0:
        print(f"  Finish fights:   {finishes['winner_correct'].mean():.1%}  ({len(finishes)} fights)  "
              f"— method acc {finishes['method_correct'].mean():.1%}")

    # Women's vs men's
    womens = df[df["weight_class"].str.startswith("Women", na=False)]
    mens   = df[~df["weight_class"].str.startswith("Women", na=False)]
    if len(womens) > 0:
        print(f"  Women's fights:  {womens['winner_correct'].mean():.1%}  ({len(womens)} fights)")
    if len(mens) > 0:
        print(f"  Men's fights:    {mens['winner_correct'].mean():.1%}  ({len(mens)} fights)")
    print()

    # ── Line movement (closing line value) ────────────────────────────────
    # This requires manually logged line movement data - show placeholder
    # until we have enough events to compute it
    print("  CLOSING LINE VALUE (CLV)")
    print("  " + "─" * 40)
    print("  Track this manually: note the line when you bet vs closing line.")
    print("  Consistent movement toward your bets = model finding real edge.")
    print("  Example logged: Shem Rock +110 → closed -120 (moved toward pick)")
    print("  After 20+ bets, add --clv flag to log these and compute average CLV.")
    print()

    # ── Per-event ─────────────────────────────────────────────────────────
    print("  ACCURACY BY EVENT")
    print("  " + "─" * 54)
    for row in (df.groupby("event")
                .agg(fights=("winner_correct","count"),
                     w=("winner_correct","mean"),
                     m=("method_correct","mean"),
                     r=("round_correct","mean"),
                     pnl=("flat_bet_return","sum"))
                .reset_index().to_dict("records")):
        trend = "📈" if row["w"] >= w_acc else "📉"
        print(f"  {trend} {row['w']:.1%}W  {row['m']:.1%}M  {row['r']:.1%}R  "
              f"${row['pnl']:+.0f}  ({int(row['fights'])} fights)  {row['event'][-40:]}")
    print()

    # ── Misses ────────────────────────────────────────────────────────────
    wrong = df[df["winner_correct"] == 0].sort_values("confidence", ascending=False)
    if len(wrong) > 0:
        print(f"  HIGH-CONFIDENCE MISSES")
        print("  " + "─" * 54)
        for _, row in wrong.head(8).iterrows():
            conf = float(row["confidence"]) if row["confidence"] else 0
            edge = float(row["model_edge"]) if row["model_edge"] else 0
            wc = row.get("weight_class", "")
            print(f"  {conf:.0%}  {row['predicted_winner']:22s} ❌  "
                  f"actual: {row['actual_winner']:20s}  "
                  f"edge {edge:+.0%}  {wc}")
    print()

    # ── Miss patterns ─────────────────────────────────────────────────────
    print("  MISS PATTERNS (method on correct winner picks)")
    print("  " + "─" * 40)
    w_wrong_m = df[(df["winner_correct"] == 1) & (df["method_correct"] == 0)]
    if len(w_wrong_m) > 0:
        from collections import Counter
        pairs = Counter(zip(w_wrong_m["method_predicted"], w_wrong_m["method_actual"]))
        for (pred_m, act_m), count in pairs.most_common():
            print(f"  Predicted {pred_m:12s} → Actually {act_m}  ({count}x)")
    else:
        print("  No data yet")
    print()

    # ── Confidence by weight class (cross-tab) ────────────────────────────
    if len(df) >= 20:
        print("  CONFIDENCE CALIBRATION BY DIVISION (requires 20+ fights)")
        print("  " + "─" * 50)
        df["conf_hi"] = df["confidence"] >= 0.70
        cross = df.groupby(["weight_class", "conf_hi"])["winner_correct"].agg(
            ["mean", "count"]
        ).reset_index()
        for wc in cross["weight_class"].unique():
            wc_data = cross[cross["weight_class"] == wc]
            hi = wc_data[wc_data["conf_hi"] == True]
            lo = wc_data[wc_data["conf_hi"] == False]
            hi_str = f"{hi['mean'].values[0]:.0%} on hi-conf ({int(hi['count'].values[0])})" if len(hi) > 0 else ""
            lo_str = f"{lo['mean'].values[0]:.0%} on lo-conf ({int(lo['count'].values[0])})" if len(lo) > 0 else ""
            if hi_str or lo_str:
                print(f"  {wc:28s}: {hi_str}  {lo_str}")
        print()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",  type=str, help="Event name to score")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--events", type=int, default=None)
    args = parser.parse_args()

    init_db()
    session = get_session()

    if args.event:
        score_event(session, args.event)
    if args.report or not args.event:
        print_report(session, last_n_events=args.events)

    session.close()


if __name__ == "__main__":
    main()
