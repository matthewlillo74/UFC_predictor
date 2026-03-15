"""
scripts/log_live_results.py
────────────────────────────
Logs actual fight results against model predictions for live accuracy tracking.

This is the most valuable accuracy signal — unlike backtests (which use fights
the model trained on), live results are pure out-of-sample ground truth.

Usage:
    # Log a single result
    python scripts/log_live_results.py --event "UFC Fight Night: Emmett vs. Vallejos"

    # Show live accuracy report
    python scripts/log_live_results.py --report

    # Show report filtered to recent events
    python scripts/log_live_results.py --report --events 10

The script reads stored Predictions from the DB (written by predict_event.py)
and updates them with actual results. It also writes a human-readable CSV log
to data/predictions/live_accuracy.csv for easy review.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database import init_db, get_session, Prediction, Fight, Fighter, Event
from config import PREDICTIONS_DIR


LIVE_LOG_PATH = PREDICTIONS_DIR / "live_accuracy.csv"
LOG_COLUMNS = [
    "event", "fight_date", "fighter_a", "fighter_b",
    "predicted_winner", "confidence", "actual_winner",
    "correct", "edge_pct", "method_predicted", "method_actual",
]


def get_or_create_log() -> list[dict]:
    """Load existing log or create empty one."""
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
    logger.success(f"Live log saved: {LIVE_LOG_PATH} ({len(rows)} fights)")


def score_event(session, event_name: str):
    """
    Find all predictions for an event and score them against actual results.
    Updates Prediction.was_correct in the DB and appends to the CSV log.
    """
    # Find event
    event = session.query(Event).filter(
        Event.name.ilike(f"%{event_name}%")
    ).first()

    if not event:
        logger.error(f"Event not found: '{event_name}'")
        logger.info("Available recent events:")
        recent = session.query(Event).order_by(Event.date.desc()).limit(10).all()
        for e in recent:
            logger.info(f"  {e.name} ({e.date})")
        return

    logger.info(f"Scoring event: {event.name} ({event.date})")

    # Get all completed fights for this event
    fights = session.query(Fight).filter(
        Fight.event_id == event.id,
        Fight.winner_id.isnot(None)
    ).all()

    if not fights:
        logger.warning("No completed fights found for this event — run post-event pipeline first")
        return

    log = get_or_create_log()
    existing_keys = {(r["event"], r["fighter_a"], r["fighter_b"]) for r in log}

    scored = 0
    new_rows = []

    for fight in fights:
        fa = session.get(Fighter, fight.fighter_a_id)
        fb = session.get(Fighter, fight.fighter_b_id)
        if not fa or not fb:
            continue

        # Find the prediction for this fight
        pred = session.query(Prediction).filter_by(fight_id=fight.id).first()
        if not pred:
            logger.debug(f"No prediction found for {fa.name} vs {fb.name}")
            continue

        actual_winner = session.get(Fighter, fight.winner_id)
        if not actual_winner:
            continue

        predicted_winner = session.get(Fighter, pred.predicted_winner_id)
        if not predicted_winner:
            continue

        correct = predicted_winner.id == actual_winner.id

        # Update DB
        pred.was_correct = correct
        pred.actual_winner_id = actual_winner.id

        # Build log row
        key = (event.name, fa.name, fb.name)
        if key not in existing_keys:
            row = {
                "event":             event.name,
                "fight_date":        fight.fight_date.date() if fight.fight_date else "",
                "fighter_a":         fa.name,
                "fighter_b":         fb.name,
                "predicted_winner":  predicted_winner.name,
                "confidence":        f"{pred.confidence_score:.1%}" if pred.confidence_score else "",
                "actual_winner":     actual_winner.name,
                "correct":           "1" if correct else "0",
                "edge_pct":          "",  # populated if odds were available
                "method_predicted":  pred.predicted_method or "",
                "method_actual":     fight.method or "",
            }
            new_rows.append(row)
            existing_keys.add(key)

        result_str = "✅" if correct else "❌"
        logger.info(f"  {result_str} {fa.name} vs {fb.name} — predicted {predicted_winner.name}, actual {actual_winner.name}")
        scored += 1

    session.commit()
    log.extend(new_rows)
    save_log(log)
    logger.success(f"Scored {scored} fights for {event.name}")


def print_report(session, last_n_events: int = None):
    """
    Print a live accuracy report from the CSV log.
    Much more trustworthy than backtest accuracy since these are true out-of-sample.
    """
    log = get_or_create_log()
    if not log:
        logger.warning("No live results logged yet. Run after events with --event flag.")
        return

    df = pd.DataFrame(log)
    df["correct"] = df["correct"].astype(int)
    df["confidence"] = df["confidence"].str.rstrip("%").astype(float) / 100

    if last_n_events:
        events = df["event"].unique()[-last_n_events:]
        df = df[df["event"].isin(events)]

    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total if total > 0 else 0

    print("\n" + "═" * 60)
    print("  LIVE PREDICTION ACCURACY")
    print(f"  (True out-of-sample — {total} fights across {df['event'].nunique()} events)")
    print("═" * 60)
    print(f"  Overall accuracy:    {accuracy:.1%}  ({correct}/{total})")
    print()

    # By confidence bucket — does high confidence actually mean high accuracy?
    print("  ACCURACY BY CONFIDENCE")
    print("  " + "─" * 40)
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90%+"]
    df["conf_bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    bucket_stats = df.groupby("conf_bucket", observed=True).agg(
        fights=("correct", "count"),
        accuracy=("correct", "mean")
    ).reset_index()
    for _, row in bucket_stats.iterrows():
        if row["fights"] > 0:
            bar = "█" * int(row["accuracy"] * 30)
            calibration = "✅ well calibrated" if abs(row["accuracy"] - (float(str(row["conf_bucket"]).split("-")[0].rstrip("%")) / 100)) < 0.1 else "⚠️ miscalibrated"
            print(f"  {row['conf_bucket']:8s}  {row['accuracy']:.1%}  ({int(row['fights'])} fights)  {calibration}")
    print()

    # By event — rolling accuracy over time
    print("  ACCURACY BY EVENT (most recent last)")
    print("  " + "─" * 40)
    event_stats = df.groupby("event").agg(
        fights=("correct", "count"),
        accuracy=("correct", "mean")
    ).reset_index()
    for _, row in event_stats.iterrows():
        bar = "█" * int(row["accuracy"] * 20)
        trend = "📈" if row["accuracy"] >= accuracy else "📉"
        print(f"  {row['accuracy']:.1%}  ({int(row['fights'])} fights)  {bar} {trend}  {row['event'][-40:]}")
    print()

    # Wrong predictions — what did we miss?
    wrong = df[df["correct"] == 0].sort_values("confidence", ascending=False)
    if len(wrong) > 0:
        print(f"  HIGH-CONFIDENCE MISSES (top {min(5, len(wrong))} by confidence)")
        print("  " + "─" * 40)
        for _, row in wrong.head(5).iterrows():
            print(f"  {row['confidence']:.0%} conf  {row['predicted_winner']} ❌  "
                  f"(actual: {row['actual_winner']})  [{row['event'][-30:]}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Log and report live prediction accuracy")
    parser.add_argument("--event", type=str, help="Event name to score (partial match ok)")
    parser.add_argument("--report", action="store_true", help="Print accuracy report")
    parser.add_argument("--events", type=int, default=None, help="Limit report to last N events")
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
