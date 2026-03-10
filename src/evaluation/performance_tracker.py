"""
src/evaluation/performance_tracker.py
───────────────────────────────────────
Tracks model performance over time.

Compares stored predictions (made BEFORE fights) against actual results.
Measures: winner accuracy, method accuracy, calibration, simulated ROI.

This is what separates a serious prediction system from a toy.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from datetime import datetime
from typing import Optional
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session


class PerformanceTracker:
    """
    Measures prediction accuracy and betting ROI over time.

    Usage:
        tracker = PerformanceTracker(session)
        tracker.update_outcomes()        # call after each event
        summary = tracker.get_summary()
        tracker.print_report()
    """

    def __init__(self, db_session: Session):
        self.session = db_session

    def update_outcomes(self, event_id: int = None):
        """
        After an event completes, match stored predictions against results.
        Updates was_correct, method_correct, round_correct on each Prediction.

        Args:
            event_id: If provided, only update predictions for this event.
                      If None, update all unresolved predictions.
        """
        from src.database import Prediction, Fight

        query = (
            self.session.query(Prediction)
            .join(Fight, Prediction.fight_id == Fight.id)
            .filter(Prediction.was_correct == None)  # unresolved
            .filter(Fight.winner_id != None)          # fight has result
        )

        if event_id:
            query = query.filter(Fight.event_id == event_id)

        predictions = query.all()
        updated = 0

        for pred in predictions:
            fight = self.session.query(Fight).get(pred.fight_id)
            if not fight or fight.winner_id is None:
                continue

            # Winner correct?
            predicted_winner_id = pred.predicted_winner_id
            pred.was_correct = (predicted_winner_id == fight.winner_id)

            # Method correct?
            if pred.prob_ko_tko and fight.method:
                method_map = {"KO_TKO": "ko_tko", "Submission": "submission", "Decision": "decision"}
                predicted_method = max(
                    method_map,
                    key=lambda m: getattr(pred, f"prob_{method_map[m]}", 0) or 0
                )
                pred.method_correct = (predicted_method == fight.method)

            # Round correct? (did we correctly predict over/under 2.5?)
            if pred.prob_under_2_5 and fight.finish_round:
                pred_under = pred.prob_under_2_5 > 0.5
                actual_under = fight.finish_round <= 2
                pred.round_correct = (pred_under == actual_under)

            updated += 1

        self.session.commit()
        logger.info(f"Updated {updated} prediction outcomes")

    def get_summary(self, since: Optional[datetime] = None) -> dict:
        """
        Return overall accuracy metrics.
        """
        from src.database import Prediction, Fight

        query = (
            self.session.query(Prediction)
            .join(Fight)
            .filter(Prediction.was_correct != None)
        )

        if since:
            query = query.filter(Fight.fight_date >= since)

        preds = query.all()

        if not preds:
            return {"error": "No resolved predictions found"}

        total = len(preds)
        correct = sum(1 for p in preds if p.was_correct)
        method_preds = [p for p in preds if p.method_correct is not None]
        method_correct = sum(1 for p in method_preds if p.method_correct)

        return {
            "total_predictions": total,
            "winner_correct":    correct,
            "winner_accuracy":   round(correct / total, 4),
            "method_predictions": len(method_preds),
            "method_accuracy":   round(method_correct / len(method_preds), 4) if method_preds else None,
            "avg_confidence":    round(sum(p.confidence_score or 0 for p in preds) / total, 4),
        }

    def get_calibration_data(self) -> pd.DataFrame:
        """
        Check if predicted probabilities match actual win rates.

        A well-calibrated model's 65% predictions should win ~65% of the time.
        Returns DataFrame showing predicted vs actual win rates by probability bucket.
        """
        from src.database import Prediction

        preds = self.session.query(Prediction).filter(Prediction.was_correct != None).all()
        if not preds:
            return pd.DataFrame()

        rows = [{
            "prob": p.prob_fighter_a,
            "correct": int(p.was_correct),
        } for p in preds]

        df = pd.DataFrame(rows)
        df["bucket"] = pd.cut(df["prob"], bins=[0, .45, .50, .55, .60, .65, .70, .75, 1.0])
        summary = df.groupby("bucket").agg(
            count=("correct", "count"),
            actual_win_rate=("correct", "mean"),
            avg_prob=("prob", "mean"),
        ).reset_index()

        return summary

    def simulate_roi(
        self,
        stake_per_bet: float = 100.0,
        min_edge: float = 0.05,
    ) -> dict:
        """
        Simulate betting ROI using stored predictions + odds.

        Only bets when model edge exceeds min_edge threshold.
        """
        from src.database import Prediction, Fight, BettingOdds
        from src.betting.value_detector import american_to_prob

        results = (
            self.session.query(Prediction, Fight, BettingOdds)
            .join(Fight, Prediction.fight_id == Fight.id)
            .join(BettingOdds, BettingOdds.fight_id == Fight.id)
            .filter(Prediction.was_correct != None)
            .filter(BettingOdds.is_closing == True)
            .all()
        )

        if not results:
            # Fall back to any odds
            results = (
                self.session.query(Prediction, Fight, BettingOdds)
                .join(Fight, Prediction.fight_id == Fight.id)
                .join(BettingOdds, BettingOdds.fight_id == Fight.id)
                .filter(Prediction.was_correct != None)
                .all()
            )

        bets = []
        for pred, fight, odds in results:
            model_prob = pred.prob_fighter_a
            market_prob = odds.implied_prob_a or 0.5
            edge = model_prob - market_prob

            if edge >= min_edge:
                # Bet on fighter A
                if pred.was_correct:
                    # Payout based on American odds
                    o = odds.odds_fighter_a
                    payout = stake_per_bet * (100 / abs(o)) if o < 0 else stake_per_bet * (o / 100)
                    profit = payout
                else:
                    profit = -stake_per_bet
                bets.append({"fighter": "A", "edge": edge, "profit": profit, "won": pred.was_correct})

        if not bets:
            return {"error": "No value bets found with current settings"}

        total_staked = len(bets) * stake_per_bet
        total_profit = sum(b["profit"] for b in bets)
        wins = sum(1 for b in bets if b["won"])

        return {
            "total_bets":    len(bets),
            "wins":          wins,
            "losses":        len(bets) - wins,
            "total_staked":  round(total_staked, 2),
            "total_profit":  round(total_profit, 2),
            "roi_pct":       round((total_profit / total_staked) * 100, 2),
            "win_rate":      round(wins / len(bets), 4),
            "min_edge_used": min_edge,
        }

    def print_report(self):
        """Print a formatted performance report to console."""
        summary = self.get_summary()

        print("\n" + "═" * 50)
        print("  PREDICTION PERFORMANCE REPORT")
        print("═" * 50)

        if "error" in summary:
            print(f"  {summary['error']}")
            print("  Run update_outcomes() after fights complete.")
        else:
            print(f"  Total predictions:  {summary['total_predictions']}")
            print(f"  Winner accuracy:    {summary['winner_accuracy']:.1%}")
            if summary.get("method_accuracy"):
                print(f"  Method accuracy:    {summary['method_accuracy']:.1%}")
            print(f"  Avg confidence:     {summary['avg_confidence']:.1%}")

        roi = self.simulate_roi()
        if "error" not in roi:
            print(f"\n  SIMULATED BETTING ROI")
            print(f"  Total bets:    {roi['total_bets']}")
            print(f"  Win rate:      {roi['win_rate']:.1%}")
            print(f"  ROI:           {roi['roi_pct']:+.1f}%")

        print("═" * 50 + "\n")
