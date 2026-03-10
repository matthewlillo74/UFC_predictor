"""
src/explainability/report_generator.py
────────────────────────────────────────
Generates human-readable prediction reports for UFC events.

This turns raw model output into something that reads like
a proper analyst breakdown, not just a probability score.
"""

from datetime import datetime
from typing import Optional
from loguru import logger


REPORT_TEMPLATE = """
╔══════════════════════════════════════════════════════════════╗
║  UFC PREDICTOR — FIGHT ANALYSIS                              ║
╚══════════════════════════════════════════════════════════════╝

{event_name}
{fight_date}

──────────────────────────────────────────────────────────────
MATCHUP: {fighter_a} vs {fighter_b}
Weight Class: {weight_class}
──────────────────────────────────────────────────────────────

WINNER PREDICTION
  {fighter_a}: {prob_a:.1%}
  {fighter_b}: {prob_b:.1%}
  → Predicted winner: {predicted_winner} ({confidence:.1%} confidence)

METHOD OF VICTORY
  KO / TKO:    {prob_ko:.1%}
  Submission:  {prob_sub:.1%}
  Decision:    {prob_dec:.1%}

ROUNDS
  Fight ends early (under 2.5): {prob_under:.1%}
  Goes the distance:            {prob_over:.1%}

KEY FACTORS FOR {fighter_a}
{factors_a}

KEY FACTORS FOR {fighter_b}
{factors_b}

BETTING VALUE
  Market implied probability ({fighter_a}): {market_prob_a:.1%}
  Model probability ({fighter_a}):          {model_prob_a:.1%}
  Edge: {edge_a:+.1%}
  {value_flag}

UPSET WATCH
  Upset score: {upset_score:+.3f}
  {upset_flag}

──────────────────────────────────────────────────────────────
Prediction generated: {generated_at}
Model version: {model_version}
──────────────────────────────────────────────────────────────
"""


class ReportGenerator:
    """
    Generates fight prediction reports.

    Usage:
        gen = ReportGenerator()
        report = gen.generate_fight_report(prediction, odds_data)
        gen.generate_event_report(event_id, predictions, odds_list)
    """

    def generate_fight_report(
        self,
        prediction: dict,
        odds_data: Optional[dict] = None,
        event_name: str = "UFC Event",
        fight_date: Optional[datetime] = None,
        weight_class: str = "Unknown",
    ) -> str:
        """
        Generate a single fight prediction report.

        Args:
            prediction:  Output from UFCPredictor.predict()
            odds_data:   Optional dict with sportsbook odds info
            event_name:  Event name string
            fight_date:  Date of the fight
        """
        fighter_a = prediction["fighter_a"]
        fighter_b = prediction["fighter_b"]
        prob_a = prediction["prob_fighter_a"]
        prob_b = prediction["prob_fighter_b"]

        methods = prediction.get("method_probabilities", {})
        rounds = prediction.get("round_probabilities", {})
        explanation = prediction.get("explanation", {})

        # Format factor lists
        factors_a = explanation.get(f"factors_favoring_{fighter_a}", [])
        factors_b = explanation.get(f"factors_favoring_{fighter_b}", [])
        fmt_factors_a = "\n".join(f"  + {f}" for f in factors_a) or "  (insufficient data)"
        fmt_factors_b = "\n".join(f"  + {f}" for f in factors_b) or "  (insufficient data)"

        # Betting value
        market_prob_a = odds_data.get("implied_prob_a", 0.5) if odds_data else 0.5
        edge_a = prob_a - market_prob_a
        value_flag = "⚠️  VALUE BET DETECTED" if edge_a > 0.07 else "  No significant edge"
        upset_score = prediction.get("upset_score", 0.0) or 0.0
        upset_flag = "🔥 HIGH UPSET POTENTIAL" if abs(upset_score) > 0.12 else "  Normal favorite/underdog"

        return REPORT_TEMPLATE.format(
            event_name=event_name,
            fight_date=fight_date.strftime("%B %d, %Y") if fight_date else "TBD",
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            weight_class=weight_class,
            prob_a=prob_a,
            prob_b=prob_b,
            predicted_winner=prediction["predicted_winner"],
            confidence=prediction["confidence"],
            prob_ko=methods.get("ko_tko", 0),
            prob_sub=methods.get("submission", 0),
            prob_dec=methods.get("decision", 0),
            prob_under=rounds.get("under_2_5", 0),
            prob_over=rounds.get("over_2_5", 0),
            factors_a=fmt_factors_a,
            factors_b=fmt_factors_b,
            market_prob_a=market_prob_a,
            model_prob_a=prob_a,
            edge_a=edge_a,
            value_flag=value_flag,
            upset_score=upset_score,
            upset_flag=upset_flag,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            model_version=prediction.get("model_version", "unknown"),
        )

    def generate_event_report(
        self,
        event_name: str,
        fight_date: datetime,
        predictions: list[dict],
        odds_list: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a full event prediction report covering all fights on the card.
        Appends a 'Top Value Picks' summary at the end.
        """
        sections = [f"\n{'═' * 62}", f"  {event_name.upper()}", f"  {fight_date.strftime('%B %d, %Y')}", f"{'═' * 62}\n"]

        odds_map = {}
        if odds_list:
            for odds in odds_list:
                key = (odds.get("fighter_a", ""), odds.get("fighter_b", ""))
                odds_map[key] = odds

        value_picks = []

        for pred in predictions:
            key = (pred["fighter_a"], pred["fighter_b"])
            odds = odds_map.get(key)
            report = self.generate_fight_report(
                prediction=pred,
                odds_data=odds,
                event_name=event_name,
                fight_date=fight_date,
            )
            sections.append(report)

            # Collect value picks
            if odds:
                market_prob = odds.get("implied_prob_a", 0.5)
                edge = pred["prob_fighter_a"] - market_prob
                if abs(edge) > 0.07:
                    value_picks.append({
                        "fighter": pred["fighter_a"] if edge > 0 else pred["fighter_b"],
                        "edge": abs(edge),
                        "model_prob": pred["prob_fighter_a"] if edge > 0 else pred["prob_fighter_b"],
                    })

        # Value summary
        if value_picks:
            value_picks.sort(key=lambda x: x["edge"], reverse=True)
            sections.append(f"\n{'═' * 62}")
            sections.append("  TOP VALUE PICKS")
            sections.append(f"{'═' * 62}")
            for i, pick in enumerate(value_picks, 1):
                sections.append(
                    f"  {i}. {pick['fighter']}  "
                    f"(Model: {pick['model_prob']:.1%}  Edge: +{pick['edge']:.1%})"
                )

        return "\n".join(sections)

    def save_report(self, report: str, filename: str):
        """Save report to the predictions directory."""
        from config import PREDICTIONS_DIR
        path = PREDICTIONS_DIR / filename
        path.write_text(report)
        logger.success(f"Report saved to {path}")
        return path
