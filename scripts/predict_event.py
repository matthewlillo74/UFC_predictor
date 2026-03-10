"""
scripts/predict_event.py
─────────────────────────
Run predictions for an upcoming UFC event.

Fetches the next card, pulls odds, generates predictions with explanations,
detects value bets, stores everything to DB, and prints a full report.

Usage:
    python scripts/predict_event.py                    # next upcoming event
    python scripts/predict_event.py --event "UFC 309"  # specific event
    python scripts/predict_event.py --no-odds          # skip odds (no API key needed)
    python scripts/predict_event.py --card "Islam Makhachev,Charles Oliveira"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from loguru import logger

from src.database import init_db, get_session, Fighter, Fight, Event, Prediction
from src.features.feature_builder import FeatureBuilder
from src.models.predict import UFCPredictor
from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights, search_fighter
from src.ingestion.odds_scraper import fetch_and_store_odds, parse_odds_response, fetch_mma_odds
from src.betting.value_detector import analyze_fight_value
from src.explainability.report_generator import ReportGenerator
from src.ingestion.data_loader import get_or_create_fighter, normalize_name
from rapidfuzz import process, fuzz


def get_upcoming_card(session, event_name: str = None) -> tuple:
    """Returns (event_name, fight_date, list of (fighter_a_id, fighter_b_id))"""

    upcoming = get_upcoming_events()
    if not upcoming:
        logger.error("No upcoming events found")
        return None, None, []

    if event_name:
        event_data = next((e for e in upcoming if event_name.lower() in e["name"].lower()), None)
        if not event_data:
            logger.error(f"Event not found: {event_name}")
            return None, None, []
    else:
        event_data = upcoming[0]

    logger.info(f"Loading card: {event_data['name']}")
    fights_raw = get_event_fights(event_data["url"])

    fight_pairs = []
    for f in fights_raw:
        fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
        fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))

        # Create Fight record if it doesn't exist
        existing = session.query(Fight).filter_by(
            fighter_a_id=fa.id, fighter_b_id=fb.id
        ).first()
        if not existing:
            existing = session.query(Fight).filter_by(
                fighter_a_id=fb.id, fighter_b_id=fa.id
            ).first()

        if not existing:
            fight_obj = Fight(
                fighter_a_id=fa.id,
                fighter_b_id=fb.id,
                fight_date=event_data["date"] or datetime.utcnow(),
                weight_class=f.get("weight_class", ""),
                is_title_fight=f.get("is_title_fight", False),
            )
            session.add(fight_obj)
            session.flush()
        else:
            fight_obj = existing

        fight_pairs.append((fa, fb, fight_obj))

    session.commit()
    return event_data["name"], event_data.get("date"), fight_pairs


def run_predictions(event_name=None, manual_card=None, use_odds=True):
    init_db()
    session = get_session()

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        logger.error(f"Could not load model: {e}. Run train_model.py first.")
        return

    builder = FeatureBuilder(session)
    report_gen = ReportGenerator()

    # ── Get card ──────────────────────────────────────────────────────────────
    if manual_card:
        # Manual card: "Fighter A,Fighter B|Fighter C,Fighter D"
        pairs_raw = [p.strip().split(",") for p in manual_card.split("|")]
        all_fighters = session.query(Fighter).all()
        name_map = {normalize_name(f.name): f for f in all_fighters}
        norms = list(name_map.keys())

        fight_pairs = []
        for pair in pairs_raw:
            if len(pair) != 2:
                continue
            def find(name):
                m = process.extractOne(normalize_name(name.strip()), norms, scorer=fuzz.token_sort_ratio, score_cutoff=60)
                return name_map[m[0]] if m else None

            fa = find(pair[0])
            fb = find(pair[1])
            if fa and fb:
                fight_obj = Fight(
                    fighter_a_id=fa.id, fighter_b_id=fb.id,
                    fight_date=datetime.utcnow()
                )
                session.add(fight_obj)
                session.flush()
                fight_pairs.append((fa, fb, fight_obj))
            else:
                logger.warning(f"Could not find fighters: {pair}")

        session.commit()
        card_name = "Manual Card"
        fight_date = datetime.utcnow()
    else:
        card_name, fight_date, fight_pairs = get_upcoming_card(session, event_name)
        if not fight_pairs:
            logger.error("No fights found on card")
            return

    # ── Fetch odds ────────────────────────────────────────────────────────────
    odds_map = {}
    if use_odds:
        try:
            raw_odds = fetch_mma_odds()
            if raw_odds:
                parsed_odds = parse_odds_response(raw_odds)
                for o in parsed_odds:
                    key_ab = (normalize_name(o["fighter_a"]), normalize_name(o["fighter_b"]))
                    key_ba = (normalize_name(o["fighter_b"]), normalize_name(o["fighter_a"]))
                    odds_map[key_ab] = o
                    odds_map[key_ba] = {"odds_a": o["odds_b"], "odds_b": o["odds_a"],
                                        "fair_prob_a": o["fair_prob_b"], "fair_prob_b": o["fair_prob_a"],
                                        "fighter_a": o["fighter_b"], "fighter_b": o["fighter_a"]}
        except Exception as e:
            logger.warning(f"Could not fetch odds: {e}")

    # ── Generate predictions ──────────────────────────────────────────────────
    all_predictions = []
    all_value_analyses = []

    print(f"\n{'═'*62}")
    print(f"  {card_name.upper()}")
    if fight_date:
        print(f"  {fight_date.strftime('%B %d, %Y') if fight_date else 'TBD'}")
    print(f"{'═'*62}\n")

    for fa, fb, fight_obj in fight_pairs:
        try:
            features = builder.build_matchup_features(
                fa.id, fb.id,
                fight_obj.fight_date or datetime.utcnow()
            )

            prediction = predictor.predict(features, fa.name, fb.name)
            all_predictions.append(prediction)

            # Find odds for this fight
            key = (normalize_name(fa.name), normalize_name(fb.name))
            fight_odds = None
            for ok, ov in odds_map.items():
                if fuzz.token_sort_ratio(ok[0], key[0]) > 80 and fuzz.token_sort_ratio(ok[1], key[1]) > 80:
                    fight_odds = ov
                    break

            # Value analysis
            value_analysis = None
            if fight_odds:
                value_analysis = analyze_fight_value(
                    fa.name, fb.name,
                    prediction["prob_fighter_a"],
                    fight_odds["odds_a"],
                    fight_odds["odds_b"],
                )
                all_value_analyses.append(value_analysis)

            # Store prediction to DB
            existing_pred = session.query(Prediction).filter_by(fight_id=fight_obj.id).first()
            if not existing_pred:
                methods = prediction.get("method_probabilities", {})
                rounds = prediction.get("round_probabilities", {})
                winner_id = fa.id if prediction["prob_fighter_a"] > 0.5 else fb.id

                pred_row = Prediction(
                    fight_id=fight_obj.id,
                    model_version=prediction["model_version"],
                    predicted_at=datetime.utcnow(),
                    prob_fighter_a=prediction["prob_fighter_a"],
                    prob_fighter_b=prediction["prob_fighter_b"],
                    predicted_winner_id=winner_id,
                    prob_ko_tko=methods.get("ko_tko"),
                    prob_submission=methods.get("submission"),
                    prob_decision=methods.get("decision"),
                    prob_under_2_5=rounds.get("under_2_5"),
                    prob_goes_distance=rounds.get("over_2_5"),
                    confidence_score=prediction["confidence"],
                    upset_score=value_analysis["upset_score"] if value_analysis else None,
                    prediction_narrative=str(prediction.get("explanation", "")),
                )
                session.add(pred_row)

            # Print fight report
            report = report_gen.generate_fight_report(
                prediction=prediction,
                odds_data=fight_odds,
                event_name=card_name,
                fight_date=fight_date,
                weight_class=fight_obj.weight_class or "",
            )
            print(report)

        except Exception as e:
            logger.error(f"Failed prediction for {fa.name} vs {fb.name}: {e}")
            import traceback; traceback.print_exc()

    session.commit()

    # ── Value summary ─────────────────────────────────────────────────────────
    if all_value_analyses:
        value_picks = [v for v in all_value_analyses if v["has_value"]]
        if value_picks:
            value_picks.sort(key=lambda x: max(
                x["value_a"]["edge"], x["value_b"]["edge"]
            ), reverse=True)

            print(f"\n{'═'*62}")
            print("  TOP VALUE PICKS")
            print(f"{'═'*62}")
            for v in value_picks[:5]:
                pick = v["best_value_pick"]
                edge = max(v["value_a"]["edge"], v["value_b"]["edge"])
                model_p = v["model_prob_a"] if pick == v["fighter_a"] else v["model_prob_b"]
                market_p = v["fair_prob_a"] if pick == v["fighter_a"] else v["fair_prob_b"]
                print(f"  {pick}")
                print(f"    Model: {model_p:.1%}  Market: {market_p:.1%}  Edge: +{edge:.1%}")
            print()

        upset_alerts = [v for v in all_value_analyses if v.get("upset_alert")]
        if upset_alerts:
            print(f"  UPSET WATCH")
            print(f"  {'─'*40}")
            for v in upset_alerts:
                print(f"  {v['upset_fighter']}  (upset score: {v['upset_score']:+.3f})")
            print()

    session.close()
    logger.success(f"Predictions complete. Stored to database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, default=None, help="Event name (partial match)")
    parser.add_argument("--card", type=str, default=None, help="Manual card: 'A,B|C,D'")
    parser.add_argument("--no-odds", action="store_true", help="Skip odds fetching")
    args = parser.parse_args()

    run_predictions(
        event_name=args.event,
        manual_card=args.card,
        use_odds=not args.no_odds,
    )
