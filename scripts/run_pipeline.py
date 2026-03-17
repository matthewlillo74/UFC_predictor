"""
scripts/run_pipeline.py
────────────────────────
Master automation pipeline. Run this before every UFC event.

Does everything in the right order:
  1. Scrape any new completed events since last run
  2. Enrich any new fighters added
  3. Rebuild training dataset + retrain if 10+ new events
  4. Fetch current odds
  5. Generate predictions + save report to data/predictions/
  6. (Post-event) Score previous predictions against results

Usage:
    python scripts/run_pipeline.py                  # full pre-event run
    python scripts/run_pipeline.py --post-event     # score after event
    python scripts/run_pipeline.py --full-retrain   # force retrain
    python scripts/run_pipeline.py --no-odds        # skip odds fetch

Cron (every Tue + Fri at 9am):
    0 9 * * 2,5 cd /path/to/UFC_predictor && venv/bin/python scripts/run_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import time
from datetime import datetime
from loguru import logger

from src.database import init_db, get_session, Event, Fighter, Fight, Prediction, FighterStats
from config import PREDICTIONS_DIR


# ── Step 1: Scrape new events ─────────────────────────────────────────────────

def step_scrape_new_events(session) -> int:
    from src.ingestion.fight_scraper import get_all_events, get_event_fights
    from src.ingestion.data_loader import _load_fight
    from src.database import Event as EventModel
    from tqdm import tqdm

    latest = session.query(Event).filter(Event.date != None).order_by(Event.date.desc()).first()
    if latest:
        logger.info(f"DB latest: {latest.name} ({latest.date.date()})")

    all_events = list(reversed(get_all_events()))  # oldest first
    new_events = [
        e for e in all_events
        if e["date"] and (not latest or e["date"] > latest.date)
    ]

    if not new_events:
        logger.info("No new events to load")
        return 0

    logger.info(f"Loading {len(new_events)} new events...")
    elo_ratings = {}
    loaded = 0

    for event_data in tqdm(new_events, desc="New events"):
        existing = session.query(EventModel).filter_by(name=event_data["name"]).first()
        if existing:
            continue

        event = EventModel(
            name=event_data["name"],
            date=event_data["date"],
            location=event_data.get("location", ""),
            is_ppv="Fight Night" not in event_data["name"],
        )
        session.add(event)
        session.flush()

        for fight_data in get_event_fights(event_data["url"]):
            try:
                _load_fight(session, event, fight_data, elo_ratings)
            except Exception as e:
                logger.debug(f"Fight skip: {e}")

        session.commit()
        loaded += 1
        time.sleep(1.5)

    # Rebuild stats snapshots for new fights
    if loaded > 0:
        from src.ingestion.data_loader import build_fighter_stats_snapshots
        build_fighter_stats_snapshots(session)

    logger.success(f"Loaded {loaded} new events")
    return loaded


# ── Step 2: Enrich new fighters ───────────────────────────────────────────────

def step_enrich_new_fighters(session) -> int:
    from src.ingestion.fight_scraper import search_fighter, get_fighter
    from tqdm import tqdm

    missing = session.query(Fighter).filter(
        (Fighter.reach_cm == None) | (Fighter.height_cm == None)
    ).all()

    if not missing:
        logger.info("No fighters need enrichment")
        return 0

    logger.info(f"Enriching {len(missing)} fighters...")
    enriched = 0

    for fighter in tqdm(missing, desc="Enriching fighters"):
        try:
            url = search_fighter(fighter.name)
            if not url:
                continue
            data = get_fighter(url)
            if not data:
                continue

            for attr in ["height_cm", "reach_cm", "stance", "date_of_birth"]:
                if data.get(attr):
                    setattr(fighter, attr, data[attr])

            latest_stats = (
                session.query(FighterStats)
                .filter_by(fighter_id=fighter.id)
                .order_by(FighterStats.as_of_date.desc())
                .first()
            )
            if latest_stats:
                for attr in ["slpm", "strike_accuracy", "sapm", "strike_defense",
                             "td_avg", "td_accuracy", "td_defense", "sub_avg"]:
                    if data.get(attr) is not None:
                        setattr(latest_stats, attr, data[attr])

            session.commit()
            enriched += 1
            time.sleep(1.5)
        except Exception as e:
            logger.debug(f"Enrich failed {fighter.name}: {e}")
            session.rollback()

    logger.success(f"Enriched {enriched} fighters")
    return enriched


# ── Step 2b: Compute style features ──────────────────────────────────────────

def step_compute_styles() -> bool:
    """Compute style fingerprints, rolling windows, and momentum scores."""
    logger.info("Computing style features...")
    result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "compute_styles.py")])
    if result.returncode == 0:
        logger.success("Style features computed")
    else:
        logger.error("Style computation failed")
    return result.returncode == 0


def step_compute_vulnerability() -> bool:
    """Compute opponent style vulnerability features."""
    logger.info("Computing style vulnerability features...")
    script = os.path.join(os.path.dirname(__file__), "compute_style_vulnerability.py")
    result = subprocess.run([sys.executable, script])
    if result.returncode == 0:
        logger.success("Style vulnerability computed")
    else:
        logger.warning("Style vulnerability computation failed (non-critical)")
    return result.returncode == 0


# ── Step 3: Retrain ───────────────────────────────────────────────────────────

def step_retrain(force: bool = False, new_events: int = 0) -> bool:
    if not force and new_events < 10:
        logger.info(f"{new_events} new events — skipping retrain (need 10+, or use --full-retrain)")
        return False

    csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "training_dataset.csv")
    if os.path.exists(csv):
        os.remove(csv)

    logger.info("Retraining model...")
    result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_model.py")])
    success = result.returncode == 0
    if success:
        logger.success("Model retrained")
    else:
        logger.error("Retrain failed")
    return success


# ── Step 4: Fetch odds ────────────────────────────────────────────────────────

def step_fetch_odds(session, skip: bool = False) -> list:
    if skip:
        return []
    try:
        from src.ingestion.odds_scraper import fetch_and_store_odds
        odds = fetch_and_store_odds(session)
        logger.info(f"Fetched odds for {len(odds)} fights")
        return odds
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}")
        return []


# ── Step 5: Predict next event ────────────────────────────────────────────────

def step_predict_next_event(session, odds_data: list) -> str:
    from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights
    from src.ingestion.data_loader import get_or_create_fighter, normalize_name
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from src.explainability.report_generator import ReportGenerator
    from rapidfuzz import fuzz

    upcoming = get_upcoming_events()
    if not upcoming:
        logger.warning("No upcoming events")
        return ""

    event_data = upcoming[0]
    event_name = event_data["name"]
    fight_date = event_data.get("date") or datetime.utcnow()
    logger.info(f"Predicting: {event_name}")

    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        logger.error(f"Model load failed: {e}. Run train_model.py first.")
        return ""

    builder = FeatureBuilder(session)
    report_gen = ReportGenerator()

    # Build odds lookup
    odds_map = {}
    for o in odds_data:
        key = (normalize_name(o.get("fighter_a", "")), normalize_name(o.get("fighter_b", "")))
        odds_map[key] = o

    report_lines = [
        f"\n{'═'*62}",
        f"  {event_name.upper()}",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"{'═'*62}\n",
    ]
    value_picks = []

    for f in get_event_fights(event_data["url"]):
        try:
            fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
            fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))

            fight_obj = (
                session.query(Fight).filter_by(fighter_a_id=fa.id, fighter_b_id=fb.id).first() or
                session.query(Fight).filter_by(fighter_a_id=fb.id, fighter_b_id=fa.id).first()
            )
            if not fight_obj:
                fight_obj = Fight(
                    fighter_a_id=fa.id, fighter_b_id=fb.id,
                    fight_date=fight_date, weight_class=f.get("weight_class", "")
                )
                session.add(fight_obj)
                session.flush()

            features = builder.build_matchup_features(fa.id, fb.id, fight_date)
            pred = predictor.predict(features, fa.name, fb.name)

            # Match odds
            fight_odds = None
            for (ok_a, ok_b), ov in odds_map.items():
                if (fuzz.token_sort_ratio(ok_a, normalize_name(fa.name)) > 80 and
                        fuzz.token_sort_ratio(ok_b, normalize_name(fb.name)) > 80):
                    fight_odds = ov
                    break

            # Store prediction (skip if already stored)
            if not session.query(Prediction).filter_by(fight_id=fight_obj.id).first():
                methods = pred.get("method_probabilities", {})
                rounds = pred.get("round_probabilities", {})
                session.add(Prediction(
                    fight_id=fight_obj.id,
                    model_version=pred["model_version"],
                    predicted_at=datetime.utcnow(),
                    prob_fighter_a=pred["prob_fighter_a"],
                    prob_fighter_b=pred["prob_fighter_b"],
                    predicted_winner_id=fa.id if pred["prob_fighter_a"] > 0.5 else fb.id,
                    prob_ko_tko=methods.get("ko_tko"),
                    prob_submission=methods.get("submission"),
                    prob_decision=methods.get("decision"),
                    prob_under_2_5=rounds.get("under_2_5"),
                    prob_goes_distance=rounds.get("over_2_5"),
                    confidence_score=pred["confidence"],
                    prediction_narrative=str(pred.get("explanation", "")),
                ))

            report_lines.append(report_gen.generate_fight_report(
                prediction=pred, odds_data=fight_odds,
                event_name=event_name, fight_date=event_data.get("date"),
                weight_class=f.get("weight_class", ""),
            ))

            if fight_odds:
                edge = pred["prob_fighter_a"] - fight_odds.get("fair_prob_a", 0.5)
                if abs(edge) > 0.07:
                    value_picks.append({
                        "fighter": fa.name if edge > 0 else fb.name,
                        "edge": abs(edge),
                        "model_prob": pred["prob_fighter_a"] if edge > 0 else pred["prob_fighter_b"],
                    })

        except Exception as e:
            logger.error(f"Prediction failed {f.get('fighter_a_name')} vs {f.get('fighter_b_name')}: {e}")

    session.commit()

    if value_picks:
        value_picks.sort(key=lambda x: x["edge"], reverse=True)
        report_lines += [f"\n{'═'*62}", "  TOP VALUE PICKS", f"{'═'*62}"]
        for i, p in enumerate(value_picks, 1):
            report_lines.append(f"  {i}. {p['fighter']}  (Model: {p['model_prob']:.1%}  Edge: +{p['edge']:.1%})")

    report = "\n".join(report_lines)
    print(report)

    fname = f"{event_name.replace(' ','_').replace(':','').replace('/','_')}_{datetime.utcnow().strftime('%Y%m%d')}.txt"
    path = PREDICTIONS_DIR / fname
    path.write_text(report)
    logger.success(f"Report saved: {path}")
    return str(path)


# ── Step 6: Score previous predictions ───────────────────────────────────────

def step_score_predictions(session):
    from src.evaluation.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker(session)
    tracker.update_outcomes()
    tracker.print_report()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UFC Predictor pipeline")
    parser.add_argument("--post-event", action="store_true", help="Score predictions after event completes")
    parser.add_argument("--full-retrain", action="store_true", help="Force full dataset rebuild and retrain")
    parser.add_argument("--no-odds", action="store_true", help="Skip odds fetch (no API key needed)")
    args = parser.parse_args()

    print("\n" + "═"*50)
    print("  UFC PREDICTOR PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═"*50 + "\n")

    init_db()
    session = get_session()

    if args.post_event:
        logger.info("Post-event mode: scoring predictions")
        step_score_predictions(session)
    else:
        logger.info("Pre-event mode: scrape → enrich → styles → vulnerability → (retrain) → predict")
        new_events = step_scrape_new_events(session)
        if new_events > 0:
            step_enrich_new_fighters(session)
            step_compute_styles()
            step_compute_vulnerability()
        step_retrain(force=args.full_retrain, new_events=new_events)
        odds = step_fetch_odds(session, skip=args.no_odds)
        step_predict_next_event(session, odds)

    session.close()
    print("\n" + "═"*50)
    print("  PIPELINE COMPLETE")
    print("═"*50 + "\n")


if __name__ == "__main__":
    main()
