"""
scripts/analyze_shap_misses.py
────────────────────────────────
Cross-references SHAP explanations against high-confidence live misses to find
systematic feature-level failure patterns.

SHAP was previously computed per-prediction for display only (dashboard fight
cards) — nothing aggregated it across misses to check whether they share a
feature-level signature. This closes that loop: for every high-confidence
prediction that was wrong, recompute SHAP values and tally which features
consistently pushed the model TOWARD its incorrect pick.

Sources misses from data/predictions/live_accuracy.csv rather than the
Prediction DB table — the DB table only has 2 rows historically (predictions
got orphaned by the duplicate-Fight-row bug fixed earlier this session), while
the CSV log is what log_live_results.py's own reporting has relied on all
along and has the full history (107 fights as of 2026-07-15).

Usage:
    python scripts/analyze_shap_misses.py                    # default: conf >= 0.65
    python scripts/analyze_shap_misses.py --min-confidence 0.70
    python scripts/analyze_shap_misses.py --top 15
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
import pandas as pd
from loguru import logger

from src.database import init_db, get_session, Fighter
from src.features.feature_builder import FeatureBuilder
from src.ingestion.data_loader import normalize_name
from src.models.predict import UFCPredictor
from config import FEATURE_COLUMNS, PREDICTIONS_DIR


def _find_fighter(session, name_map, name: str):
    from rapidfuzz import process, fuzz
    norm = normalize_name(name)
    if norm in name_map:
        return name_map[norm]
    match = process.extractOne(norm, list(name_map.keys()), scorer=fuzz.token_sort_ratio, score_cutoff=85)
    return name_map[match[0]] if match else None


def analyze(min_confidence: float = 0.65, top_n: int = 15):
    log_path = PREDICTIONS_DIR / "live_accuracy.csv"
    if not log_path.exists():
        logger.warning(f"No live accuracy log at {log_path}")
        return None

    df = pd.read_csv(log_path)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["winner_correct"] = pd.to_numeric(df["winner_correct"], errors="coerce")

    misses = df[(df["winner_correct"] == 0) & (df["confidence"] >= min_confidence)]
    if misses.empty:
        logger.warning(f"No misses found at confidence >= {min_confidence:.0%}")
        return None

    logger.info(f"Analyzing {len(misses)} high-confidence misses (>= {min_confidence:.0%})...")

    init_db()
    session = get_session()
    name_map = {normalize_name(f.name): f for f in session.query(Fighter).all()}

    predictor = UFCPredictor()
    predictor.load()
    builder = FeatureBuilder(session)

    feature_push_toward_wrong_pick = defaultdict(list)
    skipped = 0

    for _, row in misses.iterrows():
        try:
            fa = _find_fighter(session, name_map, row["fighter_a"])
            fb = _find_fighter(session, name_map, row["fighter_b"])
            if not fa or not fb:
                raise ValueError(f"Could not resolve fighters: {row['fighter_a']} / {row['fighter_b']}")

            fight_date = pd.to_datetime(row["fight_date"])
            features = builder.build_matchup_features(
                fa.id, fb.id, fight_date, fight_weight_class=row.get("weight_class"),
            )
            X = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)
            shap_values = predictor.shap_explainer.shap_values(X)[0]
        except Exception as e:
            logger.debug(f"Skipping {row.get('fighter_a')} vs {row.get('fighter_b')}: {e}")
            skipped += 1
            continue

        # Positive raw SHAP pushes toward fighter_a. Flip sign if the model's
        # wrong pick was fighter_b, so "push" is always in the direction of
        # the mistaken prediction regardless of which side it favored.
        predicted_a = normalize_name(row["predicted_winner"]) == normalize_name(row["fighter_a"])
        for feat, val in zip(FEATURE_COLUMNS, shap_values):
            push = val if predicted_a else -val
            if push > 0:
                feature_push_toward_wrong_pick[feat].append(float(push))

    if skipped:
        logger.warning(f"Skipped {skipped}/{len(misses)} misses (fighter resolution or feature rebuild failed)")

    if not feature_push_toward_wrong_pick:
        logger.warning("No misses could be analyzed — nothing to report")
        return None

    rows = [
        {
            "feature": feat,
            "n_misses_pushed_wrong": len(pushes),
            "avg_wrong_push": sum(pushes) / len(pushes),
            "total_wrong_push": sum(pushes),
        }
        for feat, pushes in feature_push_toward_wrong_pick.items()
    ]
    report_df = pd.DataFrame(rows).sort_values("total_wrong_push", ascending=False)

    print("\n" + "=" * 70)
    print(f"  SHAP MISS-PATTERN ANALYSIS — {len(misses) - skipped} misses, confidence >= {min_confidence:.0%}")
    print("=" * 70)
    print(f"  {'Feature':<28} {'# misses':>10} {'avg push':>10} {'total push':>12}")
    print("  " + "-" * 66)
    for _, row in report_df.head(top_n).iterrows():
        print(f"  {row['feature']:<28} {row['n_misses_pushed_wrong']:>10} "
              f"{row['avg_wrong_push']:>10.4f} {row['total_wrong_push']:>12.4f}")
    print()
    print("  Interpretation: features here consistently pushed the model TOWARD")
    print("  its wrong pick across high-confidence misses. High total push + high")
    print("  miss count = a systematic blind spot worth a targeted feature fix,")
    print("  not just noise from one or two unlucky fights.")
    print("=" * 70 + "\n")

    session.close()
    return report_df


def main():
    parser = argparse.ArgumentParser(description="SHAP-based aggregate miss-pattern analysis")
    parser.add_argument("--min-confidence", type=float, default=0.65,
                         help="Only analyze misses at or above this model confidence (default 0.65)")
    parser.add_argument("--top", type=int, default=15, help="Number of features to show (default 15)")
    args = parser.parse_args()
    analyze(min_confidence=args.min_confidence, top_n=args.top)


if __name__ == "__main__":
    main()
