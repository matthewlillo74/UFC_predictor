"""
scripts/train_model.py
───────────────────────
Builds the training dataset from the DB and trains all prediction models.

Run this after loading historical data with load_historical_data.py.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --eval-only    # just print metrics on saved model
    python scripts/train_model.py --save-dataset # also save dataset to CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from loguru import logger

from src.database import init_db, get_session
from src.features.feature_builder import build_training_dataset
from src.models.predict import UFCPredictor
from config import PROCESSED_DIR


def main(eval_only: bool = False, save_dataset: bool = False):
    init_db()
    session = get_session()

    # ── Step 1: Build dataset ─────────────────────────────────────────────────
    dataset_path = PROCESSED_DIR / "training_dataset.csv"

    if dataset_path.exists() and not eval_only:
        logger.info(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, parse_dates=["fight_date"])
    else:
        logger.info("Building dataset from database...")
        df = build_training_dataset(session)

        if save_dataset or True:  # always save
            df.to_csv(dataset_path, index=False)
            logger.success(f"Dataset saved to {dataset_path}")

    session.close()

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Date range: {df.fight_date.min().date()} → {df.fight_date.max().date()}")
    logger.info(f"Fighter A win rate: {df.winner.mean():.1%}")
    logger.info(f"Method breakdown:\n{df.method.value_counts()}")

    if eval_only:
        predictor = UFCPredictor()
        predictor.load()
        # Use last 200 fights as test set
        df_test = df.sort_values("fight_date").tail(200)
        metrics = predictor.evaluate(df_test)
        print("\n── Model Evaluation ──────────────────")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    # ── Step 2: Time-based train/test split ───────────────────────────────────
    # Use last 15% of fights as test set — never shuffle
    split_idx = int(len(df) * 0.85)
    df_sorted = df.sort_values("fight_date").reset_index(drop=True)
    df_train = df_sorted.iloc[:split_idx]
    df_test  = df_sorted.iloc[split_idx:]

    logger.info(f"Train: {len(df_train)} fights  |  Test: {len(df_test)} fights")
    logger.info(f"Train range: {df_train.fight_date.min().date()} → {df_train.fight_date.max().date()}")
    logger.info(f"Test range:  {df_test.fight_date.min().date()} → {df_test.fight_date.max().date()}")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    predictor = UFCPredictor()
    predictor.train(df_train)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    metrics = predictor.evaluate(df_test)

    print("\n" + "═" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("═" * 50)
    print(f"  Test fights:   {metrics['n_fights']}")
    print(f"  Accuracy:      {metrics['accuracy']:.1%}")
    print(f"  Log loss:      {metrics['log_loss']:.4f}")
    print(f"  Brier score:   {metrics['brier_score']:.4f}")
    print("═" * 50)
    print()

    # Baseline comparison: always pick the fighter listed first (fighter_a)
    baseline_acc = df_test["winner"].mean()
    print(f"  Baseline (always pick fighter_a): {baseline_acc:.1%}")
    print(f"  Model improvement: +{(metrics['accuracy'] - baseline_acc):.1%}")
    print()

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    predictor.save()
    logger.success("Training complete. Models saved.")

    # ── Step 6: Feature importance ────────────────────────────────────────────
    print("  TOP FEATURE IMPORTANCES")
    print("  " + "─" * 40)
    importances = predictor.winner_model.feature_importances_
    from config import FEATURE_COLUMNS
    pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in pairs[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-dataset", action="store_true")
    args = parser.parse_args()
    main(eval_only=args.eval_only, save_dataset=args.save_dataset)
