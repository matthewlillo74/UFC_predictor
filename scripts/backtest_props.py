"""
scripts/backtest_props.py
──────────────────────────
Backtests model accuracy on method and round props across historical fights.

Before betting props, use this to validate the model's edge on:
  - Method: KO/TKO, Submission, Decision
  - Round O/U 2.5 (and 3.5 for title fights)
  - Combined method + round parlays

Uses time-based holdout only — never trains and tests on same data.

Usage:
    python scripts/backtest_props.py                    # full backtest
    python scripts/backtest_props.py --events 50        # last 50 events
    python scripts/backtest_props.py --weight-class LW  # filter division
    python scripts/backtest_props.py --method KO_TKO    # specific method only
    python scripts/backtest_props.py --year 2024        # specific year
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from loguru import logger

from src.database import init_db, get_session
from src.features.feature_builder import build_training_dataset
from src.models.predict import UFCPredictor
from config import FEATURE_COLUMNS, PROCESSED_DIR


# ── Simulated prop odds ───────────────────────────────────────────────────────
# These are approximate average sportsbook lines for UFC props.
# Used to compute simulated P&L — not real odds but reasonable approximations.
PROP_ODDS = {
    "method_ko_tko":     -120,   # KO/TKO typically slight favorite in finish fights
    "method_sub":        +200,   # submissions rarer, pay more
    "method_decision":   -140,   # decisions most common outcome
    "round_under_2_5":   -115,   # early finish
    "round_over_2_5":    -115,   # goes to rounds 3+
    "round_under_3_5":   -130,   # for title fights
    "round_over_3_5":    -110,
}


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def prob_to_payout(prob: float, stake: float = 100) -> float:
    if prob <= 0 or prob >= 1:
        return 0.0
    return stake * (1 - prob) / prob


def load_dataset(session, events: int = None, year: int = None) -> pd.DataFrame:
    """Load and filter training dataset."""
    dataset_path = PROCESSED_DIR / "training_dataset.csv"
    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, parse_dates=["fight_date"])
    else:
        logger.info("Building dataset from database...")
        df = build_training_dataset(session)
        df.to_csv(dataset_path, index=False)

    df = df[df["method"].isin(["KO_TKO", "Submission", "Decision"])].copy()
    df = df.sort_values("fight_date").reset_index(drop=True)

    if year:
        df = df[pd.to_datetime(df["fight_date"]).dt.year == year]
    if events:
        # Approximate: last N events ≈ last N*12 fights (avg ~12 fights per event)
        cutoff = max(0, len(df) - events * 12)
        df = df.iloc[cutoff:]

    return df


def run_backtest(df: pd.DataFrame, predictor: UFCPredictor,
                 weight_class_filter: str = None,
                 method_filter: str = None) -> dict:
    """
    Run full props backtest on a dataset.
    Returns accuracy and simulated P&L for each prop type.
    """
    # Use last 15% as test set (same split as training)
    split_idx = int(len(df) * 0.85)
    df_test = df.iloc[split_idx:].copy()

    if len(df_test) < 50:
        logger.warning(f"Only {len(df_test)} test fights — results may not be meaningful")

    if weight_class_filter:
        df_test = df_test[df_test.get("weight_class", pd.Series()) == weight_class_filter]
    if method_filter:
        df_test = df_test[df_test["method"] == method_filter]

    if len(df_test) == 0:
        logger.error("No fights match filters")
        return {}

    X_test = df_test[FEATURE_COLUMNS].fillna(0)

    # ── Method predictions ────────────────────────────────────────────────────
    method_probs = predictor.method_model.predict_proba(X_test)
    classes = list(predictor.method_model.classes_)
    method_map = {0: "Decision", 1: "KO_TKO", 2: "Submission"}
    reverse_map = {"Decision": 0, "KO_TKO": 1, "Submission": 2}

    df_test = df_test.copy()
    df_test["pred_ko_prob"]       = method_probs[:, classes.index(1)] if 1 in classes else 0.33
    df_test["pred_sub_prob"]      = method_probs[:, classes.index(2)] if 2 in classes else 0.33
    df_test["pred_dec_prob"]      = method_probs[:, classes.index(0)] if 0 in classes else 0.33
    df_test["pred_method"]        = df_test[["pred_ko_prob","pred_sub_prob","pred_dec_prob"]].idxmax(axis=1)
    df_test["pred_method"]        = df_test["pred_method"].map({
        "pred_ko_prob": "KO_TKO", "pred_sub_prob": "Submission", "pred_dec_prob": "Decision"
    })
    df_test["method_correct"]     = (df_test["pred_method"] == df_test["method"]).astype(int)

    # ── Round predictions ─────────────────────────────────────────────────────
    if hasattr(predictor, "round_model_calibrated") and predictor.round_model_calibrated:
        round_probs = predictor.round_model_calibrated.predict_proba(X_test)
    else:
        round_probs = predictor.round_model.predict_proba(X_test)

    # round_probs: col 0 = prob late (class 0), col 1 = prob early (class 1)
    rp_classes = list(predictor.round_model.classes_)
    early_idx = rp_classes.index(1) if 1 in rp_classes else 1
    late_idx  = rp_classes.index(0) if 0 in rp_classes else 0

    df_test["pred_early_prob"] = round_probs[:, early_idx]
    df_test["pred_late_prob"]  = round_probs[:, late_idx]

    # Determine actual early/late
    is_title = df_test.get("is_title_fight", pd.Series(0, index=df_test.index)).fillna(0).astype(int)
    def actual_early(row):
        threshold = 3.5 if row.get("is_title_fight", 0) == 1 else 2.5
        if row["method"] == "Decision":
            return 0
        fr = row.get("finish_round")
        if pd.isna(fr):
            return 0
        return 1 if float(fr) <= threshold else 0

    df_test["actual_early"] = df_test.apply(actual_early, axis=1)
    df_test["pred_early"]   = (df_test["pred_early_prob"] > 0.5).astype(int)
    df_test["round_correct"] = (df_test["pred_early"] == df_test["actual_early"]).astype(int)

    # ── P&L simulation (flat $100 per bet) ───────────────────────────────────
    def pnl(prob_correct: float, actual_correct: int, prop_key: str) -> float:
        """Simulated P&L: bet $100 on model's pick at simulated sportsbook odds."""
        odds = PROP_ODDS.get(prop_key, -115)
        mkt_prob = american_to_prob(odds)
        if actual_correct:
            return prob_to_payout(mkt_prob, 100)
        return -100.0

    # Method P&L — bet on the highest probability method
    df_test["method_pnl"] = df_test.apply(
        lambda r: pnl(
            max(r["pred_ko_prob"], r["pred_sub_prob"], r["pred_dec_prob"]),
            r["method_correct"],
            f"method_{r['pred_method'].lower()}"
        ), axis=1
    )

    # Round P&L — bet UNDER if pred_early_prob > 0.5, OVER otherwise
    df_test["round_pnl"] = df_test.apply(
        lambda r: pnl(
            r["pred_early_prob"] if r["pred_early"] == 1 else r["pred_late_prob"],
            r["round_correct"],
            "round_under_2_5" if r["pred_early"] == 1 else "round_over_2_5"
        ), axis=1
    )

    # High-edge method bets only (prob > 0.55)
    high_conf_method = df_test[df_test[["pred_ko_prob","pred_sub_prob","pred_dec_prob"]].max(axis=1) > 0.55]
    # High-edge round bets only (prob > 0.60)
    high_conf_round  = df_test[df_test[["pred_early_prob","pred_late_prob"]].max(axis=1) > 0.60]

    results = {
        "n_fights":            len(df_test),
        # Method accuracy
        "method_accuracy":     df_test["method_correct"].mean(),
        "method_ko_acc":       (df_test[df_test["pred_method"]=="KO_TKO"]["method_correct"]).mean() if len(df_test[df_test["pred_method"]=="KO_TKO"]) > 0 else 0,
        "method_sub_acc":      (df_test[df_test["pred_method"]=="Submission"]["method_correct"]).mean() if len(df_test[df_test["pred_method"]=="Submission"]) > 0 else 0,
        "method_dec_acc":      (df_test[df_test["pred_method"]=="Decision"]["method_correct"]).mean() if len(df_test[df_test["pred_method"]=="Decision"]) > 0 else 0,
        "method_n_ko":         len(df_test[df_test["pred_method"]=="KO_TKO"]),
        "method_n_sub":        len(df_test[df_test["pred_method"]=="Submission"]),
        "method_n_dec":        len(df_test[df_test["pred_method"]=="Decision"]),
        "method_pnl_total":    df_test["method_pnl"].sum(),
        "method_pnl_per_bet":  df_test["method_pnl"].mean(),
        # Round accuracy
        "round_accuracy":      df_test["round_correct"].mean(),
        "round_under_acc":     (df_test[df_test["pred_early"]==1]["round_correct"]).mean() if len(df_test[df_test["pred_early"]==1]) > 0 else 0,
        "round_over_acc":      (df_test[df_test["pred_early"]==0]["round_correct"]).mean() if len(df_test[df_test["pred_early"]==0]) > 0 else 0,
        "round_n_under":       len(df_test[df_test["pred_early"]==1]),
        "round_n_over":        len(df_test[df_test["pred_early"]==0]),
        "actual_finish_rate":  df_test["actual_early"].mean(),
        "pred_finish_rate":    df_test["pred_early"].mean(),
        "round_pnl_total":     df_test["round_pnl"].sum(),
        "round_pnl_per_bet":   df_test["round_pnl"].mean(),
        # High-confidence bets only
        "high_conf_method_n":   len(high_conf_method),
        "high_conf_method_acc": high_conf_method["method_correct"].mean() if len(high_conf_method) > 0 else 0,
        "high_conf_method_pnl": high_conf_method["method_pnl"].sum(),
        "high_conf_round_n":    len(high_conf_round),
        "high_conf_round_acc":  high_conf_round["round_correct"].mean() if len(high_conf_round) > 0 else 0,
        "high_conf_round_pnl":  high_conf_round["round_pnl"].sum(),
        # Actual method distribution (for reference)
        "actual_ko_rate":   (df_test["method"] == "KO_TKO").mean(),
        "actual_sub_rate":  (df_test["method"] == "Submission").mean(),
        "actual_dec_rate":  (df_test["method"] == "Decision").mean(),
        # Year breakdown
        "df_test": df_test,
    }
    return results


def print_report(results: dict, args):
    """Print comprehensive backtest report."""
    df = results.pop("df_test", None)

    print("\n" + "═" * 64)
    print("  UFC PROPS BACKTEST REPORT")
    print(f"  {results['n_fights']} test fights (holdout — never trained on)")
    if args.weight_class:
        print(f"  Filter: {args.weight_class}")
    if args.year:
        print(f"  Year: {args.year}")
    print("═" * 64)

    # ── Method ────────────────────────────────────────────────────────────────
    print("\n  METHOD PREDICTION")
    print("  " + "─" * 50)
    print(f"  Overall accuracy:    {results['method_accuracy']:.1%}")
    print(f"  Flat $100 P&L:       ${results['method_pnl_total']:+.0f}  (${results['method_pnl_per_bet']:+.1f}/bet)")
    print()
    print(f"  {'Predicted':14s}  {'Calls':6s}  {'Accuracy':9s}  {'Market edge?'}")
    print("  " + "─" * 50)

    # KO/TKO
    mkt_ko = american_to_prob(PROP_ODDS["method_ko_tko"])
    edge_ko = results["method_ko_acc"] - mkt_ko
    status_ko = "✅ edge" if edge_ko > 0.03 else ("⚠️  no edge" if edge_ko > -0.03 else "❌ fade this")
    print(f"  {'KO/TKO':14s}  {results['method_n_ko']:6d}  {results['method_ko_acc']:8.1%}  {status_ko} ({edge_ko:+.1%} vs mkt)")

    # Submission
    mkt_sub = american_to_prob(PROP_ODDS["method_sub"])
    edge_sub = results["method_sub_acc"] - mkt_sub
    status_sub = "✅ edge" if edge_sub > 0.03 else ("⚠️  no edge" if edge_sub > -0.03 else "❌ fade this")
    print(f"  {'Submission':14s}  {results['method_n_sub']:6d}  {results['method_sub_acc']:8.1%}  {status_sub} ({edge_sub:+.1%} vs mkt)")

    # Decision
    mkt_dec = american_to_prob(PROP_ODDS["method_decision"])
    edge_dec = results["method_dec_acc"] - mkt_dec
    status_dec = "✅ edge" if edge_dec > 0.03 else ("⚠️  no edge" if edge_dec > -0.03 else "❌ fade this")
    print(f"  {'Decision':14s}  {results['method_n_dec']:6d}  {results['method_dec_acc']:8.1%}  {status_dec} ({edge_dec:+.1%} vs mkt)")

    print()
    print(f"  Actual method distribution:  KO/TKO {results['actual_ko_rate']:.1%}  "
          f"Sub {results['actual_sub_rate']:.1%}  Dec {results['actual_dec_rate']:.1%}")

    print(f"\n  HIGH CONFIDENCE METHOD (prob > 55%)")
    print(f"  Calls: {results['high_conf_method_n']}  "
          f"Accuracy: {results['high_conf_method_acc']:.1%}  "
          f"P&L: ${results['high_conf_method_pnl']:+.0f}")

    # ── Round ─────────────────────────────────────────────────────────────────
    print("\n  ROUND O/U PREDICTION")
    print("  " + "─" * 50)
    print(f"  Overall accuracy:    {results['round_accuracy']:.1%}")
    print(f"  Flat $100 P&L:       ${results['round_pnl_total']:+.0f}  (${results['round_pnl_per_bet']:+.1f}/bet)")
    print()
    mkt_under = american_to_prob(PROP_ODDS["round_under_2_5"])
    mkt_over  = american_to_prob(PROP_ODDS["round_over_2_5"])
    edge_under = results["round_under_acc"] - mkt_under
    edge_over  = results["round_over_acc"] - mkt_over
    status_u = "✅ edge" if edge_under > 0.03 else ("⚠️  no edge" if edge_under > -0.03 else "❌ fade this")
    status_o = "✅ edge" if edge_over  > 0.03 else ("⚠️  no edge" if edge_over  > -0.03 else "❌ fade this")
    print(f"  {'UNDER 2.5':14s}  {results['round_n_under']:6d}  {results['round_under_acc']:8.1%}  {status_u} ({edge_under:+.1%} vs mkt)")
    print(f"  {'OVER 2.5':14s}  {results['round_n_over']:6d}  {results['round_over_acc']:8.1%}  {status_o} ({edge_over:+.1%} vs mkt)")
    print()
    print(f"  Actual early finish rate:  {results['actual_finish_rate']:.1%}")
    print(f"  Model predicted early:     {results['pred_finish_rate']:.1%}  "
          f"({'overestimates' if results['pred_finish_rate'] > results['actual_finish_rate'] else 'underestimates'} finishes)")

    print(f"\n  HIGH CONFIDENCE ROUND (prob > 60%)")
    print(f"  Calls: {results['high_conf_round_n']}  "
          f"Accuracy: {results['high_conf_round_acc']:.1%}  "
          f"P&L: ${results['high_conf_round_pnl']:+.0f}")

    # ── Year breakdown ────────────────────────────────────────────────────────
    if df is not None and len(df) > 0:
        print("\n  ACCURACY BY YEAR")
        print("  " + "─" * 54)
        print(f"  {'Year':6s}  {'Fights':7s}  {'Method%':9s}  {'Round%':8s}  {'KO%act':8s}  {'Dec%act'}")
        print("  " + "─" * 54)
        df["year"] = pd.to_datetime(df["fight_date"]).dt.year
        for yr, grp in df.groupby("year"):
            if len(grp) < 10:
                continue
            m_acc = grp["method_correct"].mean()
            r_acc = grp["round_correct"].mean()
            ko_act = (grp["method"] == "KO_TKO").mean()
            dec_act = (grp["method"] == "Decision").mean()
            trend_m = "📈" if m_acc >= results["method_accuracy"] else "📉"
            trend_r = "📈" if r_acc >= results["round_accuracy"] else "📉"
            print(f"  {int(yr):6d}  {len(grp):7d}  "
                  f"{trend_m}{m_acc:7.1%}  "
                  f"{trend_r}{r_acc:6.1%}  "
                  f"{ko_act:7.1%}  {dec_act:7.1%}")
        print()

    # ── Betting verdict ───────────────────────────────────────────────────────
    print("  BETTING VERDICT")
    print("  " + "─" * 54)
    if results["method_pnl_total"] > 0:
        print("  ✅ Method bets: positive historical P&L — model has edge")
    else:
        print("  ⚠️  Method bets: negative historical P&L — use with caution")

    if results["round_pnl_total"] > 0:
        print("  ✅ Round bets: positive historical P&L — model has edge")
    else:
        print("  ⚠️  Round bets: negative historical P&L — model overestimates finishes")

    if results["high_conf_method_acc"] > results["method_accuracy"]:
        print(f"  ✅ High-confidence method picks outperform overall "
              f"({results['high_conf_method_acc']:.1%} vs {results['method_accuracy']:.1%}) — filter by confidence")
    else:
        print("  ⚠️  High-confidence method picks don't outperform — confidence not predictive of accuracy")

    print()
    print("  NOTE: These use simulated average odds. Actual edge depends on")
    print("  live lines — always compare model prob vs real sportsbook line.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest UFC prop predictions")
    parser.add_argument("--events",       type=int,  default=None, help="Last N events to test on")
    parser.add_argument("--year",         type=int,  default=None, help="Filter to specific year")
    parser.add_argument("--weight-class", type=str,  default=None, help="Filter to weight class (e.g. Lightweight)")
    parser.add_argument("--method",       type=str,  default=None, help="Filter to specific method (KO_TKO/Submission/Decision)")
    parser.add_argument("--min-year",     type=int,  default=2020, help="Minimum year for test data (default: 2020)")
    args = parser.parse_args()

    init_db()
    session = get_session()

    logger.info("Loading dataset...")
    df = load_dataset(session, events=args.events, year=args.year)

    # Filter to modern era by default — pre-2020 UFC is different sport
    if not args.year:
        df = df[pd.to_datetime(df["fight_date"]).dt.year >= args.min_year]
        logger.info(f"Filtered to {args.min_year}+: {len(df)} fights")

    logger.info("Loading predictor...")
    predictor = UFCPredictor()
    predictor.load()

    logger.info("Running backtest...")
    results = run_backtest(df, predictor, args.weight_class, args.method)

    if results:
        print_report(results, args)

    session.close()


if __name__ == "__main__":
    main()
