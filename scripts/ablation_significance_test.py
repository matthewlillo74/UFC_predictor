"""
scripts/ablation_significance_test.py
────────────────────────────────────────
Tests whether each accuracy-queue item's shipped state is a statistically
significant improvement over the state immediately before it, using
McNemar's test on paired per-fight predictions from the same fixed test set.

INCREMENTAL framing, not isolated ablation — deliberately, per review +
sign-off (2026-07-16). The question that matters is "is the model as
deployed right now better than before this queue started," state by state —
not "does each feature help in a vacuum." Isolated ablation breaks down for
a feature like the opponent-quality adjustment, which is degenerate without
the sparsity fix ahead of it in the actual build order; testing it alone
would just re-confirm something already known, not answer anything new.

States tested, each compared to the one immediately before it:
    0 -> 1: item 1 (control_time/reversals),        73 -> 75 features
    1 -> 2: item 2 (opponent-quality adjustment),    75 -> 79 features
    2 -> 5: item 5 (Elo K-factor decay),              79 features, Elo history decayed
    5 -> 6: item 6 (layoff penalty),                 79 -> 80 features
    0 -> 6: overall net effect (bonus, not part of the incremental chain)

Item 3 (per-division calibration) is deliberately NOT McNemar-tested — it's
a post-hoc probability rescaling that never touches the trained model, so it
literally cannot change which side wins the argmax call; raw and calibrated
predictions are identical on every single fight by construction. Its effect
is measured separately via log loss / Brier score. See
calibration_probability_report().

Excluded from scope (review + friend sign-off, 2026-07-16):
    - item 4 (SHAP miss analysis) — diagnostic script, no predictions changed.
    - the leakage/sparsity revert — comparing against the old leaky baseline
      would be comparing against a known-invalid number, not a fair one.

McNemar's test (exact binomial on discordant pairs, not the chi-square
approximation) is the right tool: paired binary outcomes on the identical
test fights, correctly accounting for the correlation between paired
predictions instead of treating two accuracy percentages as independent.

IMPORTANT — two real bugs were caught and fixed while building this, both
worth knowing about before trusting the numbers below:

  1. Calibration confound: the first version applied per-division calibration
     to states 3/5/6 and compared against historically-reported accuracy
     numbers that were NEVER calibrated (UFCPredictor.evaluate() calls raw
     winner_model.predict_proba() directly, always has). Comparing calibrated
     ablation predictions against raw historical numbers is comparing two
     different things measured two different ways. Fixed: every McNemar
     comparison below uses raw, uncalibrated predictions throughout.

  2. Cross-run contamination: training multiple models sequentially in one
     Python process produces WORSE results for later models than training
     the identical configuration alone in a fresh process — verified
     directly: state 6 trained 6th-in-a-chain scored 61.2%, the identical
     state 6 trained alone in a fresh process scored 62.2% (exactly matching
     the actually-committed production model). Root cause not fully
     diagnosed (a candidate: XGBoost's random_state=42 not being fully
     process-independent across repeated .fit() calls) but the fix is
     simple and verified: every state below trains in its own subprocess.

SAFETY: recomputes Elo history (flat, then restores decayed) to reconstruct
pre-item-5 states. Restores the production (decayed) Elo state in a finally
block even if something fails partway through. Does not touch
models_saved/v1/ or data/processed/training_dataset.csv. Do not run this
concurrently with anything else that trains a model or touches EloRating.

Usage:
    python scripts/ablation_significance_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import tempfile

import pandas as pd
from scipy.stats import binomtest
from loguru import logger

from src.database import init_db, get_session
from src.features.feature_builder import build_training_dataset
from config import FEATURE_COLUMNS as ORIGINAL_FEATURE_COLUMNS
from scripts.recompute_elo import recompute_elo

# Item 1 added these 2 columns; item 2 added these 4; item 6 added 1. Verified
# against git history (commit a0927048 bundled items 1+2 together) — diffed
# config.py::FEATURE_COLUMNS at b5c24214 (73, pre-queue baseline) vs
# a0927048 (79) to get the combined +6, split by known authorship.
#
# CRITICAL: state column lists are built by FILTERING production's actual
# FEATURE_COLUMNS order, not by concatenating hand-typed lists. XGBoost's
# colsample_bytree=0.8 samples columns BY INDEX POSITION, not by name — an
# earlier version of this script built state lists as
# STATE0_COLS + new_item_cols (appending new columns at the end), which
# produced a column ORDER that diverged from production (production has each
# item's new columns interspersed near their logical predecessor, e.g.
# layoff_penalty_diff sits right after days_since_last_fight_diff, not at
# the very end). Same feature SET, different ORDER -> colsample_bytree with
# the same random_state=42 samples a completely different subset of columns
# per tree -> a genuinely different (non-equivalent) trained model despite
# "the same 80 features." This was caught because it reproducibly gave 61.2%
# for the reconstructed state6 instead of the actually-committed 62.2%, even
# after ruling out cross-process contamination and CSV round-trip precision
# as causes via direct isolated tests. Filtering preserves production's exact
# relative order for every state, and STATE6_COLS is production's list
# itself, guaranteeing state6 is bit-for-bit the deployed configuration.
ITEM1_COLS = {"control_time_diff", "reversals_diff"}
ITEM2_COLS = {"sapm_adj_diff", "slpm_adj_diff", "td_avg_adj_diff", "td_def_adj_diff"}
ITEM6_COLS = {"layoff_penalty_diff"}


def _filtered(exclude: set) -> list:
    return [c for c in ORIGINAL_FEATURE_COLUMNS if c not in exclude]


STATE0_COLS = _filtered(ITEM1_COLS | ITEM2_COLS | ITEM6_COLS)
STATE1_COLS = _filtered(ITEM2_COLS | ITEM6_COLS)
STATE2_COLS = _filtered(ITEM6_COLS)
STATE6_COLS = list(ORIGINAL_FEATURE_COLUMNS)

assert len(STATE0_COLS) == 73, f"state0 should be 73 cols, got {len(STATE0_COLS)}"
assert len(STATE1_COLS) == 75, f"state1 should be 75 cols, got {len(STATE1_COLS)}"
assert len(STATE2_COLS) == 79, f"state2 should be 79 cols, got {len(STATE2_COLS)}"
assert len(STATE6_COLS) == 80, f"state6 should be 80 cols, got {len(STATE6_COLS)}"
assert STATE6_COLS == ORIGINAL_FEATURE_COLUMNS, "state6 must be exactly production's column list/order"


# ═══════════════════════════════════════════════════════════════════════════
# WORKER MODE — runs a single state's training in its own fresh process.
# Invoked by main() via subprocess, never called directly by a user.
# ═══════════════════════════════════════════════════════════════════════════

def _worker_main(args):
    import src.models.predict as predict_module
    from src.models.predict import UFCPredictor
    from sklearn.metrics import log_loss, brier_score_loss

    cols = json.loads(args.cols_json)
    predict_module.FEATURE_COLUMNS = cols

    df = pd.read_csv(args.dataset_csv, parse_dates=["fight_date"])
    split_idx = int(len(df) * 0.85)
    df_sorted = df.sort_values("fight_date").reset_index(drop=True)
    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]

    predictor = UFCPredictor()
    predictor.train(df_train)

    classes = list(predictor.winner_model.classes_)
    idx_1 = classes.index(1) if 1 in classes else 1
    X_test = df_test[cols].fillna(0)
    raw_probs = predictor.winner_model.predict_proba(X_test)[:, idx_1]

    preds = (raw_probs >= 0.5).astype(int)
    correct = (preds == df_test["winner"]).astype(bool)
    per_fight = dict(zip(df_test["fight_id"].astype(int).tolist(), correct.tolist()))

    result = {"per_fight_correct": per_fight, "accuracy": float(correct.mean())}

    if args.also_calibrated:
        cal_probs = pd.Series(index=df_test.index, dtype=float)
        for wc, group in df_test.groupby("weight_class", dropna=False):
            X_group = group[cols].fillna(0)
            cal = predictor.winner_calibrators_by_division.get(wc) if getattr(predictor, "winner_calibrators_by_division", None) else None
            if cal is not None:
                cal_probs.loc[group.index] = cal.predict_proba(X_group)[:, idx_1]
            else:
                cal_probs.loc[group.index] = predictor.winner_model.predict_proba(X_group)[:, idx_1]
        y = df_test["winner"]
        result["calibration_comparison"] = {
            "raw_log_loss": float(log_loss(y, raw_probs)),
            "cal_log_loss": float(log_loss(y, cal_probs)),
            "raw_brier": float(brier_score_loss(y, raw_probs)),
            "cal_brier": float(brier_score_loss(y, cal_probs)),
        }

    with open(args.output, "w") as f:
        json.dump(result, f)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — builds datasets, dispatches worker subprocesses, runs stats.
# ═══════════════════════════════════════════════════════════════════════════

def _run_state_subprocess(dataset_csv: str, cols: list, label: str, also_calibrated: bool = False) -> dict:
    logger.info(f"Training state '{label}' ({len(cols)} features) in isolated subprocess...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_f:
        output_path = out_f.name
    try:
        subprocess.run(
            [sys.executable, __file__, "--worker",
             "--dataset-csv", dataset_csv,
             "--cols-json", json.dumps(cols),
             "--output", output_path]
            + (["--also-calibrated"] if also_calibrated else []),
            check=True,
        )
        with open(output_path) as f:
            result = json.load(f)
        result["per_fight_correct"] = {int(k): v for k, v in result["per_fight_correct"].items()}
        logger.info(f"  state '{label}' accuracy (raw): {result['accuracy']:.1%}")
        return result
    finally:
        os.unlink(output_path)


def mcnemar_test(prev_results: dict, next_results: dict):
    """Exact binomial McNemar's test. c ~ Binomial(b+c, 0.5) under the null."""
    prev = prev_results["per_fight_correct"]
    nxt = next_results["per_fight_correct"]
    fight_ids = set(prev) & set(nxt)
    b = sum(1 for fid in fight_ids if prev[fid] and not nxt[fid])
    c = sum(1 for fid in fight_ids if not prev[fid] and nxt[fid])
    n_discordant = b + c

    prev_acc = sum(prev[f] for f in fight_ids) / len(fight_ids)
    next_acc = sum(nxt[f] for f in fight_ids) / len(fight_ids)

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0,
                "prev_acc": prev_acc, "next_acc": next_acc, "n_paired": len(fight_ids)}

    result = binomtest(c, n_discordant, p=0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": result.pvalue,
            "prev_acc": prev_acc, "next_acc": next_acc, "n_paired": len(fight_ids)}


def report(label: str, test_result: dict):
    sig = "SIGNIFICANT (p<0.05)" if test_result["p_value"] < 0.05 else "not significant"
    delta = (test_result["next_acc"] - test_result["prev_acc"]) * 100
    print(f"\n  {label}")
    print(f"    accuracy: {test_result['prev_acc']:.1%} -> {test_result['next_acc']:.1%}  "
          f"(delta {delta:+.1f}pp, n={test_result['n_paired']})")
    print(f"    discordant pairs: b(prev-right,next-wrong)={test_result['b']}  "
          f"c(prev-wrong,next-right)={test_result['c']}")
    print(f"    McNemar exact binomial p-value: {test_result['p_value']:.4f}  -> {sig}")


def calibration_report(state2_result: dict):
    cc = state2_result["calibration_comparison"]
    print("\n  2 -> 3: item 3 (per-division calibration) — probability-quality comparison")
    print("    (accuracy/McNemar not applicable: calibration cannot change the argmax call,")
    print("     raw and calibrated predictions are identical on every fight by construction)")
    print(f"    log loss:    raw={cc['raw_log_loss']:.4f}  calibrated={cc['cal_log_loss']:.4f}  "
          f"delta={cc['cal_log_loss']-cc['raw_log_loss']:+.4f}  "
          f"({'better' if cc['cal_log_loss'] < cc['raw_log_loss'] else 'worse'})")
    print(f"    Brier score: raw={cc['raw_brier']:.4f}  calibrated={cc['cal_brier']:.4f}  "
          f"delta={cc['cal_brier']-cc['raw_brier']:+.4f}  "
          f"({'better' if cc['cal_brier'] < cc['raw_brier'] else 'worse'})")


def _build_and_save_dataset(session, csv_path: str):
    df = build_training_dataset(session)
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    init_db()
    session = get_session()

    flat_csv = None
    decayed_csv = None
    try:
        # ═══ FLAT-ELO PHASE: states 0, 1, 2 ═══
        logger.info("=== Reconstructing FLAT-Elo historical state (items 0-2) ===")
        recompute_elo(session, flat=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            flat_csv = f.name
        _build_and_save_dataset(session, flat_csv)

        state0 = _run_state_subprocess(flat_csv, STATE0_COLS, "0 (baseline, pre-queue)")
        state1 = _run_state_subprocess(flat_csv, STATE1_COLS, "1 (+control_time/reversals)")
        state2 = _run_state_subprocess(flat_csv, STATE2_COLS, "2 (+opponent-adjustment)", also_calibrated=True)

        # ═══ DECAYED-ELO PHASE (production): states 5, 6 ═══
        logger.info("=== Restoring DECAYED-Elo (production) state for items 5-6 ===")
        recompute_elo(session, flat=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            decayed_csv = f.name
        _build_and_save_dataset(session, decayed_csv)

        state5 = _run_state_subprocess(decayed_csv, STATE2_COLS, "5 (+Elo K-factor decay)")
        state6 = _run_state_subprocess(decayed_csv, STATE6_COLS, "6 (+layoff penalty, current production)")

        # ═══ McNemar's tests (raw accuracy, matches evaluate()'s methodology) ═══
        print("\n" + "=" * 74)
        print("  INCREMENTAL SIGNIFICANCE TESTING — McNemar's exact binomial test")
        print("  (raw, uncalibrated predictions; each state trained in its own fresh")
        print("   subprocess — both fixes verified necessary, see module docstring)")
        print("=" * 74)

        report("0 -> 1: item 1 (control_time/reversals)", mcnemar_test(state0, state1))
        report("1 -> 2: item 2 (opponent-quality adjustment)", mcnemar_test(state1, state2))
        calibration_report(state2)
        report("2 -> 5: item 5 (Elo K-factor decay)", mcnemar_test(state2, state5))
        report("5 -> 6: item 6 (layoff penalty)", mcnemar_test(state5, state6))
        print("\n  --- bonus: net effect across the whole queue ---")
        report("0 -> 6: overall (all 6 items combined)", mcnemar_test(state0, state6))
        print("\n" + "=" * 74 + "\n")

        print(f"  Sanity check — state6 accuracy should match committed production (62.2%): "
              f"{state6['accuracy']:.1%}")

    finally:
        for p in (flat_csv, decayed_csv):
            if p and os.path.exists(p):
                os.unlink(p)
        logger.info("Restoring production Elo state (decayed K-factor)...")
        recompute_elo(session, flat=False)
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-csv")
    parser.add_argument("--cols-json")
    parser.add_argument("--output")
    parser.add_argument("--also-calibrated", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
    else:
        main()
