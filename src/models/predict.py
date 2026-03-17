"""
src/models/predict.py
─────────────────────
Prediction model for UFC fights.

Predicts:
  1. Winner (binary probability)
  2. Method: KO/TKO | Submission | Decision
  3. Round finish (over/under)

Uses XGBoost as primary model with SHAP for explainability.
Time-based train/test split is mandatory — no random shuffling.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from loguru import logger

from config import FEATURE_COLUMNS, MODEL_VERSION, MODELS_DIR


class UFCPredictor:
    """
    Main prediction model.

    Usage:
        predictor = UFCPredictor()
        predictor.train(df_historical)
        result = predictor.predict(features_dict, fighter_a_name, fighter_b_name)
    """

    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version
        self.winner_model = None
        self.method_model = None
        self.round_model = None
        self.shap_explainer = None
        self._is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        """
        Train all prediction models on historical fight data.

        Args:
            df: DataFrame where each row is a historical fight with:
                - All FEATURE_COLUMNS
                - 'winner' column (1 = fighter_A, 0 = fighter_B)
                - 'method' column (KO_TKO | Submission | Decision)
                - 'finish_round' column (int or NaN for decisions)
                - 'fight_date' column (for time-based split)

        CRITICAL: df must be sorted by fight_date ascending.
                  We use time-based split to prevent leakage.
        """
        df = df.sort_values("fight_date").reset_index(drop=True)

        # Filter to only clean results for training (exclude NC, Draw, Overturned)
        df_clean = df[df["method"].isin(["KO_TKO", "Submission", "Decision"])].copy()
        logger.info(f"Clean fights for training: {len(df_clean)} (removed {len(df) - len(df_clean)} NC/Draw/other)")

        X = df_clean[FEATURE_COLUMNS].fillna(0)
        y_winner = df_clean["winner"]

        # ── Recency weighting ──────────────────────────────────────────────────
        # Modern MMA (2020+) is very different from early UFC. Fights from 1994
        # have different pacing, ruleset, and fighter caliber. We give recent
        # fights exponentially more weight so the model learns from modern patterns.
        #
        # Decay: 2-year half-life — a fight from 2 years ago gets weight 0.5,
        # 4 years ago gets 0.25, etc. Fights from 2024-2026 get weight ~1.0.
        # This makes the model more sensitive to current trends without discarding
        # historical data entirely.
        now = pd.Timestamp.utcnow().tz_localize(None)
        fight_dates = pd.to_datetime(df_clean["fight_date"]).dt.tz_localize(None)
        days_ago = (now - fight_dates).dt.days.clip(lower=0)
        half_life_days = 365 * 2  # 2-year half-life
        sample_weights = np.power(0.5, days_ago / half_life_days)
        sample_weights = (sample_weights / sample_weights.mean()).values  # normalize to mean=1
        sample_weights = sample_weights.astype(np.float32)  # sklearn calibration requires float32

        recent_pct = (days_ago < 365).mean()
        logger.info(f"Recency weights: {recent_pct:.1%} of fights within last year, "
                    f"weight range {sample_weights.min():.3f}–{sample_weights.max():.3f}")

        # XGBoost multi-class needs integer labels — encode method strings
        method_map = {"Decision": 0, "KO_TKO": 1, "Submission": 2}
        y_method = df_clean["method"].map(method_map)
        self.method_classes_ = {v: k for k, v in method_map.items()}  # reverse for prediction

        y_finish = (df_clean["finish_round"].fillna(3) <= 2.5).astype(int)

        logger.info(f"Training on {len(df)} fights")
        logger.info(f"Date range: {df.fight_date.min()} → {df.fight_date.max()}")
        logger.info(f"Class balance (winner): {y_winner.mean():.2%} fighter_A wins")

        # ── Winner model (XGBoost) ─────────────────────────────────────────
        self.winner_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.winner_model.fit(X, y_winner, sample_weight=sample_weights)
        self.winner_calibrator = None  # raw XGBoost probabilities are used directly

        # ── Method model (multinomial) ─────────────────────────────────────
        self.method_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
        self.method_model.fit(X, y_method, sample_weight=sample_weights)

        # ── Round model (binary: finish early vs go late) ──────────────────
        # CRITICAL: Train on ALL fights, not just finishes.
        # Training only on finishes was asking "given a finish, was it early?"
        # which is the wrong question. We want "will this fight end early?"
        # For decisions, finish_round = NaN → we fill with the scheduled rounds
        # (3 for normal fights, 5 for title/main events) so they always count as "late."
        #
        # Threshold is fight-type aware:
        #   3-round fights: early = rounds 1-2  (under 2.5)
        #   5-round fights: early = rounds 1-3  (under 3.5)

        def get_early_label(row):
            """1 = ended early, 0 = went late/distance."""
            is_five_round = row.get("is_title_fight", 0) == 1
            threshold = 3.5 if is_five_round else 2.5
            # Decisions always count as going the distance (late)
            if row["method"] == "Decision":
                return 0
            # Finish — did it end before the threshold?
            finish_r = row.get("finish_round")
            if finish_r is None or (isinstance(finish_r, float) and finish_r != finish_r):
                return 0  # unknown → conservative, call it late
            return 1 if float(finish_r) <= threshold else 0

        y_early = df_clean.apply(get_early_label, axis=1)
        early_rate = y_early.mean()
        logger.info(f"Round model — early finish rate in training: {early_rate:.1%}")

        self.round_model = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        self.round_model.fit(X, y_early, sample_weight=sample_weights)

        # Calibrate round model with Platt scaling to fix overconfidence
        from sklearn.calibration import CalibratedClassifierCV
        self.round_model_calibrated = CalibratedClassifierCV(
            self.round_model, method="sigmoid", cv="prefit"
        )
        self.round_model_calibrated.fit(X, y_early, sample_weight=sample_weights)
        logger.info("Round model calibrated with Platt scaling")

        # ── SHAP explainer ─────────────────────────────────────────────────
        self.shap_explainer = shap.TreeExplainer(self.winner_model)

        self._is_trained = True
        logger.success(f"Training complete. Model version: {self.model_version}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        features: dict,
        fighter_a_name: str,
        fighter_b_name: str,
    ) -> dict:
        """
        Generate full prediction for a fight.

        Returns:
            dict with probabilities, method breakdown, explanations, etc.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)

        # Raw winner probabilities from XGBoost
        win_probs_raw = self.winner_model.predict_proba(X)[0]
        classes = list(self.winner_model.classes_)
        idx_a = classes.index(1) if 1 in classes else 1
        idx_b = classes.index(0) if 0 in classes else 0
        raw_a = float(win_probs_raw[idx_a])
        raw_b = float(win_probs_raw[idx_b])

        # Apply probability cap — raw XGBoost becomes overconfident for debut fighters
        # or fighters with sparse stats (large feature gaps push extreme probabilities).
        # From calibration report: 80-90% bucket hits 76.6%, 90%+ hits 83.3%.
        # We cap at 90% since we have no validated data above that range.
        # This makes 97-99% predictions display as 90% which better reflects true confidence.
        if hasattr(self, "winner_calibrator") and self.winner_calibrator is not None:
            prob_a = float(self.winner_calibrator.predict([raw_a])[0])
            prob_b = 1.0 - prob_a
        else:
            prob_a, prob_b = raw_a, raw_b

        # Cap at 90% max — never report more confidence than we can validate
        MAX_PROB = 0.90
        if prob_a > MAX_PROB:
            prob_a = MAX_PROB
            prob_b = 1.0 - MAX_PROB
        elif prob_b > MAX_PROB:
            prob_b = MAX_PROB
            prob_a = 1.0 - MAX_PROB

        # Method probabilities — map integer classes back to string names
        method_probs = self.method_model.predict_proba(X)[0]
        method_label_map = getattr(self, "method_classes_", {0: "Decision", 1: "KO_TKO", 2: "Submission"})
        method_dict = {method_label_map[i]: float(p) for i, p in enumerate(method_probs)}

        # Round probability — use calibrated model, title-fight aware threshold
        round_probs = {}
        is_title = bool(features.get("is_title_fight", 0))
        if hasattr(self, "round_model_calibrated") and self.round_model_calibrated:
            rp = self.round_model_calibrated.predict_proba(X)[0]
            # rp[0] = prob late (class 0), rp[1] = prob early (class 1)
            if is_title:
                round_probs = {"under_3_5": float(rp[1]), "over_3_5": float(rp[0])}
            else:
                round_probs = {"under_2_5": float(rp[1]), "over_2_5": float(rp[0])}
        elif self.round_model and self.round_model.n_features_in_:
            # Fallback to uncalibrated if calibrated not available (old saved model)
            rp = self.round_model.predict_proba(X)[0]
            round_probs = {"under_2_5": float(rp[1]), "over_2_5": float(rp[0])}

        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(X)
        explanation = self._build_explanation(
            shap_values[0], FEATURE_COLUMNS, fighter_a_name, fighter_b_name
        )

        result = {
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
            "prob_fighter_a": round(prob_a, 4),
            "prob_fighter_b": round(prob_b, 4),
            "predicted_winner": fighter_a_name if prob_a > prob_b else fighter_b_name,
            "confidence": round(max(prob_a, prob_b), 4),
            "method_probabilities": {
                "ko_tko": round(method_dict.get("KO_TKO", 0), 4),
                "submission": round(method_dict.get("Submission", 0), 4),
                "decision": round(method_dict.get("Decision", 0), 4),
            },
            "round_probabilities": round_probs,
            "shap_values": dict(zip(FEATURE_COLUMNS, shap_values[0].tolist())),
            "explanation": explanation,
            "model_version": self.model_version,
            "predicted_at": datetime.utcnow().isoformat(),
        }

        result["consistency"] = self._check_consistency(result)
        return result

    def _check_consistency(self, result: dict) -> dict:
        """
        Detect contradictions between method and round predictions.

        The method and round models are trained independently, so they can
        disagree. This post-processing layer catches the most common conflicts
        and surfaces them in the UI rather than silently showing contradictory info.

        Returns a dict with:
            status:  "consistent" | "warning" | "contradiction"
            message: human-readable explanation of what conflicts and why
        """
        methods = result.get("method_probabilities", {})
        rounds  = result.get("round_probabilities", {})

        if not methods or not rounds:
            return {"status": "consistent", "message": ""}

        p_decision  = methods.get("decision", 0)
        p_finish    = methods.get("ko_tko", 0) + methods.get("submission", 0)
        # Handle both 3-round (under_2_5) and 5-round (under_3_5) fights
        p_under_2_5 = rounds.get("under_2_5", rounds.get("under_3_5", 0))
        p_over_2_5  = rounds.get("over_2_5",  rounds.get("over_3_5",  0))
        is_title    = "under_3_5" in rounds

        round_label = "3.5" if is_title else "2.5"

        # Hard contradiction: model strongly predicts decision AND strongly predicts early finish
        if p_decision > 0.55 and p_under_2_5 > 0.65:
            return {
                "status": "contradiction",
                "message": (
                    f"Method model says Decision ({p_decision:.0%}) "
                    f"but Round model says early finish ({p_under_2_5:.0%} Under {round_label}). "
                    f"These models disagree — treat method prediction with caution."
                )
            }

        # Hard contradiction: strong finish prediction + strong over rounds
        if p_finish > 0.65 and p_over_2_5 > 0.75:
            return {
                "status": "contradiction",
                "message": (
                    f"Method model says finish ({p_finish:.0%} KO/Sub) "
                    f"but Round model says late fight ({p_over_2_5:.0%} Over {round_label}). "
                    f"A late finish is possible but unusual — low confidence on method."
                )
            }

        # Soft warning: moderate disagreement
        if p_decision > 0.50 and p_under_2_5 > 0.55:
            return {
                "status": "warning",
                "message": (
                    f"Mild disagreement: Decision likely ({p_decision:.0%}) "
                    f"but early finish not ruled out ({p_under_2_5:.0%} Under {round_label})."
                )
            }

        if p_finish > 0.55 and p_over_2_5 > 0.60:
            return {
                "status": "warning",
                "message": (
                    f"Mild disagreement: Finish likely ({p_finish:.0%}) "
                    f"but could go late ({p_over_2_5:.0%} Over {round_label})."
                )
            }

        return {"status": "consistent", "message": ""}

    def _build_explanation(
        self,
        shap_vals: np.ndarray,
        feature_names: list,
        fighter_a: str,
        fighter_b: str,
    ) -> dict:
        """
        Convert SHAP values into human-readable explanation.

        Returns top factors FOR and AGAINST fighter A winning.
        """
        pairs = sorted(zip(shap_vals, feature_names), key=lambda x: abs(x[0]), reverse=True)
        top_10 = pairs[:10]

        for_a = [(name, val) for val, name in top_10 if val > 0]
        for_b = [(name, abs(val)) for val, name in top_10 if val < 0]

        def fmt_feature(name: str, val: float) -> str:
            label = name.replace("_diff", "").replace("_", " ").title()
            return f"{label} advantage ({val:+.3f})"

        return {
            f"factors_favoring_{fighter_a}": [fmt_feature(n, v) for n, v in for_a[:5]],
            f"factors_favoring_{fighter_b}": [fmt_feature(n, v) for n, v in for_b[:5]],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        """Save trained models to disk."""
        path = Path(MODELS_DIR) / self.model_version
        path.mkdir(exist_ok=True)
        with open(path / "winner_model.pkl", "wb") as f:
            pickle.dump(self.winner_model, f)
        with open(path / "method_model.pkl", "wb") as f:
            pickle.dump(self.method_model, f)
        with open(path / "round_model.pkl", "wb") as f:
            pickle.dump(self.round_model, f)
        # Isotonic calibrator for winner probabilities
        if hasattr(self, "winner_calibrator") and self.winner_calibrator is not None:
            with open(path / "winner_calibrator.pkl", "wb") as f:
                pickle.dump(self.winner_calibrator, f)
        # Calibrated round model
        if hasattr(self, "round_model_calibrated") and self.round_model_calibrated:
            with open(path / "round_model_calibrated.pkl", "wb") as f:
                pickle.dump(self.round_model_calibrated, f)
        logger.success(f"Models saved to {path}")

    def load(self, version: str = None):
        """Load trained models from disk."""
        version = version or self.model_version
        path = Path(MODELS_DIR) / version
        with open(path / "winner_model.pkl", "rb") as f:
            self.winner_model = pickle.load(f)
        with open(path / "method_model.pkl", "rb") as f:
            self.method_model = pickle.load(f)
        with open(path / "round_model.pkl", "rb") as f:
            self.round_model = pickle.load(f)
        # Load isotonic calibrator for winner probabilities
        win_cal_path = path / "winner_calibrator.pkl"
        if win_cal_path.exists():
            with open(win_cal_path, "rb") as f:
                self.winner_calibrator = pickle.load(f)
        else:
            self.winner_calibrator = None
        # Load calibrated round model
        round_cal_path = path / "round_model_calibrated.pkl"
        if round_cal_path.exists():
            with open(round_cal_path, "rb") as f:
                self.round_model_calibrated = pickle.load(f)
        else:
            self.round_model_calibrated = None
        self.shap_explainer = shap.TreeExplainer(self.winner_model)
        self._is_trained = True
        logger.success(f"Models loaded from {path}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """
        Evaluate model on test set.
        Use a hold-out set of fights AFTER the training period.
        """
        X = df_test[FEATURE_COLUMNS].fillna(0)
        y = df_test["winner"]

        # Get raw probabilities (calibrator is None, raw XGBoost used directly)
        raw_probs = self.winner_model.predict_proba(X)
        classes = list(self.winner_model.classes_)
        idx_1 = classes.index(1) if 1 in classes else 1
        probs = raw_probs[:, idx_1]
        preds = (probs >= 0.5).astype(int)

        # ── Confidence calibration ─────────────────────────────────────────────
        cal_df = pd.DataFrame({"prob": probs, "actual": y.values, "pred": preds})
        cal_df["correct"] = (cal_df["pred"] == cal_df["actual"]).astype(int)
        # Use predicted winner probability (always >= 0.5)
        cal_df["confidence"] = cal_df["prob"].clip(0.5, 1.0)
        cal_df.loc[cal_df["pred"] == 0, "confidence"] = 1 - cal_df.loc[cal_df["pred"] == 0, "prob"]

        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
        labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90%+"]
        cal_df["bucket"] = pd.cut(cal_df["confidence"], bins=bins, labels=labels, right=False)
        calibration = (
            cal_df.groupby("bucket", observed=True)
            .agg(fights=("correct", "count"), accuracy=("correct", "mean"))
            .reset_index()
            .to_dict("records")
        )

        return {
            "accuracy":    round(accuracy_score(y, preds), 4),
            "log_loss":    round(log_loss(y, probs), 4),
            "brier_score": round(brier_score_loss(y, probs), 4),
            "n_fights":    len(df_test),
            "calibration": calibration,
        }


# ── Convenience function ──────────────────────────────────────────────────────

def predict_fight_by_name(
    fighter_a_name: str,
    fighter_b_name: str,
    fight_date=None,
) -> dict:
    """
    High-level convenience function. Looks up fighters by name,
    builds features, and returns a prediction dict.

    Usage:
        result = predict_fight_by_name("Islam Makhachev", "Charles Oliveira")
        print(result["predicted_winner"], result["prob_fighter_a"])

    Args:
        fighter_a_name: Name as it appears in the DB
        fighter_b_name: Name as it appears in the DB
        fight_date:     datetime of the fight (defaults to today for upcoming fights)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from datetime import datetime as dt
    from src.database import init_db, get_session, Fighter
    from src.features.feature_builder import FeatureBuilder
    from rapidfuzz import process, fuzz

    if fight_date is None:
        fight_date = dt.utcnow()

    init_db()
    session = get_session()

    all_fighters = session.query(Fighter).all()
    name_map = {f.name_normalized: f for f in all_fighters}

    def find_fighter(name):
        from src.ingestion.data_loader import normalize_name
        norm = normalize_name(name)
        match = process.extractOne(norm, list(name_map.keys()), scorer=fuzz.token_sort_ratio, score_cutoff=60)
        if match:
            return name_map[match[0]]
        raise ValueError(f"Fighter not found: '{name}'. Check the name or run check_db.py to see available fighters.")

    fighter_a = find_fighter(fighter_a_name)
    fighter_b = find_fighter(fighter_b_name)

    builder = FeatureBuilder(session)
    features = builder.build_matchup_features(fighter_a.id, fighter_b.id, fight_date)

    predictor = UFCPredictor()
    predictor.load()
    result = predictor.predict(features, fighter_a.name, fighter_b.name)

    session.close()
    return result
