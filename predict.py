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
        X = df[FEATURE_COLUMNS].fillna(0)
        y_winner = df["winner"]
        y_method = df["method"]
        y_finish = (df["finish_round"] <= 2.5).astype(int)  # under 2.5 rounds

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
        self.winner_model.fit(X, y_winner)

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
        self.method_model.fit(X, y_method)

        # ── Round model (binary: finish early vs go late) ──────────────────
        self.round_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
        df_finish = df[df["method"] != "Decision"]
        if len(df_finish) > 50:
            X_finish = df_finish[FEATURE_COLUMNS].fillna(0)
            y_finish_filtered = (df_finish["finish_round"] <= 2.5).astype(int)
            self.round_model.fit(X_finish, y_finish_filtered)
        else:
            logger.warning("Not enough finish data for round model")

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

        # Winner probabilities
        win_probs = self.winner_model.predict_proba(X)[0]
        prob_a = float(win_probs[1])
        prob_b = float(win_probs[0])

        # Method probabilities
        method_probs = self.method_model.predict_proba(X)[0]
        method_classes = self.method_model.classes_
        method_dict = {cls: float(p) for cls, p in zip(method_classes, method_probs)}

        # Round probability
        round_probs = {}
        if self.round_model and self.round_model.n_features_in_:
            rp = self.round_model.predict_proba(X)[0]
            round_probs = {"under_2_5": float(rp[1]), "over_2_5": float(rp[0])}

        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(X)
        explanation = self._build_explanation(
            shap_values[0], FEATURE_COLUMNS, fighter_a_name, fighter_b_name
        )

        return {
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
        probs = self.winner_model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return {
            "accuracy": round(accuracy_score(y, preds), 4),
            "log_loss": round(log_loss(y, probs), 4),
            "brier_score": round(brier_score_loss(y, probs), 4),
            "n_fights": len(df_test),
        }
