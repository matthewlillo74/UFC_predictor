"""
api/main.py
───────────
FastAPI application exposing UFC Predictor functionality via REST API.

Run with: uvicorn api.main:app --reload

Endpoints:
  GET  /health
  GET  /fighters/{name}
  GET  /upcoming-card
  POST /predict-fight
  GET  /value-bets
  GET  /performance
"""

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="UFC Predictor API",
    description="AI-powered UFC fight prediction system",
    version="1.0.0",
)


# ── Request / Response Models ─────────────────────────────────────────────────

class FightPredictionRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    fight_date: Optional[str] = None    # ISO format: 2025-03-15
    weight_class: Optional[str] = None


class FightPredictionResponse(BaseModel):
    fighter_a: str
    fighter_b: str
    prob_fighter_a: float
    prob_fighter_b: float
    predicted_winner: str
    confidence: float
    method_probabilities: dict
    round_probabilities: dict
    explanation: dict
    betting_value: Optional[dict] = None
    model_version: str
    predicted_at: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/fighters/{name}")
def get_fighter(name: str):
    """
    Return fighter profile and current stats.
    TODO: Query Fighter + FighterStats tables.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get("/upcoming-card")
def get_upcoming_card():
    """
    Return upcoming UFC card with all fights.
    TODO: Scrape upcoming card and return structured data.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.post("/predict-fight", response_model=FightPredictionResponse)
def predict_fight(request: FightPredictionRequest):
    """
    Generate prediction for a specific matchup.

    Example request:
        {
            "fighter_a": "Islam Makhachev",
            "fighter_b": "Charles Oliveira",
            "fight_date": "2025-06-01"
        }
    """
    # TODO:
    # 1. Normalize fighter names (fuzzy match to DB)
    # 2. Get fighter IDs
    # 3. Build features with FeatureBuilder
    # 4. Run UFCPredictor.predict()
    # 5. Attach odds data from BettingOdds table
    # 6. Return prediction
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get("/value-bets")
def get_value_bets(min_edge: float = 0.07):
    """
    Return current value bet opportunities where model edge exceeds threshold.
    TODO: Query upcoming fights, generate predictions, compare to odds.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get("/performance")
def get_performance_summary():
    """
    Return model performance metrics: accuracy, ROI, calibration.
    TODO: Query PerformanceTracker.get_summary()
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get("/leaderboard")
def get_elo_leaderboard(weight_class: Optional[str] = None):
    """
    Return fighter Elo ratings leaderboard.
    TODO: Query EloCalculator.get_leaderboard()
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
