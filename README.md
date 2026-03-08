# UFC Predictor

AI-powered UFC fight prediction system with explainability, betting value detection, and performance tracking.

## What It Does

- Predicts fight winners, method of victory, and round outcomes
- Explains every prediction with SHAP feature importance
- Compares model probabilities to sportsbook odds to detect value bets
- Tracks prediction accuracy and simulated ROI over time
- Generates human-readable analyst-style fight reports

## Architecture

```
UFC Card
    │
    ├── Fight Scraper          (Tapology / UFC Stats)
    ├── Odds Scraper           (The Odds API)
    │
    ▼
Feature Builder
    ├── Matchup diffs (reach, age, striking, grappling)
    ├── Elo ratings (strength of competition)
    └── Pre-fight snapshots only (no data leakage)
    │
    ▼
Prediction Model (XGBoost)
    ├── Winner probability
    ├── Method: KO/TKO | Submission | Decision
    └── Round: over/under 2.5
    │
    ▼
Explainability (SHAP)
    └── Top factors for/against each fighter
    │
    ▼
Betting Value Detector
    ├── Model prob vs market implied prob
    ├── Value gap and upset score
    └── Reverse line movement flag
    │
    ▼
Report Generator + Performance Tracker
```

## Project Structure

```
ufc-predictor/
├── config.py                     # All configuration (loaded from .env)
├── requirements.txt
├── .env.example                  # Copy to .env and fill in keys
│
├── data/
│   ├── raw/                      # Scraped data (never modify manually)
│   ├── processed/                # Cleaned datasets
│   └── predictions/              # Generated prediction reports
│
├── src/
│   ├── database.py               # SQLAlchemy ORM models (full schema)
│   ├── ingestion/
│   │   ├── fight_scraper.py      # UFC fight history + upcoming cards
│   │   ├── odds_scraper.py       # Betting odds + value detection
│   │   └── news_scraper.py       # MMA news + injury detection
│   ├── features/
│   │   ├── feature_builder.py    # Matchup feature engineering
│   │   └── elo_calculator.py     # Fighter Elo ratings
│   ├── models/
│   │   ├── predict.py            # XGBoost prediction model
│   │   └── train_model.py        # Training pipeline
│   ├── nlp/
│   │   └── sentiment_analysis.py # News + Reddit sentiment
│   ├── betting/
│   │   └── value_detector.py     # Odds comparison + upset detection
│   ├── evaluation/
│   │   └── performance_tracker.py # Accuracy + ROI tracking
│   └── explainability/
│       └── report_generator.py   # Human-readable fight reports
│
├── api/
│   └── main.py                   # FastAPI REST endpoints
│
├── dashboard/
│   └── app.py                    # Streamlit prediction dashboard
│
└── scripts/
    ├── run_pipeline.py           # Master pipeline (run before each event)
    └── train_model.py            # Model training script
```

## Setup

```bash
# Clone and enter
git clone <your-repo>
cd ufc-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -c "from src.database import init_db; init_db()"
```

## Running

```bash
# Run prediction pipeline for upcoming card
python scripts/run_pipeline.py

# Start API server
uvicorn api.main:app --reload

# Start dashboard
streamlit run dashboard/app.py
```

## Data Sources

| Source | What | How |
|--------|------|-----|
| ufcstats.com | Fighter stats, fight history | Scraper |
| Tapology | Fight cards, results | Scraper |
| The Odds API | Betting odds | Free API (500 req/month) |
| MMA Fighting / ESPN | News, injuries | Scraper |
| Reddit r/MMA | Sentiment | Reddit API (optional) |

## Key Design Decisions

**No data leakage**: All features use stats computed as-of the date *before* each fight. The `FighterStats` table stores versioned snapshots, not current stats.

**Matchup features**: All features are expressed as diffs (fighter_A - fighter_B) so the model learns matchup dynamics rather than absolute fighter quality.

**Elo ratings**: Every fighter has an Elo rating that updates after each fight, accounting for strength of competition.

**Predictions stored before results**: Every prediction is written to the DB before the fight happens. This is non-negotiable for honest performance tracking.

## API Endpoints

```
GET  /health
GET  /fighters/{name}
GET  /upcoming-card
POST /predict-fight
GET  /value-bets
GET  /performance
GET  /leaderboard
```

## Build Phases

- [x] Phase 1: Project scaffold + database schema
- [ ] Phase 2: Historical fight data scraper + Elo calculator
- [ ] Phase 3: Feature engineering pipeline
- [ ] Phase 4: Baseline prediction model + SHAP
- [ ] Phase 5: Betting odds + value detector
- [ ] Phase 6: News/sentiment layer
- [ ] Phase 7: Report generator
- [ ] Phase 8: Performance tracker + dashboard
- [ ] Phase 9: FastAPI + automation
