# 🥊 UFC Predictor

Machine learning system for UFC fight prediction, betting value detection, and EV-optimized parlay construction.

**Live dashboard:** [ufcpredictor.streamlit.app](https://ufcpredictor-z8o7owftbsjv5q47mepds7.streamlit.app)

---

## Results

| Metric | Value |
|--------|-------|
| Winner prediction accuracy | **66.2%** (baseline: 49.9%) |
| Test set | 1,286 fights, Aug 2023 → Mar 2026 |
| 100-event parlay backtest ROI | **+703%** ($2,970 wagered → $23,840) |
| Safe parlay hit rate | 34.3% (expected: ~26%) |
| Value parlay hit rate | 18.2% (expected: ~10%) |

---

## What It Does

- **Predicts fight outcomes** with win probability, method (KO/TKO, Submission, Decision), and round (over/under 2.5)
- **Detects betting value** by comparing model probabilities to live sportsbook odds
- **Builds EV-optimized parlays** across 4 tiers: Safe (3-leg), Value (5-leg), Shot, and Super (8-leg)
- **Explains every prediction** using SHAP feature importance — not a black box
- **Tracks live accuracy** with confidence calibration across events over time
- **Auto-deploys** to Streamlit Cloud on every git push

---

## How It Works

```
ufcstats.com (8,571 fights)
        │
        ▼
Feature Builder — 37 pre-fight features, versioned snapshots, zero data leakage
        │
        ▼
XGBoost Models — winner, method, round (recency-weighted, Platt-calibrated)
        │
        ▼
Value Detector — model prob vs market implied prob, vig removal, Kelly sizing
        │
        ▼
Parlay Builder — 4-tier EV-optimized construction
        │
        ▼
Streamlit Dashboard — live predictions, odds, edge, SHAP factors
```

---

## Features (37 total)

All features are diffs (Fighter A − Fighter B) computed from stats available **before** the fight date.

| Group | Count | Notes |
|-------|-------|-------|
| Physical | 3 | Reach, height, age |
| Striking | 4 | SLpM, accuracy, SAPM, defense |
| Grappling | 4 | TD avg/acc/def, sub avg |
| Record & Form | 5 | Win rate, finish rate, recent win rate, days since last fight, win streak |
| Elo Ratings | 2 | Elo diff + avg opponent Elo (strength of schedule) |
| Style Fingerprints | 5 | Pressure, wrestling, striker, finisher, grappling defense |
| Momentum | 2 | Recency-weighted win streak, recent finish rate |
| Weight Class Context | 2 | SLpM and TD percentiles within weight class |
| UFC Experience | 2 | UFC fights diff, UFC wins diff |
| Stance Matchup | 2 | Southpaw vs Orthodox (asymmetric flags) |
| Fight Context | 2 | Title fight flag, short-notice flag |
| Narrative | 3 | Sentiment, injury flags (placeholder — not yet automated) |

---

## Project Structure

```
ufc-predictor/
├── config.py                         # Central config (reads from .env or Streamlit secrets)
├── requirements.txt
│
├── data/
│   ├── ufc_predictor.db              # SQLite DB (8,571 fights, 2,637 fighters)
│   ├── processed/training_dataset.csv
│   └── predictions/                  # Stored predictions + live accuracy log
│
├── src/
│   ├── database.py                   # SQLAlchemy schema (8 tables)
│   ├── ingestion/
│   │   ├── fight_scraper.py          # ufcstats.com scraper
│   │   ├── data_loader.py            # Fighter enrichment + normalization
│   │   └── odds_scraper.py           # The Odds API integration
│   ├── features/
│   │   ├── feature_builder.py        # 37-feature matchup engineering
│   │   └── elo_calculator.py         # Chronological Elo ratings
│   ├── models/
│   │   └── predict.py                # XGBoost winner/method/round models
│   ├── betting/
│   │   ├── value_detector.py         # Edge detection + Kelly sizing
│   │   └── parlay_builder.py         # 4-tier EV-optimized parlays
│   └── evaluation/
│       └── performance_tracker.py    # Prediction scoring
│
├── models_saved/v1/                  # Trained model weights
│   ├── winner_model.pkl
│   ├── method_model.pkl
│   ├── round_model.pkl
│   └── round_model_calibrated.pkl
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard (5 pages)
│
├── scripts/
│   ├── run_pipeline.py               # Master pipeline (run before/after events)
│   ├── train_model.py                # Model training + era accuracy report
│   ├── enrich_features.py            # Backfill weight class + UFC experience
│   ├── compute_styles.py             # Style fingerprints + percentiles
│   ├── backtest_parlays.py           # Historical parlay backtesting
│   ├── log_live_results.py           # Live accuracy tracker
│   └── migrate_db.py                 # DB schema migrations
│
└── .streamlit/
    ├── config.toml                   # Theme config
    └── secrets.toml.template         # Secrets template (never commit actual secrets)
```

---

## Setup (Local)

```bash
git clone https://github.com/matthewlillo74/UFC_predictor.git
cd UFC_predictor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Create .env with your keys
echo 'ODDS_API_KEY=your_key_here' > .env
echo 'DATABASE_URL=sqlite:///./data/ufc_predictor.db' >> .env

# Launch dashboard
streamlit run dashboard/app.py
```

Get a free Odds API key at [the-odds-api.com](https://the-odds-api.com) (500 requests/month free).

---

## Workflow

### Before every event
```bash
python scripts/run_pipeline.py
```
Scrapes upcoming card → enriches fighters → computes styles → fetches odds → generates predictions.

### After every event
```bash
python scripts/run_pipeline.py --post-event
python scripts/log_live_results.py --event "Event Name"
python scripts/log_live_results.py --report
```
Scores predictions → logs live accuracy → shows confidence calibration report.

### Retrain (every ~50 events)
```bash
rm data/processed/training_dataset.csv
python scripts/train_model.py
```

### Deploy to cloud
```bash
git add -f data/ufc_predictor.db models_saved/v1/
git commit -m "post-event: [event name]"
git push
```
Streamlit Cloud auto-redeploys within 60 seconds.

### Backtest parlays
```bash
python scripts/backtest_parlays.py --events 100
```

---

## Dashboard

| Page | What it shows |
|------|---------------|
| 🥊 Upcoming Event | Full fight card with win probabilities, market edge, SHAP factors, AI analysis |
| ⚡ Value Bets | Fights where model edge > 5% vs market odds |
| 🎰 Parlays | 4 tiers: Safe, Value, Shot, Super — each with EV and Kelly sizing |
| Fighter Matchup | Head-to-head prediction for any two fighters |
| Elo Leaderboard | All-time rankings (Jon Jones 1774, GSP 1756, Makhachev 1755) |

Odds are fetched once per session — tab switching does not consume API quota.  
Users can supply their own Odds API key in the sidebar, or use the host key if configured.

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (public required for free tier)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `main`, main file `dashboard/app.py`
4. Settings → Secrets → add `ODDS_API_KEY = "your_key"`
5. Deploy — auto-redeploys on every push

Note: The DB and model files must be committed to the repo for cloud deployment. `.env` should never be committed — use Streamlit secrets instead.

---

## Known Limitations

- **Theoretical ceiling ~65–70%** for MMA prediction using historical stats alone. At 66.2% the model is near this ceiling. Further gains require real-time information (camps, injuries, motivation) not available in public data.
- **Narrative features are zeros** — injury flags and sentiment not yet automated (3 of 37 features unused).
- **Method and round models are independent** — contradiction detection added but a joint model would be more accurate.
- **Elo cold start** — fighters new to UFC begin at 1500 regardless of regional record.
- **Cloud DB is read-only** — fighters not in the committed DB are skipped on Streamlit Cloud until next push.
- **No camp/gameplan intel** — this is structural, not fixable with more data.

---

## Data Sources

| Source | What | How |
|--------|------|-----|
| ufcstats.com | All UFC fight history, fighter stats | Scraper |
| the-odds-api.com | Live pre-fight odds | Free API |

---

## License

For personal and educational use.
