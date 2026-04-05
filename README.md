# 🥊 UFC Predictor

Machine learning system for UFC fight prediction, betting value detection, and EV-optimized parlay construction.

**Live dashboard:** [ufcpredictor.streamlit.app](https://ufcpredictor-z8o7owftbsjv5q47mepds7.streamlit.app)

---

## Results

| Metric | Value |
|---|---|
| Winner prediction accuracy (test set) | 65.0% (baseline: 50.0%) |
| Test set | 8,585 fights, 1994 → Mar 2026 |
| Live winner accuracy (3 events, 53 fights) | 75.5% out-of-sample |
| Live method accuracy | 55.6% |
| Live round O/U accuracy | 40.7% |
| 100-event parlay backtest ROI | +703% ($2,970 wagered → $23,840) |
| Safe parlay hit rate | 34.3% (expected: ~26%) |
| Value parlay hit rate | 18.2% (expected: ~10%) |

Live accuracy is tracked across every event using `scripts/log_live_results.py` — pure out-of-sample ground truth, not backtest.

---

## What It Does

- Predicts fight outcomes with win probability, method (KO/TKO, Submission, Decision), and round (over/under 2.5)
- Detects betting value by comparing model probabilities to live sportsbook odds
- Builds EV-optimized parlays across 4 tiers: Safe (3-leg), Value (5-leg), Shot, and Super (8-leg)
- Explains every prediction using SHAP feature importance — not a black box
- Tracks live accuracy with confidence calibration, weight class breakdown, and P&L simulation across events
- Auto-deploys to Streamlit Cloud on every git push

---

## How It Works

```
ufcstats.com (8,600+ fights)
        │
        ▼
Feature Builder — 63 pre-fight features, versioned snapshots, zero data leakage
        │
        ▼
XGBoost Models — winner, method, round (recency-weighted)
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

## Features (63 total)

All features are diffs (Fighter A − Fighter B) computed from stats available before the fight date.

| Group | Count | Notes |
|---|---|---|
| Physical | 4 | Reach, height, age, age vs peak |
| Striking | 4 | SLpM, accuracy, SAPM, defense |
| Grappling | 4 | TD avg/acc/def, sub avg |
| Record & Form | 5 | Win rate, finish rate, recent win rate, days since last fight, win streak |
| Elo Dynamics | 6 | Elo diff, avg opponent Elo, Elo trend, Elo uncertainty, Elo vs peak, momentum score |
| Style Fingerprints | 12 | Pressure, wrestling, striker, finisher (career + last-3 + last-5 rolling windows) |
| Durability | 3 | KD absorbed per fight, KD ratio, composite SAPM-based durability score |
| Strike Location | 3 | Head strike rate, leg strike rate, ground strike share |
| Cardio Decay | 2 | R3/R1 output ratio, early round output share (72.5% fighter coverage) |
| Style Vulnerability | 3 | Win rate vs wrestlers, strikers, pressure fighters |
| Interaction Features | 5 | TD success probability, striking edge, grapple dominance, finish threat, reach×accuracy |
| Weight Class Context | 2 | SLpM and TD percentiles within weight class |
| UFC Experience | 2 | UFC fights diff, UFC wins diff |
| Fight Context | 4 | Title fight flag, stance matchup (2 asymmetric flags), short-notice flag |
| Narrative | 3 | Sentiment, injury flags (placeholder — not yet automated) |

---

## Project Structure

```
ufc-predictor/
├── config.py                         # Central config (reads from .env or Streamlit secrets)
├── requirements.txt
│
├── data/
│   ├── ufc_predictor.db              # SQLite DB (8,600+ fights, 2,649 fighters)
│   ├── processed/training_dataset.csv
│   └── predictions/                  # Stored predictions + live accuracy log
│       └── live_accuracy.csv         # Per-fight live results tracker
│
├── src/
│   ├── database.py                   # SQLAlchemy schema (8 tables)
│   ├── ingestion/
│   │   ├── fight_scraper.py          # ufcstats.com scraper
│   │   ├── data_loader.py            # Fighter enrichment + deduplication
│   │   └── odds_scraper.py           # The Odds API integration
│   ├── features/
│   │   ├── feature_builder.py        # 63-feature matchup engineering
│   │   └── elo_calculator.py         # Chronological Elo ratings
│   ├── models/
│   │   └── predict.py                # XGBoost winner/method/round models (90% prob cap)
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
│   ├── train_model.py                # Model training + calibration report
│   ├── compute_styles.py             # Style fingerprints + rolling windows + percentiles
│   ├── compute_style_vulnerability.py # Opponent style vulnerability features
│   ├── scrape_fight_stats.py         # Round-level stats scraper (cardio decay)
│   ├── backtest_parlays.py           # Historical parlay backtesting
│   ├── log_live_results.py           # Live accuracy tracker (winner/method/round/P&L)
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
python scripts/run_pipeline.py
python scripts/log_live_results.py --event "Event Name"
python scripts/log_live_results.py --report
python scripts/compute_style_vulnerability.py
git add -f data/ufc_predictor.db models_saved/v1/ data/predictions/
git commit -m "post-event: [event name]"
git push
```

### Retrain (every ~10 events)
```bash
rm data/processed/training_dataset.csv
python scripts/train_model.py
git add -f data/ufc_predictor.db models_saved/v1/
git commit -m "retrain: [version notes]"
git push
```

### Backtest parlays
```bash
python scripts/backtest_parlays.py --events 100
```

---

## Dashboard

| Page | What it shows |
|---|---|
| 🥊 Upcoming Event | Full fight card with win probabilities, market edge, SHAP factors |
| ⚡ Value Bets | Fights where model edge > 5% vs market odds |
| 🎰 Parlays | 4 tiers: Safe, Value, Shot, Super — each with EV and Kelly sizing |
| Fighter Matchup | Head-to-head prediction for any two fighters |
| Elo Leaderboard | All-time rankings by Elo rating |

Odds are fetched once per session — tab switching does not consume API quota. Users can supply their own Odds API key in the sidebar, or use the host key if configured.

---

## Live Accuracy Tracker

`scripts/log_live_results.py` tracks every prediction against actual results:

- Winner accuracy by confidence bucket (calibration check)
- Method accuracy (KO/TKO vs Submission vs Decision)
- Round O/U accuracy (model said under 2.5, did it go early?)
- Accuracy by weight class and fight type
- Favorite vs underdog split (model vs market agreement)
- Flat $100 P&L simulation per event
- High-confidence miss patterns

### Current live calibration notes (3 events, 53 fights)
- 50–65% confidence: model is underconfident, hits higher than predicted
- 70–75% confidence: slight overconfidence (~5pp gap)
- Women's divisions: less reliable — smaller training sample, avoid betting until 20+ live results
- Round O/U: model overestimates finish rate (~36% actual vs ~45% expected)
- Miss pattern: "Predicted Decision → Actually KO/TKO" most common method error

---

## Betting Rules

Rules derived from live event data and model calibration:

1. **Avoid Heavyweight** — highest KO variance, single-punch outcomes undermine stats-based models. *Exception: dominant wrestler vs pure striker with no TD defense.*
2. **Never bet debut UFC fighters** — cold-start Elo and sparse stats make model unreliable. *Exception: fighter has extensive regional record with finishes against named opposition.*
3. **Skip fights where model and market disagree by >30%** — either odds are wrong or model is missing something. *Exception: you have specific human knowledge explaining the gap.*
4. **No negative-edge legs in parlays** — heavy favorites inflate payout while adding bookmaker juice. *Exception: small lottery-ticket parlays with bonus money only.*
5. **Minimum 65% model confidence to bet** — below this the model lacks conviction. *Exception: underdog value plays where model is 58-60% but market implies 35%.*
6. **Women's divisions need extra scrutiny** — model less calibrated, smaller training sample. *Exception: established veterans with 10+ UFC fights each.*
7. **Never parlay more than 5 legs with real money** — variance overwhelms edge beyond 5 legs. *Exception: small bonus bets treated as lottery tickets.*
8. **If line moves >20pts against your pick before betting, investigate** — sharp money may know something the model doesn't. *Exception: identified as public money, not sharp action.*

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (public required for free tier)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `main`, main file `dashboard/app.py`
4. Settings → Secrets → add `ODDS_API_KEY = "your_key"`
5. Deploy — auto-redeploys on every push

**Note:** The DB and model files must be committed to the repo for cloud deployment. `.env` should never be committed — use Streamlit secrets instead.

---

## Known Limitations

- Theoretical ceiling ~65–70% for MMA prediction using historical stats alone. Further gains require real-time information (camps, injuries, motivation) not available in public data.
- Narrative features are zeros — injury flags and sentiment not yet automated (3 of 63 features unused).
- Method and round models are trained independently — a joint model would be more accurate.
- Elo cold start — fighters new to UFC begin at 1500 regardless of regional record.
- Cloud DB is read-only — fighters not in the committed DB are skipped on Streamlit Cloud until next push.
- Round O/U systematically overestimates finish rate — discount UNDER predictions until further calibration data.
- Women's division calibration immature — treat predictions with lower confidence until 20+ live events.
- No camp/gameplan intel — structural limitation, not fixable with more data.

---

## Data Sources

| Source | What | How |
|---|---|---|
| ufcstats.com | All UFC fight history, fighter stats, round-level data | Scraper |
| the-odds-api.com | Live pre-fight odds | Free API |

---

## License

For personal and educational use.
