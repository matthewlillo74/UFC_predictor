# UFC Predictor — Claude Code Agent Handoff

> **This document is a point-in-time narrative, not a live reference.** It was written
> after a multi-month inactive period and describes state as of that catch-up. Several of
> its claims went stale during the 2026-07-15 catch-up session — see `SESSION_LOG.md` for
> what actually changed and `AGENT.md` for the read-order/trust hierarchy across these docs.
> The "Known Pending Issues" section below (Fixes 1-3) is now historical — all three were
> already applied before 2026-07-15, this doc just never got updated to say so.
 
---
 
## Project Overview
 
End-to-end UFC fight prediction system. Goals: accurate predictions + betting value detection.
 
- **Repo:** github.com/matthewlillo74/UFC_predictor (public)
- **Live app:** ufcpredictor-z8o7owftbsjv5q47mepds7.streamlit.app
- **Local path:** `/mnt/c/users/matth/UFC_predictor/`
- **Language:** Python 3.12, XGBoost, SQLite, Streamlit
- **Activate venv first:** `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
---
 
## Current Model State

> **`main` runs the pre-queue baseline as of 2026-07-16, deliberately.** The full 6-item
> accuracy queue (73→80 features, per-division calibration, Elo K-factor decay) is real
> work and not deleted — it's preserved on branch `parked/accuracy-queue-2026-07-16` — but
> it didn't reach statistical significance (McNemar's test) and its calibration layer was
> found to actively harm probability quality (log loss 0.66→0.90 when applied). Given this
> weekend's closing-line data collection needs the most trustworthy state, not the most
> feature-rich one, `main` was reverted to 73 features / flat Elo / no calibration. See
> `SESSION_LOG.md` for the full decision trail and root-cause analysis of the calibration bug.

| Metric | Value |
|---|---|
| Features | **73** (XGBoost) — reverted to the pre-queue baseline 2026-07-16. Was 80 mid-session (full accuracy queue); that state is preserved on `parked/accuracy-queue-2026-07-16`, not deleted. |
| Test set accuracy | **60.6%** (baseline 49.1%) — pre-queue baseline, verified 2026-07-16 to exactly match the independently-reconstructed "state0" from the significance-testing script (see SESSION_LOG.md). The parked branch's 62.2% was NOT statistically confirmed better (closest p-value 0.083) and included a calibration layer since confirmed harmful — don't treat 62.2% as the better number without re-validating both findings first. |
| Live accuracy (107 fights, 11 events, scored 2026-07-15) | 61.7% winner / 47.7% method / 50.5% round — this reflects a **different, older model's** real-world predictions being scored (predates this session's leakage fix). No live results exist yet for the current (baseline) model. |
| Live winner accuracy — Featherweight | 80% (most reliable division) — pre-catch-up figure, not yet reverified |
| Live winner accuracy — Women's divisions | 25–47% (avoid betting) — pre-catch-up figure, not yet reverified |
| OVER 2.5 rounds backtest | 62.7% on 332 fights, +$5,470 P&L — pre-catch-up figure, not yet reverified |
| Method prediction | 46.2% — do NOT bet method props — pre-catch-up figure, not yet reverified |
| Training data | 8,771 fights, 1994–Jul 2026 |
| DB fighters | 2,677 |
 
---
 
## IMMEDIATE CATCH-UP SEQUENCE
 
**The project has been inactive for a few months. Run these commands in order before anything else.**
 
### Step 1 — Sync the DB with recent events
```bash
cd /mnt/c/users/matth/UFC_predictor
source venv/bin/activate
python scripts/run_pipeline.py
```
This scrapes all UFC events that happened while the project was inactive, enriches new fighters, recomputes styles and vulnerabilities, fetches odds, and generates predictions for the next upcoming event. It also auto-scores any unscored completed events.
 
Expect this to take 60–90 minutes due to fighter enrichment. Let it run fully.
 
### Step 2 — Score missed events and get live accuracy report
```bash
python scripts/run_pipeline.py --post-event
python scripts/log_live_results.py --report
```
 
### Step 3 — Check what's unscored
```bash
python -c "
import csv
from src.database import init_db, get_session, Event, Fight
from datetime import datetime, timezone
from config import PREDICTIONS_DIR
 
init_db()
s = get_session()
 
scored = set()
log_path = PREDICTIONS_DIR / 'live_accuracy.csv'
if log_path.exists():
    with open(log_path) as f:
        from collections import Counter
        counts = Counter()
        for row in csv.DictReader(f):
            name = row.get('event','').strip()
            if name and row.get('winner_correct','') != '':
                counts[name] += 1
        scored = {n for n,c in counts.items() if c >= 3}
 
now = datetime.now(timezone.utc).replace(tzinfo=None)
events = s.query(Event).filter(Event.date <= now).order_by(Event.date.desc()).limit(25).all()
print('UNSCORED:')
for e in events:
    if e.name in scored: continue
    n = s.query(Fight).filter_by(event_id=e.id).filter(Fight.winner_id.isnot(None)).count()
    if n > 0:
        print(f'  {e.date.date()}  {n} fights  {e.name}')
print('SCORED:', sorted(scored))
"
```
 
### Step 4 — Retrain with current data
```bash
rm data/processed/training_dataset.csv
python scripts/train_model.py
python scripts/backtest_props.py
```
 
### Step 5 — Commit everything
```bash
git add -f data/ufc_predictor.db models_saved/v1/ data/predictions/
git commit -m "catch-up: retrain + score missed events after inactive period"
git push
```
 
---
 
## Known Pending Issues — RESOLVED as of 2026-07-15

**Fixes 1-3 below were all already applied before the 2026-07-15 catch-up session** (confirmed
against `git log` and the actual source). This section is kept for historical context only —
do not re-apply these. Several *new* bugs were found and fixed instead that day, including two
this section didn't know about (a duplicate-Fight-row bug this doc previously claimed was
already fixed, and a title-fight round-prediction storage bug). Full detail in `SESSION_LOG.md`.

### PENDING FIX 1 — Auto-scorer bug (CRITICAL) — ~~pending~~ RESOLVED
**File:** `scripts/run_pipeline.py`
**Problem:** The auto-scorer in `step_auto_score_live_results()` reads `live_accuracy.csv` to detect already-scored events, but treats events with 0 fights logged (from failed scoring runs) as "already scored". This means 11+ events were silently skipped.
**Fix:** Change the scored_events detection from checking if event name exists to checking if at least 3 fights are logged for that event.
 
Find this block in `step_auto_score_live_results`:
```python
# Get already-scored events from CSV
scored_events = set()
if LIVE_LOG_PATH.exists():
    with open(LIVE_LOG_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored_events.add(row.get("event", ""))
```
 
Replace with:
```python
# Get already-scored events from CSV — only count events with at least 3 scored fights
scored_events = set()
if LIVE_LOG_PATH.exists():
    with open(LIVE_LOG_PATH, newline="") as f:
        reader = csv.DictReader(f)
        from collections import Counter
        event_fight_counts = Counter()
        for row in reader:
            event_name = row.get("event", "").strip()
            if event_name and row.get("winner_correct", "") != "":
                event_fight_counts[event_name] += 1
        scored_events = {name for name, count in event_fight_counts.items() if count >= 3}
```
 
### PENDING FIX 2 — Round calibration date bug (IMPORTANT) — ~~pending~~ RESOLVED
**File:** `src/models/predict.py`
**Problem:** Round model calibration used `pd.Timestamp.now()` (today's date) as anchor, but training data only goes to Nov 2023. "Last 2 years from today" finds 0 fights in training set, causing fallback to full dataset and a WARNING in the logs.
**Fix:** Use `max date in training data` as anchor instead of today.
 
Find this block in the `train()` method:
```python
recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
raw_dates = pd.to_datetime(df_clean["fight_date"])
if raw_dates.dt.tz is not None:
    raw_dates = raw_dates.dt.tz_localize(None)
df_recent_mask = raw_dates >= recent_cutoff
df_recent = df_clean[df_recent_mask]
```
 
Replace with:
```python
raw_dates = pd.to_datetime(df_clean["fight_date"])
if raw_dates.dt.tz is not None:
    raw_dates = raw_dates.dt.tz_localize(None)
max_train_date = raw_dates.max()
recent_cutoff = max_train_date - pd.Timedelta(days=730)
df_recent_mask = raw_dates >= recent_cutoff
df_recent = df_clean[df_recent_mask]
```
 
After this fix, training output should say:
`Round calibration — using 1019 recent fights, finish rate: 43.2%`
instead of:
`WARNING: Only 0 recent fights — using full dataset for calibration`
 
### PENDING FIX 3 — Recency weight date bug (IMPORTANT) — ~~pending~~ RESOLVED
**File:** `src/models/predict.py`
**Problem:** The recency weight calculation used `pd.Timestamp.utcnow().tz_localize(None)` which can fail on timezone-naive dates. Same root cause as Fix 2.
**Fix:** Use `pd.Timestamp.now()` and strip timezone explicitly.
 
Find:
```python
now = pd.Timestamp.utcnow().tz_localize(None)
fight_dates = pd.to_datetime(df_clean["fight_date"]).dt.tz_localize(None)
```
 
Replace with:
```python
now = pd.Timestamp.now()
fight_dates = pd.to_datetime(df_clean["fight_date"])
if fight_dates.dt.tz is not None:
    fight_dates = fight_dates.dt.tz_localize(None)
```
 
After applying fixes 2 and 3, retrain:
```bash
rm data/processed/training_dataset.csv
python scripts/train_model.py
```
 
### PENDING FIX 4 — Stale calibrator pickle warnings — ~~pending~~ RESOLVED (files never present)
**Problem:** Streamlit and training logs show sklearn version warnings about stale calibrator pickle files from old experiments.
**Fix:** Delete them:
```bash
rm -f models_saved/v1/winner_model_calibrated.pkl
rm -f models_saved/v1/winner_calibrator.pkl
```
Verified 2026-07-15: neither file exists in `models_saved/v1/` — nothing to do.
 
### PENDING FIX 5 — Deprecation warnings (LOW PRIORITY)
These don't break anything but clutter logs:
- `session.query(Model).get(id)` → replace with `session.get(Model, id)` throughout codebase
- `datetime.utcnow()` → replace with `datetime.now(datetime.UTC)` in `run_pipeline.py` and `dashboard/app.py`
---
 
## Standard Workflows
 
### After every event (AUTOMATED — just run these two)
```bash
python scripts/run_pipeline.py --post-event   # scores predictions, auto-scores missed events
python scripts/run_pipeline.py                 # scrapes next event, predicts, updates styles
git add -f data/ufc_predictor.db models_saved/v1/ data/predictions/
git commit -m "post-event: [event name]"
git push
```
 
### Retrain (every ~10 events or after adding features)
```bash
rm data/processed/training_dataset.csv
python scripts/train_model.py
python scripts/backtest_props.py
git add -f models_saved/v1/ data/ufc_predictor.db
git commit -m "retrain: [notes]"
git push
```
 
### Check live accuracy
```bash
python scripts/log_live_results.py --report
python scripts/log_live_results.py --report --events 5   # last 5 events only
```
 
### Props backtest
```bash
python scripts/backtest_props.py                    # full 2020+ holdout backtest
python scripts/backtest_props.py --year 2026        # current year only
python scripts/backtest_props.py --weight-class Lightweight
```
 
---
 
## File Structure
 
```
UFC_predictor/
├── config.py                          ← FEATURE_COLUMNS (79 features), paths, constants
├── requirements.txt
├── data/
│   ├── ufc_predictor.db               ← SQLite DB (committed to repo)
│   ├── processed/training_dataset.csv ← cached; delete before retrain
│   └── predictions/
│       ├── live_accuracy.csv          ← live results log (most important file)
│       └── UFC_*.txt                  ← prediction reports per event
├── models_saved/v1/
│   ├── winner_model.pkl               ← XGBoost winner model
│   ├── method_model.pkl               ← XGBoost method model (KO/Sub/Dec)
│   ├── round_model.pkl                ← XGBoost round model (early/late)
│   └── round_model_calibrated.pkl     ← Platt-scaled round model
├── src/
│   ├── database.py                    ← SQLAlchemy schema (8 tables)
│   ├── ingestion/
│   │   ├── fight_scraper.py           ← ufcstats.com scraper
│   │   ├── data_loader.py             ← fighter enrichment + deduplication fix
│   │   └── odds_scraper.py            ← The Odds API integration
│   ├── features/
│   │   ├── feature_builder.py         ← 79-feature matchup engineering
│   │   └── elo_calculator.py          ← chronological Elo ratings
│   ├── models/
│   │   └── predict.py                 ← XGBoost models, probability cap 90%
│   ├── betting/
│   │   ├── value_detector.py          ← edge detection + Kelly sizing
│   │   └── parlay_builder.py          ← 4-tier EV-optimized parlays
│   └── evaluation/
│       └── performance_tracker.py     ← prediction scoring
├── dashboard/
│   └── app.py                         ← Streamlit dashboard (7 pages: Upcoming Event, Value Bets,
│                                          Parlays, Props, Fighter Matchup, Performance, Elo Leaderboard)
└── scripts/
    ├── run_pipeline.py                ← master pipeline (pre + post event)
    ├── train_model.py                 ← training + calibration report
    ├── log_live_results.py            ← comprehensive live accuracy tracker
    ├── backtest_props.py              ← method + round prop backtest
    ├── compute_styles.py              ← style fingerprints + rolling windows
    ├── compute_style_vulnerability.py ← opponent style vulnerability features
    ├── scrape_fight_stats.py          ← round-level stats (cardio decay)
    ├── backtest_parlays.py            ← parlay backtest
    └── migrate_db.py                  ← DB schema migrations
```
 
---
 
## Feature Groups (79 total)
 
All features are diffs (Fighter A − Fighter B) computed from pre-fight data only.
 
| Group | Features | Notes |
|---|---|---|
| Physical | reach, height, age, age_vs_peak | age_vs_peak = distance from peak age for division |
| Striking | slpm, acc, sapm, def | 4 features |
| Grappling | td avg/acc/def, sub avg | 4 features |
| Record/Form | win rate, finish rate, recent win rate, days since, streak | 5 features |
| Elo Dynamics | elo, avg_opp_elo, elo_trend, elo_uncertainty, elo_vs_peak, momentum | 6 features |
| Style Fingerprints | pressure, wrestling, striker, finisher (career + l3 + l5) | 12 features |
| Durability | kd_absorbed, kd_ratio, composite_sapm | 3 features |
| Strike Location | head share, leg share, ground share | 3 features |
| Cardio Decay | r3/r1 output ratio, early round share | 2 features, 72.5% coverage |
| Style Vulnerability | winrate vs wrestlers/strikers/pressure | 3 features |
| Interaction | td_success_prob, striking_edge, grapple_dom, finish_threat, reach×acc | 5 features |
| Weight Class Context | slpm percentile, td percentile within division | 2 features |
| UFC Experience | ufc_fights_diff, ufc_wins_diff | 2 features |
| Fight Context | title fight, stance flags (2), short notice (2) | 5 features |
| Narrative | sentiment, injury flags | All 0 — not yet automated |
| Method Rates | ko_rate, sub_rate, decision_rate, ko_vulnerability | 4 features |
| Weight Class Debut | fighter_a_wc_debut, fighter_b_wc_debut | 1.0 if first fight at this weight class |
| Style Suppression | suppression_a, suppression_b, suppression_diff, clash_advantage | 4 features |
 
---
 
## Live Accuracy Calibration (96 fights, 9 events)
 
**Use this when evaluating which predictions to bet.**
 
| Confidence Bucket | Live Hit% | Status | Betting Rule |
|---|---|---|---|
| 50–55% | 54.2% | ✅ calibrated | small bets ok |
| 55–60% | 68.8% | 📈 underconfident | good value |
| 60–65% | 60.0% | ✅ calibrated | trust it |
| 65–70% | 90.9% | 📈 underconfident | **best bucket** |
| 70–75% | 58.8% | ⚠️ overconfident | discount by ~10pp |
| 75–80% | 42.9% | ⚠️ badly overconfident | avoid or heavily discount |
| 80–90% | 66.7% | ⚠️ overconfident | discount by ~15pp |
| 90%+ | 100% | ✅ calibrated | rare, trust |
 
**Division accuracy:**
- Featherweight: 80% ← most reliable
- Flyweight: 80%
- Bantamweight: 67%
- Heavyweight: 67%
- Light Heavyweight: 67%
- Middleweight: 67%
- Lightweight: 54% ← weak, avoid parlays
- Welterweight: 50% ← coin flip live
- Women's (all): 25–47% ← **never bet**
---
 
## Betting Rules (derived from live data)
 
1. **Avoid Heavyweight** — single-punch variance kills stats models. Exception: dominant wrestler vs pure striker.
2. **Never bet debut UFC fighters** — cold-start Elo, no real features. Exception: deep regional record with named opposition.
3. **Skip fights where model/market disagree >30%** — check for odds matching bugs first, then skip if real. Exception: you have specific human knowledge explaining the gap.
4. **No negative-edge legs in parlays** — heavy favorites add juice without adding value. Exception: small lottery-ticket bonus bets.
5. **Minimum 65% model confidence** — below this the model lacks conviction. Exception: underdog value where model is 58%+ but market implies 35%.
6. **Never bet women's fights** — confirmed across sufficient live sample (25–47% accuracy). No exceptions yet.
7. **Never parlay more than 5 legs with real money** — variance overwhelms edge. Exception: bonus bets treated as lottery tickets.
8. **If line moves >20pts against your pick before betting, investigate** — sharp money knows something. Exception: identified as public money not sharp action.
9. **Discount 75%+ model predictions by ~10pp** — live calibration shows systematic overconfidence in this range.
10. **OVER 2.5 rounds is your best prop bet** — 62.7% hit rate on 332 backtest fights, +$5,470 P&L. Bet when model says 60%+ OVER and book is at -110 to -115.
---
 
## Props Betting
 
Based on `backtest_props.py` results on 480 holdout fights (2020+):
 
| Prop | Hit Rate | P&L | Verdict |
|---|---|---|---|
| OVER 2.5 rounds | 62.7% | +$5,470 | ✅ BET — confirmed edge |
| UNDER 2.5 rounds | 48–52% | -$X | ❌ SKIP — no consistent edge |
| Method (any) | 46.2% | -$8,700 | ❌ SKIP — negative P&L |
| KO/TKO prop | 36–40% | very negative | ❌ Never bet |
| Submission prop | 37.5% | slight positive | ⚠️ Watch after retrain |
| Decision prop | 48–50% | negative | ❌ Skip |
 
**How to use OVER 2.5 props:** Check the Props tab on the dashboard. When a fight shows 60%+ OVER probability, check the real book line. If the book has it at -110 to -115, the model has +7-9% edge vs market. That's a bet.
 
---
 
## Architecture Notes
 
### Zero data leakage design
All features are computed from data available strictly before the fight date. `build_matchup_features(fighter_a_id, fighter_b_id, as_of_date, fight_weight_class)` takes an `as_of_date` parameter and all DB queries filter to `fight_date < as_of_date`. Never violate this — it's what makes test set accuracy meaningful.
 
### Prediction dict structure
`predictor.predict(features, fa_name, fb_name)` returns:
```python
{
    "fighter_a": str,
    "fighter_b": str,
    "predicted_winner": str,
    "confidence": float,          # max(prob_a, prob_b), capped at 0.90
    "prob_fighter_a": float,
    "prob_fighter_b": float,
    "method_probabilities": {
        "ko_tko": float,
        "submission": float,
        "decision": float,
    },
    "round_probabilities": {
        "under_2_5": float,
        "over_2_5": float,
    },
    "explanation": dict,          # SHAP values per feature
    "consistency": dict,          # method/round contradiction check
}
```
 
### Probability cap
Winner probabilities are capped at 90% max in `predict.py`. The model's raw output can reach 97-99% on debut fighters with sparse data — those extreme values are meaningless. The cap prevents overconfident displays.
 
### Duplicate fight row prevention
`data_loader.py` checks both fighter orderings when looking for existing Fight rows (A vs B and B vs A), and **updates the existing row with results** instead of creating a new one. This is critical — before this fix, the post-event scrape created duplicate Fight rows with different IDs, orphaning predictions and breaking live scoring.
 
### Fighter-ID fallback in log_live_results
`score_event()` has a two-level lookup:
1. Primary: `filter_by(fight_id=fight.id)` — works when pipeline ran correctly
2. Fallback: search sibling fights by fighter_a_id + fighter_b_id — catches orphaned predictions from before the data_loader fix
### Auto-scorer in run_pipeline
`step_auto_score_live_results()` is called automatically in both `--post-event` mode and regular mode (after scraping new events). It detects events with results in DB that have fewer than 3 rows in `live_accuracy.csv` and scores them. The threshold of 3 (not 0) prevents treating failed scoring runs as "already done".
 
### Elo system
- All fighters start at 1500
- K-factor is **flat** (`ELO_K_FACTOR=32` for everyone) on `main` as of 2026-07-16 — this
  is a deliberate revert to the pre-queue baseline, not a re-introduced bug. Experience-
  decayed K-factor (`decayed_k_factor(n)` in `elo_calculator.py`, ranges `ELO_K_MAX=48`
  debut down toward `ELO_K_MIN=20` veteran) exists and works — it showed the single
  largest accuracy gain of the accuracy queue (+1.1pp) — but that gain didn't reach
  statistical significance (McNemar's test, p=0.24), so it's parked on branch
  `parked/accuracy-queue-2026-07-16` along with the rest of the queue, not deleted.
  `scripts/recompute_elo.py` defaults to flat (matching `main`); pass `--decayed` only if
  deliberately working on the parked branch's feature set.
- `elo_uncertainty` = inverse of fights fought (high uncertainty = debut fighter)
- `elo_trend` = Elo change over last 3 fights
- `avg_opponent_elo` = strength of schedule proxy
- Cold start limitation: new UFC fighters start at 1500 regardless of regional record
### Weight class debut flag
`fighter_a_wc_debut` / `fighter_b_wc_debut` = 1.0 if this is the fighter's first fight at this weight class. Built by comparing current fight's `weight_class` to most recent previous fight's `weight_class`. Catches moves like Sterling to FW, Luque to MW, Costa to FW — all cases where model stats from old division are partially invalid.
 
### Style matchup suppression
`style_suppression_diff` = A's vulnerability to B's dominant style minus B's vulnerability to A's dominant style. Built from `winrate_vs_wrestlers/strikers/pressure` crossed with `dominant_style()` of opponent. Positive = A has style advantage. Addresses the Allen/Costa, Hooper/Gibson type misses where the model saw raw output stats but not how those stats collapse against specific opponent styles.
 
---
 
## DB Schema (key tables)
 
```
Event: id, name, date, url
Fight: id, event_id, fighter_a_id, fighter_b_id, winner_id,
       method, finish_round, finish_time, weight_class,
       is_title_fight, fight_date, fight_url
Fighter: id, name, ufc_url, weight_class
FighterStats: id, fighter_id, fight_date, [all stat columns]
              wins, losses, wins_ko_tko, wins_sub, wins_decision,
              losses_ko_tko, elo, elo_trend, elo_uncertainty,
              style_pressure, style_wrestling, style_striker,
              winrate_vs_wrestlers, winrate_vs_strikers, winrate_vs_pressure,
              cardio_decay, ko_vulnerability, durability_score, ...
BettingOdds: id, fight_id, implied_prob_a, implied_prob_b, odds_a, odds_b
Prediction: id, fight_id, predicted_winner_id, confidence_score,
            prob_fighter_a, prob_fighter_b,
            prob_ko_tko, prob_submission, prob_decision,
            prob_under_2_5, prob_over_2_5,
            was_correct, method_correct, round_correct,
            actual_winner_id, predicted_at
RoundStats: id, fight_id, fighter_id, round_num, sig_strikes_head, ...
```
 
---
 
## What Was Decided and Why (Design Decisions)
 
**Why XGBoost not neural net:** 8,700 fights is too small for deep learning. XGBoost with well-engineered features outperforms neural nets at this data size. Revisit when dataset exceeds 20,000 fights.
 
**Why SQLite not Postgres:** Single user, read-heavy, committed to repo for Streamlit Cloud deployment. No concurrent writes. No need to migrate.
 
**Why 63 → 79 features over time:** Each addition was tested and improved either test accuracy or addressed a specific live miss pattern. Features that didn't show up in top importances (method rates, debut flags) were kept because they provide targeted corrections on specific fight types even if globally weak.
 
**Why 2-year half-life for recency weighting:** 1-year half-life was tested and made 2026 accuracy worse. The problem is structural (missing narrative features) not a training parameter issue. Tightening recency weight destabilized the middle confidence buckets.
 
**Why probability cap at 90%:** Raw XGBoost reaches 97-99% on debut fighters where elo_uncertainty_diff pushes extreme values. We have no validated calibration data above 90% live, so capping prevents misleading displays. The 90%+ bucket does hit 100% on 3 live fights but sample is too small to trust uncapped.
 
**Why separate method and round models:** Simpler to train and debug than a joint model. Contradiction detection added (warns when method/round predictions logically conflict). Joint model is on the roadmap but low priority vs feature improvements.
 
**Why never retrain more often than every 10 events:** Each retrain risks overfitting to recent variance. The 2024–2025 accuracy is strong; aggressive retraining would sacrifice that stability for marginal gains on the most recent 20 fights.
 
---
 
## Next Features to Build (Prioritized)

### Current working queue (set 2026-07-15, see SESSION_LOG.md for what's landed)

Re-derived from a fresh codebase audit rather than the list below — grounded in what's
actually missing/dead in the current code, ranked by leverage vs. cost:

1. ~~**Wire up `control_time_secs`/`reversals` into features**~~ **DONE 2026-07-15** — see
   `SESSION_LOG.md`. Test accuracy 60.3% → 60.6%, real non-degenerate signal, didn't crack
   top-10 importances (modest gain, not a home run).
2. ~~**Opponent-quality adjustment on raw counting stats**~~ **DONE 2026-07-15** — see
   `SESSION_LOG.md`. Uncovered and fixed a much bigger problem along the way: the earlier
   leakage revert had left `slpm`/`sapm`/`td_avg`/etc. ~99.7% zero across the training set
   (not just sparser) since those columns were only ever written to a fighter's *latest*
   snapshot. Built a real per-fight backfill from `FightStats` (leakage-safe, ~85% coverage)
   to properly restore them, then added the 4 opponent-adjustment diffs on top.
   `sapm_adj_diff`/`slpm_adj_diff` now consistently rank top-10 in feature importance.
3. ~~**Division-specific calibration layer**~~ **DONE 2026-07-15** — see `SESSION_LOG.md`.
   Per-division Platt calibration fit for all 11 real divisions, verified to genuinely
   shift output probabilities. Not yet wired into `dashboard/app.py`'s 5 predict() call
   sites (safe no-op fallback) or `evaluate()` (bypasses calibration entirely, so this
   isn't visible in the quick train-time accuracy report). Caveat: smaller divisions'
   calibration samples are thin (e.g. Lightweight, 143 fights) — validate against live
   results before fully trusting.
4. ~~**SHAP-based aggregate miss-pattern analysis**~~ **DONE 2026-07-16** — see
   `SESSION_LOG.md` and `scripts/analyze_shap_misses.py`. Found real signal: Elo-family
   features (`elo_diff`, `elo_uncertainty_diff`, `avg_opponent_elo_diff`, `elo_trend_diff`)
   are 4 of the top 5 by total wrong-push across 15 high-confidence live misses —
   `elo_diff` alone pushed toward the wrong pick in 13/15. Direct empirical support for
   item 5 below, not just the theoretical case.
5. ~~**Elo K-factor decay by fight count**~~ **DONE 2026-07-16** — see `SESSION_LOG.md` and
   "Elo system" section above. **Test accuracy 60.1% → 61.2%, the largest gain of any
   queue item this session** — directly validated item 4's SHAP-driven prioritization.
6. ~~**Non-linear layoff penalty transform**~~ **DONE 2026-07-16** — see `SESSION_LOG.md`.
   **Test accuracy 61.2% → 62.2%, second-largest gain of the session** — much better ROI
   than its "trivial, 30 min" sizing suggested; `layoff_penalty_diff` cracked top-10
   feature importances.

**All 6 items in this queue are now done as of 2026-07-16.** Cumulative: 73 → 80
features, test accuracy 60.3% (right after the leakage revert alone) → 62.2%. See
`SESSION_LOG.md` for the full trail; a fresh accuracy-improvement pass should start from
scratch rather than extend this list further.

> **IMPORTANT — the per-item accuracy deltas above ("60.3% → 60.6%", "the largest gain of
> any queue item," etc.) are NOT statistically confirmed.** A proper McNemar's
> significance test (2026-07-16, see `SESSION_LOG.md` "Closing-line capture + statistical
> significance testing") found **none of the 6 items individually reach p<0.05**, and
> neither does the full cumulative 60.6%→62.2% effect (closest: p=0.083, still short of
> conventional significance). On a 1,316-fight test set, ~1pp swings are consistent with
> noise. Treat every accuracy number above as directionally suggestive, not proven — and
> note per-division calibration (item 3) measurably *worsened* log loss/Brier when tested
> this way, the opposite of its intended effect. Don't extend this queue further without
> addressing that finding and the closing-line validation (`scripts/capture_closing_odds.py`)
> first — accuracy alone doesn't establish a betting edge exists.

> **UPDATE 2026-07-16 (later same day):** given the above, `main` was reverted to the
> pre-queue baseline (73 features, flat Elo, no calibration) for the weekend's live
> closing-line test — the most trustworthy known state, not the most feature-rich one. The
> full 6-item queue described above is preserved, unchanged, on branch
> `parked/accuracy-queue-2026-07-16` (commit `d5b010c0`) — nothing was deleted. The
> calibration regression was root-caused (in-sample calibration fitting — a real bug in
> item 3's implementation, not a repeat of the significance-testing script's bugs) and
> confirmed harmful via direct test (0.65→0.72 log loss with calibration on). Revisit the
> parked branch once there's more live data to properly evaluate against; see
> `SESSION_LOG.md`'s "Calibration root-cause + revert to pre-queue baseline" entry.

Items 1-4 of the older list below (odds movement tracker, injury/camp NLP, CLV tracking,
joint method+round model) are still valid ideas and not superseded — this new list is about
model/feature-engineering leverage specifically, the older list is more about new data sources.

### Older list (pre-2026-07-15, not re-verified against current code)

### 1. Odds movement tracker (HIGH PRIORITY — easiest, most impactful)
Scrape odds from The Odds API daily in the week before each event. Store in a new `OddsHistory` table with timestamp. Compute `line_movement` = closing line minus opening line for each fighter. Large movement toward a fighter = sharp money knows something (injury in opponent camp, dominant sparring leaked, weight cut issues). This is narrative information for free.
 
Schema addition needed:
```sql
CREATE TABLE odds_history (
    id INTEGER PRIMARY KEY,
    fight_id INTEGER,
    timestamp DATETIME,
    odds_a INTEGER,
    odds_b INTEGER,
    implied_prob_a FLOAT,
    implied_prob_b FLOAT
);
```
Feature: `line_movement_a` = closing_implied_prob_a - opening_implied_prob_a
 
### 2. News scraper for injury/camp flags (MEDIUM PRIORITY)
Scrape MMA Fighting and ESPN MMA for headlines about each fighter on the upcoming card. Run through Claude API to extract structured signals: `injury_mentioned` (bool), `weight_cut_concern` (bool), `positive_camp_report` (bool). Updates `fighter_a_injury_flag` and `fighter_b_injury_flag` which currently sit at 0.0. This is the most direct path to making the narrative features do real work.

> **Feasibility-checked 2026-08-09, not built.** `mmafighting.com` and `espn.com/mma` both
> return clean 200s with no bot-wall like ufcstats' Anubis challenge — scraping itself is
> viable. Not attempted beyond that: a real build needs fighter-name matching against
> free-text headlines, an LLM extraction step, and — the part that actually matters —
> careful leakage-safety (only using news dated strictly before the fight, mirroring the
> `as_of_date` discipline everywhere else in this codebase). Worth its own session with
> proper verification, not a bolt-on. See `SESSION_LOG.md` 2026-08-09.
 
### 3. CLV (Closing Line Value) tracking (MEDIUM PRIORITY)
> **Built 2026-07-19 — this item is done, not a pending idea anymore.** `--clv` now
> exists on `log_live_results.py`, computing real CLV from `BettingOdds.is_opening`/
> `is_closing` rows rather than the manually-logged columns this section originally
> proposed (opening/closing capture is itself now automated too — see `SESSION_LOG.md`'s
> 2026-07-19 entry for both). No real CLV data exists yet — closing-odds capture's first
> live run (2026-07-18) captured nothing due to a since-fixed scheduling bug — so the
> first real numbers won't appear until whichever card captures cleanly, likely 2026-07-25.
> The description below is kept for the original design rationale, not as a to-do.

Add `--clv` flag to `log_live_results.py`. When logging a bet, record both the line at bet time and the closing line. After 30+ bets, compute average CLV to determine if the model consistently finds value before market corrects. Positive average CLV = the system is finding real edge, not just getting lucky.
 
New columns in `live_accuracy.csv`:
- `bet_placed` (1/0)
- `line_at_bet` (American odds)
- `closing_line` (American odds)
- `clv` = closing_implied_prob - bet_implied_prob
### 4. Inactivity/layoff non-linear transform (LOW PRIORITY — 30 min)
`days_since_last_fight_diff` is currently linear. Research shows performance degrades significantly after 12+ months. Add exponential decay beyond 365 days:
```python
def layoff_penalty(days):
    if days <= 365:
        return days / 365  # linear up to 1 year
    return 1.0 + (days - 365) / 365 * 0.5  # accelerating penalty beyond 1 year
```
Would have helped flag Choi (18 months out) and Sterling (coming back after loss).
 
### 5. Joint method + round model (LOW PRIORITY — bigger project)
Currently method and round are trained independently, which allows logical contradictions (60% KO in round 1 + 80% OVER 2.5 rounds). A joint model would output `(method, round_bucket)` as a single prediction, eliminating contradictions. Requires restructuring the training target and prediction pipeline. Estimated 1-2 days of work.
 
---
 
## Diagnostic Commands
 
```bash
# Check DB state
python -c "
from src.database import init_db, get_session, Event, Fight, Fighter, FighterStats
init_db()
s = get_session()
print('Events:', s.query(Event).count())
print('Fights:', s.query(Fight).count())
print('Fighters:', s.query(Fighter).count())
print('FighterStats:', s.query(FighterStats).count())
"
 
# Check most recent events and whether they have results
python -c "
from src.database import init_db, get_session, Event, Fight
init_db()
s = get_session()
for e in s.query(Event).order_by(Event.date.desc()).limit(10).all():
    fights = s.query(Fight).filter_by(event_id=e.id).all()
    n_results = sum(1 for f in fights if f.winner_id)
    print(f'{e.date.date()}  {n_results}/{len(fights)}  {e.name}')
"
 
# Check if predictions exist for upcoming event
python -c "
from src.database import init_db, get_session, Event, Fight, Prediction
init_db()
s = get_session()
e = s.query(Event).order_by(Event.date.desc()).first()
fights = s.query(Fight).filter_by(event_id=e.id).all()
preds = sum(1 for f in fights if s.query(Prediction).filter_by(fight_id=f.id).first())
print(f'{e.name}: {preds}/{len(fights)} fights have predictions')
"
 
# Check cardio decay coverage
python -c "
from src.database import init_db, get_session
from sqlalchemy import text
init_db()
s = get_session()
total = s.execute(text('SELECT COUNT(*) FROM fighter_stats')).scalar()
with_decay = s.execute(text('SELECT COUNT(*) FROM fighter_stats WHERE cardio_decay IS NOT NULL')).scalar()
print(f'Cardio decay: {with_decay}/{total} ({with_decay/total*100:.1f}%)')
"
 
# Verify new features are computing correctly
python -c "
from src.database import init_db, get_session
from src.features.feature_builder import FeatureBuilder
from datetime import datetime
init_db()
s = get_session()
builder = FeatureBuilder(s)
# Test with Jon Jones (id varies) vs any fighter
from src.database import Fighter
f1 = s.query(Fighter).filter(Fighter.name.like('%Jones%')).first()
f2 = s.query(Fighter).filter(Fighter.name.like('%Prochazka%')).first()
if f1 and f2:
    features = builder.build_matchup_features(f1.id, f2.id, datetime.now(), 'Light Heavyweight')
    print('style_suppression_diff:', features.get('style_suppression_diff'))
    print('style_clash_advantage:', features.get('style_clash_advantage'))
    print('fighter_a_wc_debut:', features.get('fighter_a_wc_debut'))
"
```
 
---
 
## Environment & Dependencies
 
```bash
# Python 3.12
# Key packages (all in requirements.txt):
# xgboost, scikit-learn, pandas, numpy
# sqlalchemy, loguru
# streamlit, plotly
# rapidfuzz (fuzzy name matching for odds)
# beautifulsoup4, requests (scraping)
# shap (SHAP explanations)
# python-dotenv
 
# Environment variables needed:
# ODDS_API_KEY=your_key_here        (from the-odds-api.com, 500 req/month free)
# DATABASE_URL=sqlite:///./data/ufc_predictor.db
 
# For local: create .env file in project root
# For Streamlit Cloud: Settings → Secrets
```
 
---
 
## Streamlit Cloud Deployment
 
- Auto-deploys on every `git push` to main
- Redeploy takes ~60 seconds
- DB and model files must be committed to repo (they are, tracked with git add -f)
- `.env` must NOT be committed — use Streamlit secrets panel
- The app serves whatever is in the committed repo — always push after post-event updates
---
 
## Important Gotchas
 
1. **Delete training_dataset.csv before retrain** — if it exists, training uses the cached version and ignores DB changes. Always `rm data/processed/training_dataset.csv` before `python scripts/train_model.py`.
2. **git add -f for data files** — `data/` and `models_saved/` are likely in `.gitignore`. Always use `git add -f` for these.
3. **Odds API quota** — 500 requests/month free tier. The pipeline uses ~10-15 per run. Don't run the pipeline repeatedly in a day unnecessarily. Check quota: logged in pipeline output as "API quota: X used, Y remaining".
4. **Fighter name fuzzy matching** — odds are matched to fighters using `rapidfuzz.token_sort_ratio` with threshold 75. When odds appear wrong (e.g. Wood at -238 when he should be +195), it's usually a matching bug. Always verify manually before betting.
5. **Weight class debut flag requires fight_weight_class** — `build_matchup_features()` must receive `fight_weight_class` parameter or debut flag returns 0.0 for all fights. Both the training dataset builder and `run_pipeline.py` pass this correctly, but any new callers must also pass it.
6. **The `--post-event` flag behavior** — only scores predictions and auto-scores live results. It does NOT scrape new data. Always run `python scripts/run_pipeline.py` (without flag) after to get next event predictions.
7. **Simon draw handling** — when a fight ends in a draw, most books void that leg from a parlay. The live logger treats draws as neither correct nor incorrect (skips them). Check your book's draw policy.
---
 
## Session Context
 
This handoff was written after ~4 months of active development across multiple Claude chat sessions. The owner (Matthew) is a UFC fan who bets recreationally with house money (profits from previous bets). Bankroll is small ($10–$30 range). The goal is a long-term system that compounds over many events, not a get-rich-quick tool.
 
The model is understood to have a ~65% ceiling using public stats alone. Narrative features (injuries, camps, motivation) would be needed to go higher. The owner understands and accepts this.
 
Key personality notes for interactions:
- Skeptical of model when it contradicts his fight knowledge (usually right to be)
- Prefers concrete commands he can run over abstract explanations
- Wants to understand *why* before accepting a recommendation
- Does not want to bet women's fights (confirmed by live data)
- Has ruled out Heavyweight as a reliable betting division
- Tracks all bets and results carefully
The most actionable next steps are: apply the pending fixes, run the catch-up sequence, then build the odds movement tracker. That's the highest ROI work available right now.
 