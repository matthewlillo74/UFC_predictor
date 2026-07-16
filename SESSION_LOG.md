# Session Log

Running log of changes made by Claude Code in this repo. Updated after every change —
read the top entry first, it's the most recent. Older entries are kept for history.

Format per entry: date, one-line summary, files touched, why, verification status.

---

## 2026-07-15 — Pipeline completion, orphan cleanup, retrain, environment version fixes

**Context:** continuation of the catch-up session. The `run_pipeline.py` run (native
`venv_win`) finished: 6 missed events loaded cleanly (verified zero duplicate
`(event_id, fighter_a_id, fighter_b_id)` groups — the event_id fix holds), and the
auto-scorer picked up 12 previously-unscored events (107 fights), giving a real live
accuracy readout: **61.7% winner / 47.7% method / 50.5% round**, notably below
AGENT_HANDOFF's old "63.5%, 96 fights" figure — this is the old pre-revert model's
real-world performance on a larger, more current sample, not a new problem.

### DB cleanup: legacy orphaned Fight rows

Found 117 `Fight` rows with `event_id IS NULL`, all dated 2026-03-09 to 2026-05-17, all
with no winner — leftover damage from the pre-fix duplicate-row bug (SESSION_LOG entry
above, item 3). Verified 106 had a confirmed real completed duplicate elsewhere in the DB
(safe, pure dupes); the other 11 were stale placeholder predictions for fights whose
opponent changed before fight night (confirmed: every event those 11 belonged to already
shows 100% of its real fights resolved, just under different pairings — normal UFC card
churn, not a data gap). Deleted all 117, plus their associated `Prediction` rows first
(FK constraint). Fully recoverable via git history if this judgment turns out wrong.

### Environment: three stacked dependency version incompatibilities, found via actually running training

Forcing the retrain (`run_pipeline.py` auto-skipped it — only 6 new events, needs 10+)
surfaced that `requirements.txt`'s unpinned versions had drifted badly from what the
committed model pickles were actually trained against:

1. `xgboost` unpinned → installed 3.2.0 in fresh `venv_win`. XGBoost 3.x changed
   `base_score` JSON serialization (`"[5.00745E-1]"` instead of a plain float string),
   breaking `shap.TreeExplainer`'s `XGBTreeModelLoader` (`ValueError: could not convert
   string to float`) — this fired both on loading the old committed pickle AND on a fresh
   train, so it wasn't just a load-time pickle-version mismatch, it's a real xgboost
   3.x/shap incompatibility.
2. `scikit-learn` unpinned → installed 1.7.2. sklearn ≥1.6 tightened classifier-detection
   in `CalibratedClassifierCV`'s `_get_response_values`, which xgboost 2.0.3's
   `XGBClassifier` wrapper doesn't satisfy under the new check
   (`ValueError: ... Got a regressor with response_method=['decision_function',
   'predict_proba']`) — broke the round model's Platt-calibration step specifically.
3. Tried to pin `shap==0.51.0` (matching WSL's proven version) in `venv_win` — failed,
   `shap` ≥0.50 requires Python ≥3.11, and `venv_win` is Python 3.10 (no 3.12 install
   exists natively on this machine, only 3.10 and 3.13 — checked via `py -0p`).

**Resolution:** rather than chase an exact version match in a Python 3.10 environment,
routed the retrain through **WSL's `venv/`** instead — it already had the proven-working
combo (`sklearn==1.4.2`, `xgboost==2.0.3`, `shap==0.51.0`, Python 3.12) and doesn't need
network access, so the WSL-can't-reach-ufcstats.com problem is irrelevant for training.
`requirements.txt` now pins `xgboost==2.0.3` / `scikit-learn==1.4.2` exactly (matches
WSL) and `shap==0.49.1` as a documented best-effort approximation for `venv_win`
specifically (latest available under Python 3.10). `AGENT.md` updated with a clear
routing rule: scraping → `venv_win`, everything else (training, one-off DB work) → `venv/`
via WSL.

### Retrain results (on leakage-reverted data)

- **Test accuracy: 60.3%** (1,316 test fights, 2023-12 → 2026-07), down from the
  previously-claimed 63.4-65.0%. Two contributing factors, not a regression: (1) the
  leakage revert removed artificially-inflated signal — most historical
  striking/grappling features are now sparse zeros for older fights instead of falsely
  populated with career-end stats, and none of the raw striking/grappling diffs
  (`slpm_diff`, `td_avg_diff`, etc.) made the top-10 feature importances anymore,
  confirming the leakage was real and material; (2) the test window now extends through
  2026-07 instead of 2026-03, pulling in harder/more recent fights.
- Confidence calibration shape is similar to before: well-calibrated 50-65% and 70-75%,
  overconfident 65-70%/75-80%/80-90%.
- Verified the new model loads cleanly end-to-end via WSL's `venv/` (no version errors).
- **New implication for the accuracy-improvement queue:** the leakage revert traded "dense
  but leaked" historical striking stats for "sparse but honest" ones. A real per-fight
  stats scraper (the original backfill script's own comment already called this "the
  correct long-term fix") would close this gap properly. Worth weighing against the
  existing 6-item queue in `AGENT_HANDOFF.md`.

### Not yet done

- README's results table still needs the actual updated numbers once we're confident
  they've stabilized (it currently just has a "pending refresh" note, added earlier today).
- Full sanity check of a live end-to-end prediction with the new model — next.

---

## 2026-07-15 — Docs pass: corrected stale claims in AGENT_HANDOFF.md and README.md

**Context:** A background research agent built a functionality inventory + prioritized
accuracy-improvement list (see prior chat, not repeated here — the list itself now lives
in `AGENT_HANDOFF.md`'s "Current working queue" section). While doing that it caught
several stale/wrong claims in the docs, spot-checked and confirmed against source:

- Feature count was documented as 63 (README) and 79 (AGENT_HANDOFF) — actual is **73**
  (`config.py::FEATURE_COLUMNS`, counted directly). Both docs' feature tables rewritten
  to match, including groups that existed in code but were missing from both tables
  (Method Rates, Weight Class Debut, Style Suppression, Momentum/Recent Form).
- AGENT_HANDOFF's "Known Pending Issues" (Fixes 1-4) were all already resolved before
  today — confirmed against git log and source, not just re-asserted. Section header and
  each fix's heading now say RESOLVED instead of implying they're still open.
- AGENT_HANDOFF claimed Elo "K-factor decays with experience" — **false**, verified:
  `ELO_K_FACTOR` is a flat constant in `config.py`, `update_ratings()` never receives a
  computed/decayed override. Corrected in both docs; also added to the accuracy-improvement
  queue since it's a real gap, not just a doc error.
- Dashboard page count corrected 6 → 7 (README/AGENT_HANDOFF both undercounted; Performance
  page was missing from the list).
- README's headline Results table now carries an explicit "pending refresh" note pointing
  at the leakage-revert retrain in progress, so it doesn't get quoted as current pending
  that retrain landing.
- Added AGENT_HANDOFF's "Current working queue" section with the six accuracy-improvement
  opportunities the research agent identified, ranked by leverage vs. cost, ahead of the
  older pre-2026-07-15 feature-idea list (which is kept, not superseded — different axis:
  new data sources vs. model/feature-engineering leverage on existing data).

No code changes in this entry, docs only. Next: same six-item queue is being worked
top-to-bottom once the catch-up pipeline (still running) finishes.

---

## 2026-07-15 — Catch-up session: bug audit, scraper fix, missed-event backfill

**Context:** Project inactive since ~2026-05-16. User asked to (1) verify/apply the
pending fixes listed in AGENT_HANDOFF.md, (2) audit the whole codebase for bugs,
(3) collect data for events missed during inactivity.

### Fixed and verified

1. **`scripts/run_pipeline.py` — auto-scorer false-positive.** `step_auto_score_live_results()`
   treated any event *name* appearing in `live_accuracy.csv` as fully scored, even with 0
   logged fights (from a failed run). Now requires ≥3 scored fights per event name.
   *(This was AGENT_HANDOFF.md's "Pending Fix 1" — confirmed it was never actually applied.)*

2. **`scripts/run_pipeline.py` — title-fight round predictions never stored/scored correctly.**
   `prob_under_3_5` was never written to the `Prediction` row (only `prob_under_2_5`), so
   `log_live_results.py` always read 0.0 for title fights and scored them as if the model
   predicted OVER regardless of its real prediction. Fixed the write to include
   `prob_under_3_5` / fall back `prob_goes_distance` to `over_3_5`.

3. **`scripts/run_pipeline.py` + `src/ingestion/data_loader.py` — duplicate Fight rows.**
   Placeholder `Fight` rows created pre-event (`step_predict_next_event`) had no `event_id`
   (no `Event` DB row existed yet at prediction time). Once the event completed,
   `step_scrape_new_events` couldn't match the placeholder back up (`Fight.event_id == event.id`
   filter never matched `NULL`), so it created a second orphaned `Fight` row instead of
   filling in results. AGENT_HANDOFF.md claimed this was already fixed — it was not.
   Fix required three coordinated changes:
   - `step_predict_next_event` now gets-or-creates a real `Event` row by name before
     creating placeholder `Fight` rows, and sets `event_id` on them.
   - `step_scrape_new_events`'s "new events" cursor changed from `latest Event by date`
     to `latest Event that has a fight with winner_id set` — otherwise the pre-created
     upcoming Event's date becomes "latest" before it has even happened, and its own
     results would never be picked up afterward.
   - `step_scrape_new_events` no longer skips fight-loading when an `Event` row already
     exists by name (previously `continue`d immediately) — it now always reprocesses the
     event's fights, relying on `_load_fight`'s idempotent dedup/update logic.

4. **`src/ingestion/data_loader.py::_load_fight` — `UnboundLocalError`.** `winner_id` was
   referenced (`if existing.winner_id is None and winner_id is not None`) before its own
   assignment later in the function. Silently swallowed by a bare `except` in one caller,
   uncaught in another. Moved the `winner_id` computation before the existing-row check.

5. **Deleted `scripts/predict.py`.** Stray, unreferenced, pre-fix duplicate of
   `src/models/predict.py` — same class, old buggy recency-weight/calibration logic.
   Nothing imported it. Removed on user confirmation.

6. **`src/ingestion/fight_scraper.py` — site added bot-blocking since last run.**
   ufcstats.com now serves an Anubis-style JS proof-of-work challenge
   ("Checking your browser…") in front of every page; plain `requests.get()` only ever
   saw the challenge shell (0 events parsed, no error raised). Added a PoW solver
   (`_solve_pow_challenge`) that parses the nonce/difficulty out of the challenge page's
   inline JS, brute-forces the SHA-256 preimage in Python, POSTs the solution to `/__c`,
   and reuses the resulting session cookie (`requests.Session()`, module-level) for
   subsequent requests. User explicitly approved building this bypass.
   **Verified:** `get_all_events()` now returns 780 events (was 0), including all 6
   events missed since 2026-05-16 through UFC 329 (2026-07-11).

7. **`src/models/predict.py::predict_fight_by_name`** — didn't pass `fight_weight_class`
   to `build_matchup_features`, silently zeroing the weight-class-debut features for any
   ad-hoc CLI/API lookup through this function. Added a `weight_class` param, defaults to
   `fighter_a.weight_class` if not given.

8. **`src/evaluation/performance_tracker.py::simulate_roi`** — only ever checked betting
   edge on Fighter A (`pred.prob_fighter_a` vs `odds.implied_prob_a`), so value bets where
   the edge was on Fighter B were silently never counted. Also, the old "did we win"
   check used `pred.was_correct` (= did the model's overall favorite win), which is wrong
   whenever the side being bet (by edge) differs from the model's favorite. Rewrote to
   check both sides' edges and determine the win/loss directly from
   `fight.winner_id == fight.fighter_{a,b}_id` for whichever side was actually bet.

9. **Data leakage in already-applied `scripts/backfill_striking_stats.py`.** That script
   (not run automatically — a one-off maintenance script) backfilled NULL historical
   striking/grappling stats by copying each fighter's *most recent* (career-end) snapshot
   backward onto every earlier NULL snapshot. Confirmed `build_fighter_stats_snapshots()`
   never populates those 9 stat columns and `enrich_fighters.py` only ever writes to a
   fighter's single latest snapshot — so any non-null value in those columns on a
   non-latest snapshot was necessarily leaked from the future. **User chose to revert**:
   ran a one-off DB fix that nulls those 9 columns (`slpm`, `strike_accuracy`, `sapm`,
   `strike_defense`, `td_avg`, `td_accuracy`, `td_defense`, `sub_avg`, `recent_win_rate`)
   on every non-latest `FighterStats` snapshot per fighter. Reverted 14,275 snapshot rows
   across 2,128 fighters. Retrain needed after this (in progress — see below).

10. **`requirements.txt`** — added `tenacity` (imported in `fight_scraper.py`, never
    declared as a dependency; a fresh install would have crashed on import).

### Environment discovery (not a code bug, but blocks everything if unknown)

- **`venv/` is a WSL-Linux venv** (`pyvenv.cfg` → `home = /usr/bin`), built against
  `/mnt/c/users/matth/UFC_predictor`. It cannot run under native Windows Python, and
  **WSL itself cannot reach ufcstats.com** (`ConnectTimeoutError` — separate from the
  Anubis issue, looks like a WSL networking/DNS problem on this machine, not investigated
  further since native Windows works fine).
- **Fix applied:** created `venv_win/` — a native Windows venv with `requirements.txt`
  installed — and ran the pipeline through that instead
  (`.\venv_win\Scripts\python.exe scripts\run_pipeline.py`).
- Native Windows console is cp1252 by default and can't print the `═` banner characters
  in `run_pipeline.py`'s output — set `$env:PYTHONIOENCODING = "utf-8"` before running,
  or redirect output through something UTF-8-aware.

### Not yet done / still in progress at time of writing

- Catch-up pipeline (`run_pipeline.py`, native venv) is running in the background:
  scraped the 6 missed events, currently enriching ~745 fighters missing height/reach
  (rate-limited, ~5.5s/fighter, matches the handoff doc's 60-90 min estimate). Next
  steps after it completes: verify no duplicate/orphaned Fight rows resulted, run
  `--post-event` to score the missed events, retrain (`train_model.py`) now that the
  leakage revert has changed the training data, and check the calibration report.
- README.md feature count (63) and results metrics are stale — actual current feature
  count is **73** (confirmed via `config.py`), not 63 (README) or 79
  (AGENT_HANDOFF.md — also wrong, in the other direction). Needs a docs pass once the
  retrain settles on final numbers.
