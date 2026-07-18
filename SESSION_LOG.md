# Session Log

Running log of changes made by Claude Code in this repo. Updated after every change —
read the top entry first, it's the most recent. Older entries are kept for history.

Format per entry: date, one-line summary, files touched, why, verification status.

---

## 2026-07-16 — Accuracy queue item #6: non-linear layoff penalty transform

**What:** `days_since_last_fight_diff` was linear, but MMA performance degrades
disproportionately after 12+ months out (ring rust, harder to assess after a long
layoff) — already fully spec'd in `AGENT_HANDOFF.md`'s older feature-idea list, just
never implemented. Scoped as "trivial, ~30 min."

**Design:** implemented the formula exactly as already specified —
`_layoff_penalty(days)` in `feature_builder.py`: linear ramp to 1.0 at 365 days, then
continues growing at half the rate beyond that rather than plateauing. Added
`layoff_penalty_diff` to `FEATURE_COLUMNS` (79 → 80), additive alongside the existing
linear `days_since_last_fight_diff` rather than replacing it.

**Verified:** retrained — **test accuracy 61.2% → 62.2%**, the second-largest gain of
any queue item this session (right behind item 5's Elo fix). Log loss (0.6588) and
Brier score (0.2322) both improved to new session-best levels. `layoff_penalty_diff`
itself cracked the top-10 feature importances (rank 10) — genuinely useful signal, not
just added noise. Notably better ROI than its "trivial" sizing suggested going in.

---

## 2026-07-16 — Accuracy queue item #5: Elo K-factor decay by fight count

**What:** `ELO_K_FACTOR` was a flat 32 for every fighter regardless of experience — a
debut fighter's single result swung their rating exactly as much as a 25-fight
veteran's. Item 4's SHAP miss analysis directly motivated this: Elo-family features
were 4 of the top 5 by total wrong-push across live misses.

**Design:** added `decayed_k_factor(n_prior_fights)` to `elo_calculator.py` —
`k(n) = ELO_K_MIN + (ELO_K_MAX - ELO_K_MIN) / (1 + n / ELO_K_DECAY_FIGHTS)`, new config
constants `ELO_K_MAX=48`, `ELO_K_MIN=20`, `ELO_K_DECAY_FIGHTS=10` (debut fighters get up
to 1.5x the old flat K, asymptotes to 0.625x for veterans, roughly halfway there by ~10
fights). `update_ratings()` now takes optional `fights_a`/`fights_b` and computes a
per-side decayed K when given, falling back to the old flat behavior if not (backward
compatible). `data_loader.py::_load_fight` now looks up each fighter's prior fight count
via a DB count query (cheap — only a handful of fights per event during normal
incremental pipeline runs) and passes it through.

**Full historical recompute:** Elo is inherently chronological (each rating builds on
the fighter's prior one), so the decay can't be applied retroactively to existing rows —
the whole history has to be replayed from `ELO_BASE_RATING` in fight-date order. Built
`scripts/recompute_elo.py`: deletes all `EloRating` rows, replays all 8,771 fights
chronologically maintaining in-memory rating + fight-count dicts (O(n), not the O(n²) an
naive DB-count-per-fight approach would be over the full history). Confirmed
`elo_diff`/`avg_opponent_elo_diff`/etc. read directly from `EloRating` via
`EloCalculator` at feature-build time (not a `FighterStats` column), so no snapshot
rebuild is needed — just retrain.

**Verified:** 17,542 old rows deleted, 17,542 new ones written (2 per fight × 8,771,
matches). Sanity-checked a real fighter's career arc (Islam Makhachev, 18 UFC fights):
first-fight rating swing was ~26 points (debut, high K), late-career swings (fights
16→17→18) were only ~11 points each (veteran, low K) — decay working as designed.
**Test accuracy 60.1% → 61.2%** — the single largest gain of any queue item this
session. Log loss (0.6609) and Brier score (0.2334) both improved to new session-best
levels.

Re-ran `analyze_shap_misses.py` afterward as a secondary check — `elo_diff` still shows
up as the top culprit across the *same* 15 historical live misses. Worth being precise
about what this does and doesn't show: those 15 fights were already scored wrong by the
*old* model in `live_accuracy.csv`, a frozen historical record — re-explaining them with
the new model's SHAP values doesn't retroactively test whether the new model would call
them correctly now (that can only be measured going forward, via new live events). The
60.1% → 61.2% test-set accuracy jump is the real evidence this fix worked; the SHAP
re-check is a consistency check, not independent validation.

---

## 2026-07-18 — Fixed silent-failure risk in daily_pipeline.yml

**What:** user asked directly whether the automation was truly "set and forget." Honest
answer surfaced a real gap: several steps in `daily_pipeline.yml` use
`continue-on-error: true` so one hiccup doesn't block downstream steps (in particular, the
DB commit should still happen even if scoring or the email step had a problem) — but that
setting also makes GitHub report the *whole workflow* as successful even when a step
failed, which silently disables GitHub's free default failure-notification email for
scheduled runs. For a system explicitly meant to run unattended for months, that's exactly
backwards.

**Fix:** every step that uses `continue-on-error` now has an `id`; a final step checks all
their `outcome`s and explicitly `exit 1`s if any failed. Every step still runs to
completion (original goal preserved), but the job now correctly reports failure overall if
anything did — restoring the failure-email safety net. `closing_odds_poll.yml` didn't have
this problem (no `continue-on-error` on its capture step), so no fix needed there.

---

## 2026-07-18 — Email results report after each event (Gmail SMTP)

**What:** user wanted an email after each event summarizing results, mentioned Gmail API
specifically. Recommended Gmail SMTP + an App Password instead — the full Gmail API
(OAuth, token refresh, a Google Cloud project) is built for reading/managing mail, which
is overkill for a one-way "send me a report" use case; SMTP with an App Password does the
same job with far less setup.

**Built `scripts/email_report.py`** — deliberately thin, reuses `log_live_results.py`'s
`print_report(session, last_n_events=1)` for all the actual analysis (winner/method/round
accuracy, P&L, misses) rather than duplicating it, via the same dynamic-import pattern
`run_pipeline.py`'s auto-scorer already uses for the same module. Captures that function's
stdout output (it prints rather than returns a string) and emails it as plain text.

**Only sends once per event**, not once per (daily) pipeline run — tracks the last-emailed
event name in `data/predictions/.last_emailed_event.txt`, committed back to the repo like
the rest of the pipeline's output so state persists across GitHub Actions runs (each run
starts from a fresh checkout). Degrades gracefully to printing instead of emailing if the
Gmail secrets aren't set, rather than failing the pipeline.

Wired into `daily_pipeline.yml` as a step after scoring/prediction, before the commit —
`continue-on-error: true` so an email hiccup doesn't block the DB commit.

**Verified:** ran locally without Gmail credentials set — correctly identified the latest
scored event ("UFC Fight Night: Song vs. Figueiredo") from `live_accuracy.csv`, produced a
properly-scoped report (10 fights, matches expected count), and fell back to printing
cleanly. Deleted the resulting local state-file artifact before committing, since it could
otherwise incorrectly suppress the first real automated email in production if it happened
to match the actual latest event by coincidence.

**Requires user action, can't be done from code:** enable 2-Step Verification on the
sending Gmail account, generate an App Password, add `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD`/
`REPORT_EMAIL_TO` as GitHub Actions secrets. Full instructions in README.md.

---

## 2026-07-18 — Free cloud automation: GitHub Actions for closing-capture + daily pipeline

**What:** user wanted the closing-line capture and the regular pipeline to run
automatically, for free, without manually timing anything. GitHub Actions on a public
repo has no minute limit — no cost, no new signups.

**The core design problem:** cron can only express fixed schedules, but "T-60 minutes
before the next UFC card's first fight" moves every event. Solved with polling instead of
precise scheduling: a workflow runs every 15 minutes and checks a condition, only acting
when it's actually met — turns a dynamic-time requirement into something a fixed
scheduler can handle.

**Built `scripts/maybe_capture_closing.py`:** fetches odds, cross-references matched
fighters against real upcoming (unresolved) `Fight` rows in the DB — filtering out noise
from other MMA promotions that might appear in the same Odds API response — and takes the
earliest `commence_time` among confirmed matches as the actual first-fight time. (Our own
`Event.date` only stores a calendar date, no time-of-day, so it can't be used for this —
`commence_time` from The Odds API, already being parsed in `odds_scraper.py`, is the real
timestamp source.) Computes a 24-minute-wide window centered on T-60 (wider than the
15-min polling interval so a poll can't skip over it), checks idempotency (won't
double-capture if it fires more than once inside the window), and captures via the
existing `store_odds(..., is_closing=True)` if conditions are met. Costs 0 extra API quota
on runs that don't capture — the check and the capture share the same fetch.

**Two GitHub Actions workflows:**
- `.github/workflows/closing_odds_poll.yml` — runs the above every 15 min, restricted to
  Saturday 12:00 UTC through Sunday 06:00 UTC, not all week. (Updated same day, after the
  user correctly pointed out UFC is Saturdays only: a naive Saturday-only cron would have
  missed most US primetime cards, since GitHub Actions cron runs in UTC and prelims
  ~6-8pm ET land at 22:00-04:00+ UTC — crossing into Sunday. Also caught a quota problem:
  polling every 15 min for a full day, even once a week, would still burn ~2,880
  calls/month against the 500/month free cap without the day/hour restriction.)
- `.github/workflows/daily_pipeline.yml` — runs `run_pipeline.py --post-event` then
  `run_pipeline.py` (default mode, which has the auto-scorer built in) once daily,
  plus `compute_style_vulnerability.py`, matching the README's documented "after every
  event" sequence minus the one step that needs a manual event name.

Both commit DB/model changes back to the repo with `git pull --rebase` before push (low
risk of conflict between the two schedules, self-heals on the next run either way if it
does happen) — this is what triggers Streamlit Cloud's auto-redeploy downstream.

**Verified:** `maybe_capture_closing.py` tested against the live Odds API and current DB —
correctly fetched real odds, matched 42/56 fights by name, found zero corresponding to a
confirmed upcoming `Fight` row (none currently in the DB), and exited cleanly with no
capture and no error. Both workflow YAML files validated as syntactically correct.

**Not verified, flagged clearly in README:** whether ufcstats.com's Anubis bot-challenge
solver (built 2026-07-15) works from GitHub Actions' datacenter IP ranges — it's only ever
been run from a residential IP locally. Anti-bot systems sometimes treat known
cloud/datacenter ranges more aggressively regardless of whether the challenge itself is
solved correctly. Recommended testing both workflows manually via `workflow_dispatch`
before relying on the schedule.

**Requires user action, can't be done from code:** add `ODDS_API_KEY` as a GitHub Actions
secret (repo Settings → Secrets and variables → Actions).

---

## 2026-07-17 — Streamlit Cloud deploy failure: Python version drift

**What happened:** after pushing the baseline-revert, the user's Streamlit Cloud deploy
failed outright — dependency install error, `llvmlite==0.36.0` (a transitive `shap`
dependency) failing to build. Streamlit Cloud's deploy log showed it provisioning
**Python 3.14.6** — `llvmlite==0.36.0` only supports Python `<3.10`, and the cascading
resolver failure (falling back to ancient `numpy==2.0.0rc1` candidates) confirms the whole
dependency tree was trying to solve for a Python version far newer than anything the
pinned versions (`scikit-learn==1.4.2`, `xgboost==2.0.3`, `shap==0.51.0` — deliberately
pinned to the exact combo verified working together locally, see the 2026-07-15/16 entries
on xgboost/sklearn/shap incompatibilities) were ever built against.

**Root cause:** no Python version was pinned for the deploy environment at all — the repo
had no `runtime.txt` or `.python-version` file, so Streamlit Cloud used whatever its
current platform default is, which has apparently drifted to 3.14 over time. This is the
same *class* of problem as the local xgboost/sklearn/shap version-compatibility saga
earlier in the week, just surfacing on a different platform (Streamlit Cloud's infra)
with a different, even newer default Python.

**Fix:** added `runtime.txt` (`python-3.12`) — matches the WSL venv where the pinned
`scikit-learn==1.4.2`/`xgboost==2.0.3`/`shap==0.51.0` combination is confirmed to install
cleanly and work together (verified repeatedly throughout this session's local retrains).
Documented in `README.md`'s Deployment section so this doesn't get silently reintroduced —
flagged explicitly that if this fails again, check whether Streamlit Cloud's default
Python has drifted further and whether `runtime.txt` is still the authoritative mechanism
(Cloud's Advanced Settings UI may also need the Python version set directly).

**Not yet verified:** whether Streamlit Cloud actually honors `runtime.txt` in its current
form — this is a reasonable, standard fix but hasn't been confirmed against a successful
redeploy yet. If it doesn't take effect, the fallback is checking the app's Advanced
Settings in the Streamlit Cloud dashboard for an explicit Python version selector.

---

## 2026-07-16 — Calibration root-cause + revert to pre-queue baseline for weekend live test

**Context:** continuation of the same review. The friend's response to the significance
results flagged the item-3 calibration regression (0.66→0.90 log loss when tested) as more
urgent than the significance findings themselves — this weekend's whole point is comparing
model probabilities against the closing line, so a corrupted probability output would taint
the first day of real validation data. Asked directly: is this a methodology bug in the
testing script (like the 3 already found) or a real problem with calibration itself, and
would predictions be more honest this weekend with calibration on or off.

### Root cause, verified empirically (not just diagnosed by inspection)

Found it directly in `train()`: `X_div = X[use_mask]`, where `X` is the exact same data
`self.winner_model.fit(X, y_winner, ...)` was just trained on. `CalibratedClassifierCV(...,
cv="prefit")` only means "don't refit the base estimator" — it does **not** mean the
calibration set is held out. Fitting a calibrator on in-sample data is invalid methodology:
the base model's probabilities there are artificially confident relative to true
generalization, so the sigmoid mapping learned against them doesn't transfer.

Verified with a direct test (raw model vs. in-sample calibration vs. a properly held-out
calibration set the base model never saw):

| Approach | log loss | Brier |
|---|---|---|
| Raw (no calibration) | 0.6497 | 0.2291 |
| **In-sample calibration (current shipped code)** | **0.7229** | **0.2492** |
| Held-out calibration (proper fix) | 0.6526 | 0.2303 |

Confirms the diagnosis: in-sample calibration is clearly harmful, held-out calibration is
much closer to neutral (not clearly better than raw either, but nowhere near as damaging).
**This is a real bug in item 3's implementation, not a repeat of the three testing-script
bugs from the earlier entry** — it's in shipped production code. Answer given: calibration
should be off this weekend. A proper held-out-calibration-set fix is a legitimate follow-up
but wasn't rushed into production hours before the live test it would affect most.

### Decision: revert `main` to the pre-queue baseline for the weekend

Given (a) none of the 6 queue items reached significance, (b) calibration is confirmed
actively harmful, and (c) the weekend's closing-line comparison needs the most trustworthy
state available, not the most feature-rich one — asked the user directly what should
actually run. Chose the pre-queue baseline (matches the friend's recommendation).

**Preserved, not deleted:** created branch `parked/accuracy-queue-2026-07-16` at the tip of
all the queue + Part A/B testing work (commit `d5b010c0`) before touching anything on
`main`. Full queue is fully recoverable from there.

**On `main`:**
- `config.py::FEATURE_COLUMNS` reverted to the exact 73-column pre-queue set — verified by
  set-equality against the actual commit `b5c24214` config.py, not reconstructed from
  memory. (The opponent-adjustment, control_time/reversals, and layoff_penalty feature
  *code* stays in `feature_builder.py` — harmless, unused dead code now, not ripped out.)
- `scripts/recompute_elo.py` re-run in flat-K mode (now the documented main default; the
  script's CLI flipped from `--flat` to `--decayed` to match, and its warning language now
  correctly frames flat as the deliberate `main` state rather than an ablation-only mode).
- Calibration explicitly disabled at the two live-prediction call sites that had it
  (`run_pipeline.py`, `predict_fight_by_name`) by no longer passing `weight_class` to
  `predict()` — `train()` still fits calibrators harmlessly in the background (unused), a
  proper fix (gating the fit itself) is lower priority than making sure nothing applies them.

**Verified:** retrained — **60.6% test accuracy**, exactly matching the ablation script's
independently-verified `state0` reconstruction from the prior entry (strong consistency
check that this revert is faithful, not just "close enough"). End-to-end sanity check
(`predict_fight_by_name`) confirms 73 features, no calibration shift applied, consistency
check passes.

**Not yet done:** dashboard/app.py's 5 predict() call sites never passed `weight_class` in
the first place (from the original item-3 rollout), so they were already unaffected —
confirmed, not changed. Still holding the push per the friend's explicit guidance — push
once this is reviewed, not automatically after committing.

**Parked hypothesis, not a dead one:** the queue's individual effect sizes (Elo decay
+1.1pp, layoff penalty +1.1pp) are directionally consistent with the theoretical case for
each — p=0.083 on the combined effect is "close but not there," not evidence the ideas are
wrong. Revisit with more live data once the closing-line capture (this entry's sibling) has
accumulated enough for a real read, not before.

---

## 2026-07-16 — Closing-line capture (Part A) + statistical significance testing (Part B)

**Context:** external review (framed against a prior "insider-trading project" rigor
standard) flagged two gaps before trusting the accuracy-queue numbers: (1) no closing-line
data exists to check whether the model beats the market, so "accuracy" alone doesn't prove
a betting edge; (2) the queue's reported gains (e.g. "60.3% → 60.6%") were never tested for
statistical significance on ~1,300 test fights. Reviewed both parts, confirmed the premises
against the actual codebase, and built what was asked with the friend's explicit direction:
incremental (not isolated) ablation for Part B, excluding item 4 and the leakage fix from
testing scope.

### Part A — closing-line capture mechanism

Confirmed the premise directly: `is_closing` is **never set `True` anywhere** in the
codebase (`odds_scraper.py::store_odds()` hardcoded `is_closing=False` on every insert),
and only 57 unlabeled odds rows exist across 8,771 fights. No historical closing-line data
exists and none can be reconstructed after the fact — it can only be captured going
forward. Fixed the hardcoded `False` (now threads through properly) and built
`scripts/capture_closing_odds.py`, with an explicit operational definition: **T-60 minutes
before the first fight of the card**, documented in the script so future snapshots are
comparable to each other. Also confirmed `implied_prob_a/b` are already de-vigged at
storage time (`remove_vig()`, proportional method) — documented that as a known
simplification (not Shin's method) directly in `value_detector.py` per the review.

Not run yet — this weekend's card is the first real opportunity to capture a genuine
closing snapshot. `simulate_roi()`'s existing `is_closing == True` filter will start
actually matching rows once this runs, instead of always falling through to its "any
odds" fallback as it silently has been.

### Part B — incremental significance testing (McNemar's exact binomial test)

Built `scripts/ablation_significance_test.py` to reconstruct the accuracy-queue's
incremental states (pre-queue baseline → after item 1 → after item 2 → after item 5 →
after item 6, i.e. current production) and test each transition against the one before it
on the same 1,316 fixed test fights. Item 3 (calibration) tested separately via log
loss/Brier since it's a post-hoc probability rescaling that can never change the argmax
call. Items 4 and the leakage/sparsity fix excluded from scope per the friend's
sign-off (no predictions changed / comparing against a known-invalid baseline respectively).

**Three real bugs found and fixed while building this — the debugging process is worth
recording since it's a demonstration of exactly the kind of check the friend's review
was asking for (verify before trusting):**

1. **Calibration confound.** First version applied per-division calibration to the
   "after item 3/5/6" states and compared against historically-reported accuracy numbers
   that were *never* calibrated — `UFCPredictor.evaluate()` calls raw
   `winner_model.predict_proba()` directly and always has. Comparing calibrated ablation
   predictions against raw historical numbers is comparing two different things measured
   two different ways. First symptom: reconstructed "current production" scored 61.1%
   instead of the actually-committed 62.2%. Fixed by using raw, uncalibrated predictions
   throughout every McNemar comparison, matching `evaluate()`'s actual methodology exactly.

2. **Cross-run "contamination" (partially misdiagnosed, see #3).** After fixing #1, the
   mismatch persisted (61.2% vs 62.2%). Diagnostic: trained the identical "state 6"
   configuration alone in a fresh process — reproduced 62.2% exactly. Concluded (too
   quickly) that training 6 models sequentially in one process was the cause, and
   restructured the script to run each state in its own subprocess. This did **not**
   fully fix it (still 61.2%) — the isolated diagnostic that "confirmed" this hypothesis
   had accidentally changed two things at once (fresh process *and* no CSV round-trip),
   so process-isolation alone wasn't actually the fix. Kept the subprocess-per-state
   structure anyway since it's harmless and correctly rules out one variable, but the
   real cause was #3.

3. **Column-order mismatch (the actual root cause).** XGBoost's `colsample_bytree=0.8`
   samples columns **by index position**, not by name. The script built each state's
   feature list by concatenating hand-typed lists (`STATE0_COLS + new_item_cols`),
   appending each item's new columns at the end — but production's real
   `config.py::FEATURE_COLUMNS` has new columns *interspersed* near their logical
   predecessor (e.g. `layoff_penalty_diff` sits right after `days_since_last_fight_diff`,
   not at the very end). Identical column **set**, different **order** → the same
   `random_state=42` samples a completely different subset of columns per tree → a
   genuinely different, non-equivalent trained model despite "the same 80 features."
   Fixed by building every state's column list as a filter over production's actual
   `FEATURE_COLUMNS` order (`[c for c in FEATURE_COLUMNS if c not in excluded]`) rather
   than concatenation, and asserting `STATE6_COLS == FEATURE_COLUMNS` exactly. Verified:
   reconstructed state 6 now reproduces the committed **62.2%, log loss 0.6588, Brier
   0.2322** bit-for-bit.

**Final, verified results — reported plainly, including the parts that don't support the
queue's headline framing:**

| Transition | Accuracy | McNemar p-value | Significant? |
|---|---|---|---|
| 0→1 (control_time/reversals) | 60.6% → 60.4% (−0.2pp) | 0.867 | No |
| 1→2 (opponent-quality adj.) | 60.4% → 60.1% (−0.3pp) | 0.793 | No |
| 2→3 (calibration) | N/A (argmax can't change) | — | log loss/Brier both **worse** with calibration applied (0.662→0.901, 0.234→0.279) |
| 2→5 (Elo K-factor decay) | 60.1% → 61.2% (+1.1pp) | 0.239 | No |
| 5→6 (layoff penalty) | 61.2% → 62.2% (+1.1pp) | 0.231 | No |
| **0→6 (net, all 6 items)** | **60.6% → 62.2% (+1.6pp)** | **0.083** | **No** (closest to significant, doesn't clear p<0.05) |

**None of the six items individually reach conventional significance, and neither does
the full cumulative effect of the queue** — the overall 0→6 comparison gets closest
(p=0.083) but doesn't clear p<0.05. This matches the friend's own framing of what a
non-significant result means: not a failure, an honest answer given the sample size
(~1,300 test fights is a real constraint — detecting ~1pp accuracy differences with
adequate power typically needs a considerably larger paired sample than that).

The item 3 finding is the one that deserves the most attention going forward: per-division
calibration measurably **worsens** log loss and Brier score on this test, which is the
opposite of calibration's intended effect. This was measured on the state at which
calibration was actually introduced (flat-Elo era, matching the real historical build
order — item 3 shipped before item 5), so it's not a reconstruction artifact. Worth
investigating before continuing to trust the calibration layer's output, independent of
whatever Part A's closing-line data eventually shows.

**Not yet done:** re-testing calibration's effect on the *current* (decayed-Elo,
layoff-penalty-included) production model specifically — this measured it against the
state-2 configuration where it was first introduced, not the final state 6. Docs below
updated to stop presenting the queue's accuracy gains as confirmed improvements.

---

## 2026-07-16 — Accuracy queue item #4: SHAP-based aggregate miss-pattern analysis

**What:** SHAP was computed per-prediction for display only; nothing cross-referenced it
against `log_live_results.py`'s high-confidence-miss list to check whether misses share a
feature-level signature.

**Built:** `scripts/analyze_shap_misses.py` — pulls high-confidence misses (default
confidence >= 65%), recomputes SHAP values for each via `FeatureBuilder` + the trained
model's `shap_explainer`, and tallies which features consistently pushed the model
*toward* its wrong pick (sign-flipped so "push" always means "toward the mistake"
regardless of which fighter was favored). Reports features ranked by total push across
misses, with per-feature miss counts.

**Had to pivot the data source mid-build:** initially queried the `Prediction` DB table
(`was_correct == False AND confidence_score >= threshold`), but that table only has **2
rows total** — historical predictions got orphaned by the duplicate-Fight-row bug fixed
earlier this session, and apparently were never being reliably written to/read from the
DB table in the first place. `log_live_results.py`'s own reporting has relied on
`data/predictions/live_accuracy.csv` all along (107 fights) — rewrote the script to
source misses from there instead, resolving fighter names back to DB `Fighter` rows via
`normalize_name` + rapidfuzz (same pattern as `predict_fight_by_name`).

**Result — genuinely useful, not just a completed checkbox:** ran against all 107 live
fights, 15 misses at >=65% confidence. Elo-family features dominate: `elo_diff` pushed
toward the wrong pick in **13 of 15** misses (highest total push of any feature),
`elo_uncertainty_diff` in 7/15, `avg_opponent_elo_diff` in 10/15, `elo_trend_diff` in
11/15 — 4 of the top 5 features by total wrong-push are Elo-derived. This is direct
empirical evidence (not just the theoretical case already made) that queue item #5 (Elo
K-factor decay) is the right next fix — proceeding to it next with this in hand.

**Follow-up worth doing but not done here:** this is a one-off analysis script, not
wired into the regular pipeline or `log_live_results.py`'s report. Re-run manually after
future retrains to see whether the miss signature shifts.

---

## 2026-07-15 — Session paused after accuracy queue item #3

**Status:** user asked to pause here deliberately, not a stopping point forced by a
blocker. Working tree clean, 4 commits ahead of `origin/main` (not pushed — no auto-deploy
triggered). Plan: resume with queue items 4-6 (SHAP miss-pattern analysis, Elo K-factor
decay, layoff transform), then **step back and evaluate** the whole batch of changes
before deciding what's next — that evaluation hasn't happened yet, treat items 1-3 +
the priority sparsity fix as shipped-but-not-yet-fully-reviewed-in-aggregate.

Not done, flagged in earlier entries, still open:
- Division calibration (item 3) not wired into `dashboard/app.py`'s 5 predict() call
  sites, or into `evaluate()` (train-time accuracy report bypasses calibration).
- Live accuracy / parlay backtest numbers in README predate this session's fixes,
  marked "not yet reverified" rather than updated with new numbers.
- Two remaining minor bugs from the original codebase audit were never addressed:
  `get_early_label()` in `predict.py` uses `is_title_fight` instead of the more accurate
  `Fight.scheduled_rounds` column to pick the round-model threshold; `Fight.scheduled_rounds`
  itself is barely used elsewhere despite existing in the schema.
- `--full-retrain` flag and the model's committed pickles are current as of this session's
  last retrain (item #3's, log timestamp 2026-07-15 22:09) — no further retraining pending.

---

## 2026-07-15 — Accuracy queue item #3: per-division winner probability calibration

**What:** one global probability scale for the winner model despite documented live
accuracy spread by division (25-80%). The round model already had a working Platt
(sigmoid) calibration pattern — reused it per weight class instead of globally.

**Design:** in `src/models/predict.py::train()`, after the round model calibration
block, fit a `CalibratedClassifierCV(self.winner_model, method="sigmoid", cv="prefit")`
per division with ≥150 fights, using that division's recent fights (last 2 years, same
`df_recent_mask` the round model already computes) when there are ≥100 of them, falling
back to the division's full history otherwise — same recent-vs-full fallback logic as
the round model. Stored in `self.winner_calibrators_by_division: dict[str, ...]`,
persisted via `save()`/`load()` as `winner_calibrators_by_division.pkl`.

`predict()` gained an optional `weight_class` param: looks up a division-specific
calibrator first, falls back to the old global `winner_calibrator` (currently always
None, dead code path), falls back to raw probabilities if neither exists — fully
backward compatible, nothing breaks for callers that don't pass it. Updated the two
highest-value call sites (`run_pipeline.py::step_predict_next_event`,
`predict_fight_by_name`) to pass it. **Not yet updated:** the 5 call sites in
`dashboard/app.py` — left as a known follow-up since the fallback is safe (dashboard
predictions just won't get the calibration boost until updated).

Also fixed a latent interface bug while in there: the old `winner_calibrator` slot
expected a scalar `.predict([raw_prob])` interface, inconsistent with how the round
model's calibrator is actually used (`.predict_proba(X)` on the full feature matrix) —
never caught because it was always None. `predict()` now handles both interfaces so
old saved models with a populated scalar-style calibrator won't break, but new
calibrators (division or global) use the standard `predict_proba(X)` interface.

**Verified:** all 11 real divisions (excludes Open/Catch/Super Heavyweight and Women's
Featherweight — too few fights) got a calibrator fit. Confirmed calibration actually
changes output: same feature vector predicted 0.4451 (no weight_class/raw+cap), 0.2882
(Lightweight-calibrated), 0.4002 (deliberately wrong division, Heavyweight) — proves the
per-division lookup is real, not a no-op. **Caveat:** `evaluate()` (used for the
train-time accuracy/logloss report) calls raw `predict_proba` directly and bypasses
calibration entirely, so this feature's effect isn't visible in that report (accuracy
held flat at 60.1% as expected — calibration reshapes probability magnitudes, rarely
flips the argmax decision). Also: some divisions (e.g. Lightweight, 143 recent fights)
have a fairly small calibration sample — a 15.7pp swing on one test case could partly
reflect calibration noise rather than pure signal. Worth validating against live
results (`log_live_results.py`'s per-division calibration table) before fully trusting,
rather than a hard guarantee of correctness from this session alone.

---

## 2026-07-15 — Accuracy queue item #2: opponent-quality-adjusted counting stats

**What:** raw `slpm`/`td_avg`/`td_def`/`sapm` read identically whether earned against
weak or elite competition — only Elo and win/loss-based `style_vuln_*` accounted for
strength of schedule before this.

**Design:** added 4 new diffs (`slpm_adj_diff`, `td_avg_adj_diff`, `td_def_adj_diff`,
`sapm_adj_diff`) in `src/features/feature_builder.py`, scaling each raw stat by
`avg_opponent_elo / ELO_BASE_RATING` (already computed for the existing
`avg_opponent_elo_diff` feature, no new data needed). "Higher is better" stats
(slpm, td_avg, td_def) scale up with tougher opposition — same output against better
competition is more impressive. `sapm` (lower is better) scales inversely — keeping
absorbed strikes low against elite strikers is scaled to look even better. New `_opp_adj()`
helper returns `None` (not a fabricated 0) when either input is missing, so `_diff()`
falls back to 0.0 exactly like every other diff feature — no invented signal from
partial data. Additive, not a replacement — the raw diffs stay in the feature set too.

**Note:** this was built and first tested *before* the root-cause sparsity fix above
(previous entry) and initially looked neutral-to-negative, because it was scaling raw
stats that were themselves ~99.7% zero at the time (correlation with the raw diff was
0.999 — it was just inheriting the same near-total sparsity). After the real backfill
landed, `sapm_adj_diff`/`slpm_adj_diff` consistently show up in top-10 feature
importances. Kept `config.py::FEATURE_COLUMNS` at 73 → 77 (4 new) through this entry;
combined with item #1's 2 new features, total is now 79.

---

## 2026-07-15 — Root-cause fix: real per-fight backfill for slpm/sapm/td_avg/td_def/etc

**Context:** while building the opponent-quality-adjustment feature (queue item #2),
discovered the leakage revert from earlier today had a much bigger side effect than
understood at the time: `slpm_diff`/`sapm_diff`/`td_avg_diff`/etc. were **zero in 99.7%**
of training fights, not just "sparser." Root cause: `enrich_fighters.py` only ever wrote
those 9 columns to a fighter's single *latest* snapshot, so nulling the leaked
backward-copies on all non-latest snapshots left almost the entire historical dataset with
no striking/grappling signal at all. Flagged to the user; they chose to fix it properly
now rather than continue the original queue on top of broken data.

**The real fix:** `FightStats` has genuine per-fight granular data (~96% coverage,
16,780 rows / 8,771 fights × 2) that was never aggregated into snapshot-level rate stats.
Built `backfill_real_striking_grappling_stats()` in `scrape_fight_stats.py` — same
prior-fights-only leakage-safe pattern as the existing `backfill_kd_features()` /
`backfill_control_features()`. Offense stats (slpm, sapm, strike_accuracy, td_avg,
td_accuracy, sub_avg) come directly from the fighter's own `FightStats` rows; defense
stats (strike_defense, td_defense) need the opponent's attempted counts for the same
fight (preloaded once into a `fight_id -> {fighter_id: FightStats}` map to avoid N+1
queries). Also added `backfill_recent_win_rate()` — `recent_win_rate` was in the same
leaked-then-reverted column set but isn't `FightStats`-derived at all, just win/loss
history over the last `RECENT_FIGHTS_WINDOW` fights, always available.

**Two rounds of bugs found and fixed while verifying (not shipped silently):**
1. First backfill pass produced physically impossible values (`strike_defense` min
   -2.75, `slpm` max 34-56) — very-short fights (e.g. a 15-second KO as a fighter's only
   prior fight) blow up per-minute rate stats when the denominator is tiny. Added a
   1-minute duration floor plus hard clips (`slpm`/`sapm` ≤20, `td_avg` ≤15, `sub_avg`
   ≤10, defense/accuracy percentages clipped to [0,1]).
2. Second pass still showed values above the new caps. Root cause: the duration-floor
   and no-prior-fights skip paths did a bare `continue`, which leaves whatever a
   *previous* (pre-fix) run had already written in place — `continue` doesn't clear
   fields, it just doesn't update them. Added an explicit `_clear()` helper that nulls
   all 8 rate columns on every skip path and at the start of the normal compute path, so
   every run leaves a clean, fully-current state regardless of what a prior run wrote.
3. One remaining outlier (`td_avg=22.5` for a fighter with zero locally-scraped
   `FightStats` rows) is not a bug — that fighter is skipped entirely by this backfill
   (nothing to aggregate), so his snapshot still holds whatever `enrich_fighters.py`
   originally scraped from his UFC profile page directly. That's a genuine, if
   small-sample-noisy, officially-reported stat (UFC's own site computes TD-avg the same
   per-15-min way, which is noisy for fighters with very few octagon minutes) — not
   something this backfill touches or introduces. Affects 2 of 15,341 non-null rows.

**Verified:** coverage jumped from ~13% to 79-87% across all 9 columns (14,246-15,341 of
17,687 snapshots, depending on column). Value ranges now physically plausible after the
clipping fix. Retrained 3 times through this process (before clip fix, after clip fix,
after stale-value fix) — final state: **60.1% test accuracy**, log loss 0.6627, Brier
0.2345 (both within the best range seen all session; small fluctuations between the three
retrains are expected noise from feature-set changes affecting tree splits, not a
regression). `sapm_adj_diff` and `slpm_adj_diff` (queue item #2's opponent-adjustment,
built in parallel — see next entry) now consistently appear in top-10 feature
importances, confirming they were inert before only because the underlying raw stats
were empty, not because the idea itself was wrong.

**Files:** `scripts/scrape_fight_stats.py` (`backfill_real_striking_grappling_stats`,
`backfill_recent_win_rate`, `_fight_duration_minutes`, wired into `--backfill-all`).

---

## 2026-07-15 — Accuracy queue item #1: wired up control_time / reversals features

**What:** `control_time_secs` and `reversals` were scraped into `FightStats` (fight-level
totals) for a long time but never aggregated into a `FighterStats` snapshot feature —
identified in the earlier accuracy-improvement research pass as the highest-leverage item
since the hard part (scraping) was already done.

**Changes:**
- `src/database.py` — added `FighterStats.control_time_avg_secs` and `.reversals_per_fight`.
- `scripts/migrate_db.py` — added the two new columns.
- `scripts/scrape_fight_stats.py` — added `backfill_control_features()`, same
  prior-fights-only pattern as the existing `backfill_kd_features()` (leakage-safe: only
  aggregates fights strictly before each snapshot's `as_of_date`). Wired into
  `--backfill-all`. While in there, also removed ~55 lines of dead/duplicated code: a
  stray, unreachable copy of `backfill_kd_features()`'s body was sitting directly inside
  `backfill_strike_and_cardio_features()` after its own `session.commit()` (no `def`
  separating them), silently re-running the same KD computation a second time on every
  `--backfill-all` call. Harmless (deterministic, same result) but wasteful and confusing.
- `config.py` — added `control_time_diff` / `reversals_diff` to `FEATURE_COLUMNS` (73 → 75).
- `src/features/feature_builder.py` — wired the two new diffs in next to the existing
  `kd_ratio_diff` block.

**Verified:** ran migration (both columns added), ran `--backfill-all`
(14,923 snapshots updated), retrained. New features have real variance, not degenerate —
`control_time_diff` nonzero in 56.6% of training fights (std ~115s, range -707 to +650),
`reversals_diff` nonzero in 31.4% (reversals are naturally rarer events, std ~1.1, range
-11 to +22). Test accuracy 60.3% → **60.6%**; calibration improved slightly in the
75-90% confidence buckets (still overconfident, but less so). Neither new feature cracked
the top-10 importances — a modest, honest gain, not a home run, consistent with control
time being a secondary signal on top of the existing takedown-based grappling features.

**Docs:** marked item 1 done in `AGENT_HANDOFF.md`'s working queue.

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
