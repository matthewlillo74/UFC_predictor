# Session Log

Running log of changes made by Claude Code in this repo. Updated after every change —
read the top entry first, it's the most recent. Older entries are kept for history.

Format per entry: date, one-line summary, files touched, why, verification status.

---

## 2026-08-09 — Worked the "what's next" list: revisited parked queue, method-accuracy
## diagnosis, CLV status check, narrative-feature scoping, one data fix

**What:** after the data-integrity fixes earlier today, worked through five follow-up
items in order.

**1. Revisited `parked/accuracy-queue-2026-07-16` with fresh data — decision unchanged,
now with real evidence instead of a one-time snapshot.** Re-ran
`scripts/ablation_significance_test.py` (existing infra, untouched) against the current
DB — ~50 more real fights than the original July 16 run. Had to temporarily swap
`config.py` to the parked branch's 80-feature list (the script filters *down* from the
full set to reconstruct earlier states, so it needs that as the starting point) — verified
`models_saved/v1/*.pkl` untouched throughout (unchanged mtimes, no git diff), restored
`config.py` and Elo to production (flat-K) state immediately after, verified both restored
correctly and a live prediction still works. Results: **still no individual item reaches
significance** (item 1 p=0.876, item 2 p=0.436, item 5 Elo-decay p=0.204, item 6 p=0.870),
**overall net effect still not significant** (p=0.714, +0.5pp), and **calibration is still
clearly harmful — worse than before** (log loss 0.679→0.934, Brier 0.242→0.292, both worse
than the original 2026-07-16 finding). State6's accuracy on the current test split (59.7%)
doesn't reproduce the originally-reported 62.2% either, since the 85/15 split boundary has
shifted with more data — consistent with 62.2% having been at least partly a favorable-split
artifact rather than a robust improvement. **Decision: keep the current baseline (73
features, flat Elo, no calibration) deployed. Don't merge.** This was always meant to be
revisited with more data, not a permanent verdict — it now has been, twice, with the same
answer both times.

**2. Diagnosed (not fixed) the method-prediction weakness.** On the held-out test set, the
model's *average* predicted probability per class roughly matches true base rates
(49%/33%/18% predicted vs. 51%/31%/17% actual for Decision/KO/Sub) — but the argmax picks
Decision 87% of the time (1144/1316), because P(Decision) has a tight median/IQR of
0.498 [0.443, 0.540] that beats KO's 0.322 [0.284, 0.372] on nearly every fight regardless
of matchup specifics. Root cause candidate, from feature importances: `ko_rate_diff` and
`sub_rate_diff` — the features most directly relevant to *why* a fight would end in a
finish — rank only 45th/42nd of 73 by importance, while `ko_vulnerability_diff` contributes
essentially nothing (0.0000, rank 71/73). The model has diffuse, weak signal spread across
many features rather than a few strong differentiators, so it defaults toward the
population base rate almost everywhere. A real fix (rebalancing feature weight, possibly
class-weighted training) needs the same significance-testing rigor as item 1 — didn't
attempt it here to avoid roughly doubling this session's scope with an unvalidated change.

**3. CLV data status — healthy, not enough volume yet.** 29 opening / 24 closing odds rows
now (up from 15 total two events ago) — the capture pipeline continues working correctly.
Nothing to act on; just confirming it's still accumulating.

**4. Scoped (didn't build) narrative-feature automation** (injury/camp-report flags,
`AGENT_HANDOFF.md`'s older item 2). Confirmed feasibility — mmafighting.com and espn.com/mma
both return clean 200s, no bot-wall like ufcstats had. But a real build needs a new scraper,
fighter-name matching against free-text article headlines, LLM-based signal extraction,
careful leakage-safety (only using news dated before the fight), and a retrain — a genuine
multi-session feature. Deliberately did not rush a partial version into production; this
session has been largely about the cost of exactly that kind of shortcut.

**5. Fixed one cosmetic data issue.** Fighter id 2713 was stored as "Henrique Da Silva
Lopes"; ufcstats now displays the same profile (same fighter URL, confirmed) under "Jose
Montanha" — surfaced while reviewing HIGH-CONFIDENCE MISSES in an earlier report. Renamed.
Never affected matching or scoring (both keyed on fighter ID / URL, not display name).

**Verified:** every restoration step checked explicitly (model files, config.py,
Elo state, a live prediction, `check_data_integrity.py`) rather than assumed; all commits
pushed cleanly.

---

## 2026-08-09 — Added a daily data integrity check (follow-up to the bug-fix entry below)

**What:** the three result-tracking bugs in the entry below all shared one property — every
`daily_pipeline.yml` run showed a green checkmark the entire time the data underneath was
wrong. User's stated design goal for this project is "collect and save everything so I can
forget about it and come back whenever" — a check that only catches corruption when someone
happens to go looking defeats that. Built the automated version of the manual sweep used to
find the three bugs.

**Design:** `scripts/check_data_integrity.py` — pure DB read, two checks: (1) duplicate
`Fight` rows for the same `event_id` + fighter pair (should be structurally impossible given
`_load_fight`'s matching, but that's exactly the invariant that broke silently once), (2) any
event more than 5 days old with a still-unresolved fight. Deliberately detection-only, no
auto-fix — repairing a duplicate/stuck fight needs verification against the live ufcstats.com
page first (that's how all three bugs below got fixed correctly; blindly deleting/merging
without checking risks discarding the wrong row). Exits 1 on any finding so it fails the
GitHub Actions job loudly and triggers the default failure-notification email.

Wired into `daily_pipeline.yml`'s `run-pipeline` job as a new `continue-on-error: true` step
(consistent with the existing pattern — later steps like the DB commit still run even if this
fails), added to the final "fail the job if anything errored" check alongside the other
non-blocking steps.

**Verified:** ran against the now-clean DB (exit 0, no findings). Injected a synthetic
duplicate `Fight` row to confirm the check actually catches the exact failure mode from the
entry below (correctly flagged it, exit 1, with a clear message identifying the event and
fight IDs) — then deleted the synthetic row and re-ran to confirm it goes back to clean.
YAML re-validated after wiring in the new step.

---

## 2026-08-09 — Found & fixed three related result-tracking bugs across three events

**What:** user asked for a status/results check after several events had run unattended.
Investigating that surfaced real, active data corruption — not cosmetic.

**Three distinct bugs, all variants of "how do we find the existing Fight row for a
matchup," each breaking a different way:**

1. **`_load_fight`'s existing-match required `Fight.fight_date == event.date` exactly.**
   My own 2026-07-23 backfill (fixing that day's `get_upcoming_events()` date-parsing bug)
   updated `Event.date` for event 782 but never touched the 13 already-created
   `Fight.fight_date` values for that event — permanently breaking the match. Every
   pipeline run since created a fresh duplicate row per fight instead of updating the
   original: 25 rows for a 12-fight card, predictions stuck on the original rows,
   results stuck on new orphaned duplicates, `log_live_results.py` unable to score any
   of them directly (only worked via an existing sibling-fallback in `score_event()`).
   **Fix:** dropped the date-equality condition — matches on `event_id` + fighter pair
   only, which is what should have been used from the start.

2. **Opponent replacements (injury pullouts etc.) never cleaned up the stale pre-
   replacement `Fight`/`Prediction` rows.** Real, recurring occurrence — found it on
   three separate events (Jacoby's opponent changed 2026-07-18 session; two more fighters
   on the 2026-08-01 card; one on the 2026-08-08 card). The old row just sits there
   unresolved forever since the new opponent pairing doesn't match it. **Fix:** `_load_fight`
   now detects this — if neither fighter matches an existing row for the event, but either
   one has an unresolved row against a *different* opponent for the same event, that stale
   row (and its prediction) gets deleted before the real one is created.

3. **`step_scrape_new_events`'s cursor treated "an event has ≥1 resolved fight" as "this
   event is done, don't revisit it."** Combined with bug #2 (opponent-replacement fights
   resolve immediately on creation, while the stale original sits unresolved), an event
   could show "some resolved" long before it actually was — and once a *later* event got
   any result in, the cursor advanced past the earlier one and abandoned it. Confirmed
   this happened for real: the 2026-08-01 card sat at 2/16 resolved for over a week,
   stuck the moment the 2026-08-08 card got its first result. **Fix:** cursor now requires
   *every* fight in an event to be resolved before treating it as "done"; additionally,
   any event within the last 45 days that still has an unresolved fight gets re-checked
   on every run regardless of the cursor position.

**Data repair** (all three affected events, verified against the live ufcstats.com page
before every change, not assumed):
- Event 782 (Ankalaev vs. Guskov): merged 12 duplicate pairs (copied the result from the
  orphaned duplicate onto the original prediction-carrying row, deleted the duplicate).
  Also discovered the Dulatov vs. Turman fight was fully cancelled — no longer appears on
  the event page at all, not even as pending — deleted that row and its prediction rather
  than leaving a phantom unresolved fight. Final: 12/12 fights, all resolved, all with
  predictions intact.
- Event 783 (Medic vs. Rodriguez): deleted 2 stale opponent-replacement rows, then
  re-ran the fixed `_load_fight` against the real event page — all 12 previously-stuck
  fights resolved correctly on the first attempt. Re-ran `log_live_results.py --event`
  to score the 12 that were never logged (only 2/14 had been scored before, since the
  other 12 weren't resolved yet at the time). Final: 14/14 resolved, all logged.
- Event 784 (Gamrot vs. Salkilld): deleted 1 stale opponent-replacement row. Final: 12/12.

**Verified:** re-queried each event post-fix to confirm fight/resolved/prediction counts
match (12/12/12, 14/14/14, 12/12); re-ran `_load_fight` against live data and watched it
correctly match+update existing rows instead of creating duplicates; both code fixes
syntax-checked. This was working from live production data throughout, not a synthetic
test — each deletion was checked against the current ufcstats.com page first.

---

## 2026-07-25 — First real closing-odds capture; fixed a cross-workflow git-push race

**What:** live card day (UFC Fight Night: Ankalaev vs. Guskov, Abu Dhabi — an early
international-timeslot card, prelims ~10:00, main card ~12:00 per user report). User
reported `daily_pipeline.yml`'s `closing-odds-fallback` job failed with exit code 1.

**Good news first:** closing odds captured successfully for the first time ever under
this automation — 5 rows with `is_closing=True` for this event, confirmed via direct DB
query. The T-90/T-30 window + fallback design from 2026-07-19 is working.

**Root-caused the failure** (not guessed — pulled real job/step data via the Actions
REST API, `/repos/.../actions/runs/<id>/jobs`): the failure was specifically in the
"Commit and push if the DB changed" step, not the capture logic itself (which completed
successfully). Cross-referenced timing: `closing_odds_poll.yml` ran at 14:49:10Z the same
day; `closing-odds-fallback` (part of `daily_pipeline.yml`) ran at 14:57:52Z — 8 minutes
apart, easily overlapping given each run takes a few minutes. This is a real gap in the
2026-07-19 fix: `needs: run-pipeline` + `if: always()` on `closing-odds-fallback` only
sequences that job against `run-pipeline` *within the same workflow file* — it does
nothing to prevent a collision against `closing_odds_poll.yml` or `live_results_poll.yml`,
which are separate workflow files with independent triggers and no ordering relationship
to `daily_pipeline.yml` at all. All three commit to the same `data/ufc_predictor.db`, and
during the live-event window (when several of them are actually likely to have real
changes to push) a same-minute collision is entirely plausible — which is exactly what
happened.

**Fix:** replaced every single-shot `git pull --rebase && git push` (four of them, across
all three workflow files) with a 5-attempt retry loop (`git pull --rebase origin main &&
git push origin main && break`, with a small random 5-14s backoff between attempts to
reduce the chance of two racing retries colliding again). This is the correct fix for the
actual failure mode — sequencing one workflow's own jobs was necessary but not
sufficient; only a retry can recover from a collision against a workflow it has no
relationship to.

**Also cleaned up while investigating:** found and deleted a stale duplicate `Fight` row
(id 8908, "Uran Satybaldiev vs Dustin Jacoby") — Jacoby's real opponent changed to
Muhammad Saidov between the 2026-07-19 and 2026-07-21 scrapes (a real fight-card change,
not a scraper bug), but the old matchup's `Fight`/`Prediction` rows were never removed
since the get-or-create logic matches on fighter-ID pairs, which changed. Verified against
the live event page (`get_event_fights()`) before deleting — confirmed 13 real fights on
the card, Jacoby's current opponent listed as "Muhammad Saidov" (our DB has him as
"Muhammad Said" — a truncation worth watching; could cause a duplicate Fighter row when
results come in if `get_or_create_fighter`'s matching doesn't reconcile the two spellings,
not yet investigated further).

**Verified:** confirmed via the GitHub Actions REST API that the retry-loop change
doesn't alter any other step's behavior (YAML re-validated with `yaml.safe_load` across
all three files); manually exercised the loop's bash logic locally to confirm the
retry/break control flow is correct. Real-world proof of the retry logic itself will only
come from the next actual collision — by definition an intermittent race, not something
reproducible on demand.

---

## 2026-07-23 — Off-week gate for weekend polling; found & fixed a real date-parsing bug

**What:** asked to add a way to skip the weekend polling workflows entirely on weeks
without a UFC card (they were firing on their normal cron regardless, checking for
nothing). Building that gate required trusting `Event.date` to reflect the *real* fight
date — while verifying that, found it didn't: every auto-created upcoming `Event` row's
`date` was actually the timestamp of whichever pipeline run created it, not the real
card date. Root cause and full fix below.

**Root cause:** `fight_scraper.get_upcoming_events()` was reading `cells[1]` as the date
string — that's actually the **location** column. The real date lives in a
`<span class="b-statistics__date">` inside `cells[0]`, alongside the name link. Confirmed
by pulling the raw HTML directly: `cells[0]` contained
`"UFC Fight Night: Ankalaev vs. GuskovJuly 25, 2026"` (name + date concatenated, no
separator, since `link.get_text()` only pulls the `<a>` tag's text) — parsing `cells[1]`
("Abu Dhabi, Abu Dhabi, United Arab Emirates") as a date via
`strptime(..., "%B %d, %Y")` failed silently on **every single event**, caught by a bare
`except ValueError: date = None`, which `run_pipeline.py::step_predict_next_event` then
silently replaced with `datetime.utcnow()`. Also discovered in the same pass: this
function never included a `"location"` key in its returned dicts at all, despite
`step_predict_next_event` reading `event_data.get("location", "")` — so `Event.location`
has always been empty for auto-created events too.

**Fix:** `get_upcoming_events()` now reads the date from
`cells[0].find("span", class_="b-statistics__date")` and location from `cells[1]`
correctly. Verified live against ufcstats.com: correctly parsed real dates and locations
for all 8 currently-listed upcoming events (Ankalaev vs. Guskov → 2026-07-25, Abu Dhabi;
Medic vs. Rodriguez → 2026-08-01, Belgrade; etc.), none silently `None` anymore.

**Backfilled the two affected existing rows** (the bug only affects `Event` rows
auto-created via `step_predict_next_event`'s get-or-create path, which only fires for
each week's "next" event — checked all Event rows for the fallback's telltale non-
midnight timestamp signature, found exactly these two, nothing else affected):
event 781 ("Du Plessis vs. Usman", already-completed, cosmetic only) → 2026-07-18;
event 782 ("Ankalaev vs. Guskov", this Saturday, the one that actually matters for the
new gate below) → 2026-07-25, location "Abu Dhabi, Abu Dhabi, United Arab Emirates".

**Off-week gate:** new `scripts/is_ufc_weekend.py` — pure DB read (no network), checks
whether any `Event` (with `Fight` rows) falls within "this weekend" (Saturday through
Sunday UTC, computed correctly regardless of which day it's actually run — handles the
Sunday-UTC-poll-still-belongs-to-Saturday's-card case explicitly rather than always
walking forward to the *next* Saturday). Exit code 0/1 signals yes/no. Wired into all
three weekend-polling workflows (`closing_odds_poll.yml`, `live_results_poll.yml`,
`daily_pipeline.yml`'s `closing-odds-fallback` job) as their first real step — each
converts the exit code into a `has_event` step output via `if/then/else` (a raw nonzero
exit would otherwise fail the whole job and trigger GitHub's failure-notification email
on every off week, which is exactly the false alarm this needs to avoid) and gates every
subsequent step on it. `run-pipeline` (the main daily job) is deliberately NOT gated —
it does other daily work (scoring, retraining, predicting) unrelated to whether there's
a game this specific weekend, and is already a harmless no-op on quiet days.

**Verified:** caught and fixed a real bug in my own first draft of
`is_ufc_weekend.py` — the initial "days since Saturday" formula walked *backward* to the
most recent past Saturday, not forward to the upcoming one, so testing it on today
(Thursday 2026-07-23) incorrectly matched last week's already-completed card instead of
this Saturday's. Fixed and re-verified: correctly finds "Ankalaev vs. Guskov" for this
weekend now. Also verified the off-week path against a real gap already in the data (the
week of 2026-07-04 has no card) via a mocked `datetime.utcnow()` — correctly returned
`False` with the right message. YAML syntax of all three modified workflow files
confirmed via `yaml.safe_load()`, not just eyeballed.

---

## 2026-07-19 — Root-caused and fixed the closing-odds capture failure; wired up real CLV

**What:** a friend asked pointed questions about the "8/12 correct" email from the
2026-07-18 card: did closing-odds capture actually run, is there real data for those 12
fights, and why is the CLV section still placeholder text. Investigated properly instead
of assuming — pulled actual GitHub Actions run history via the public REST API (no `gh`
CLI available) rather than trusting workflow "success" status, which turned out to be
necessary: every run reported success, but success only meant `maybe_capture_closing.py`
exited cleanly on its normal "not in window" no-op path, not that anything was captured.

**Root cause, confirmed with evidence, not guessed:**
- `closing_odds_poll.yml`'s 11 runs that weekend landed at 1-3 hour gaps (17:56, 19:10,
  20:08, 21:07, 22:08, 23:11, 00:11, 02:44, 05:39, 07:50 UTC) against a configured 15-min
  cron — GitHub's `schedule` trigger drifting badly under load, a documented
  characteristic of free-tier Actions, made worse by 2-3 frequent-cron workflows
  competing in the same repo.
- The whole automation pipeline (`closing_odds_poll.yml`, `daily_pipeline.yml`, the email
  report) was only merged to `main` at 16:49-17:18 UTC on 2026-07-18 — confirmed via
  `git log` commit timestamps matching GitHub's own workflow `created_at`/`updated_at`
  metadata to the second. That left almost no runway before the card's actual capture
  window.
- Directly queried the DB: 0 `BettingOdds` rows for the event, any type, and the most
  recent row in the entire table predates 2026-05-01 — confirming zero captures, not
  just a formatting issue in the report.
- `--clv` was never wired to anything: grepped `log_live_results.py`'s argparse block —
  only `--event`/`--report`/`--events` exist. The CLV section was hardcoded placeholder
  text printed unconditionally, not a stub reading from `BettingOdds` that silently
  failed.
- Found in passing: `run_pipeline.py`'s own odds-fetch step called
  `fetch_and_store_odds(session)` with no `is_opening` kwarg, so even on a successful
  daily fetch, rows went in tagged neither opening nor closing.

**Fixes, sequenced as agreed (robustness before precision, cheap fix while fresh, build
now / prove later):**

1. **Widened the capture window + added redundancy**, not just retuned the cron offset —
   no cron-minute tuning fixes a scheduler that's inherently best-effort.
   `maybe_capture_closing.py`: window changed from a precise T-60±12min (24-minute) target
   to T-90-to-T-30 (60-minute) with a late-fallback past the T-30 deadline (capture
   immediately rather than risk missing the card, logged distinctly as `late fallback` so
   it's identifiable later). `daily_pipeline.yml` gained a new `closing-odds-fallback` job
   on its own separate schedule (22:30 UTC Sat / 02:30 UTC Sun, past T-30 for essentially
   any realistic UFC start time) — a structurally independent trigger (different workflow
   file, different cron registration) so a `closing_odds_poll.yml`-specific failure
   doesn't silently take out capture for the whole event. The `run-pipeline` job is gated
   (`if: github.event.schedule == '0 14 * * *' || ...`) so the two new fallback-only cron
   times don't also re-trigger the full daily scrape/retrain/predict cycle. Caught while
   implementing (not part of the original ask): both jobs write `data/ufc_predictor.db`,
   and with no ordering between them they'd run in parallel on the main 14:00 UTC trigger
   — risking the exact git-push race ("fetch first, rejected") this session already hit
   manually twice tonight, just automated and easy to miss. Added `needs: run-pipeline` +
   `if: always()` on `closing-odds-fallback` to force sequential execution without
   reintroducing the "skipped needed-job skips this job too" default GitHub behavior
   (which would've silently killed the fallback on exactly the two cron times it exists
   for).
2. **Fixed the `is_opening` bug.** `run_pipeline.py::step_fetch_odds` now checks, per
   matched fight, whether it already has any prior `BettingOdds` row before deciding
   `is_opening=True` vs leaving it untagged — `store_odds()` has no dedup, so a naive
   `is_opening=True` on every daily call would've mistagged every subsequent day's
   snapshot as "opening" too, for however many days lead up to the event. Verified
   locally: first fetch of a real upcoming card correctly tagged 10/10 new fights
   `is_opening=True`; re-running immediately after correctly reclassified all 10 as
   `routine` (no duplicate opening rows); a fight-level DB check confirmed exactly one
   `is_opening=True` row and one untagged row per fight, not two opening rows.
3. **Wired up real `--clv`.** New `_compute_clv()` in `log_live_results.py`: for each
   logged fight, re-identifies the DB `Fight` row (the CSV log has no `fight_id`, only
   event/fighter names — same re-identification approach `score_event()` already uses),
   requires both an `is_opening=True` and `is_closing=True` `BettingOdds` row for it
   (most fights won't have both yet), and computes `(closing_prob − opening_prob) /
   opening_prob` for the picked fighter. Positive = market moved toward the model's pick
   after the opening line, independent of whether the pick actually won. `--clv` flag
   added to argparse; placeholder print block replaced with real output (or an honest
   "no fights yet have both snapshots" message when `--clv` is passed but nothing
   qualifies — not a silent empty section). Verified against synthetic opening/closing
   rows injected into (and cleaned back out of) a real fight — both the
   picked-fighter-A and picked-fighter-B code paths matched hand-computed expected CLV%
   exactly; confirmed zero rows leaked into the DB after cleanup. Real proof still
   requires an actual card that captures cleanly — nothing to compute yet
   (`--clv` correctly reports that honestly rather than fabricating a number).

**Explicit tracking note (per discussion):** the 2026-07-18 card's "8/12 correct" result
is a valid, honest raw-accuracy data point (winner/method/round — unrelated to CLV). It
does **not** count as card #1 of the CLV series — it structurally couldn't produce a CLV
number (zero closing data captured that night). The CLV series starts from whichever
event is the first to capture opening + closing cleanly after this fix, likely next
weekend (2026-07-25). Don't let 2026-07-18 quietly get counted as CLV card #1 later.

**Verified:** all three fixes tested locally against real DB state (not just "code
compiles") — `step_fetch_odds` against the real 2026-07-25 card's odds (10 real fights
correctly separated from ~19 noise/hypothetical-matchup entries the Odds API also
returns), `--clv`'s math against synthetic rows with hand-verified expected values, the
new workflow YAML structure reviewed for correct job/schedule gating. Not yet verified:
an actual live GitHub Actions run of the widened window + fallback job against a real
card — that only happens for real on 2026-07-25, the next event.

---

## 2026-07-18 — Live automation verified end-to-end against a real card

**What:** "UFC Fight Night: Du Plessis vs. Usman" ran tonight — first real test of
everything built earlier today (per-fight DB updates, end-of-card summary email, the
closing-odds scraping gate, all of it) against an actual live event rather than a
not-yet-started one.

**Result:** worked as designed, no manual intervention needed. `live_results_poll.py`'s
summary email arrived ~1:30am with the correct 8/12 record and accurate per-fight
picks-vs-actual detail; `daily_pipeline.yml`'s broader analytics report arrived
separately the next morning on its normal 14:00 UTC schedule, no duplication or overlap
between the two. No false/premature results were written mid-card — confirms the
`_is_genuinely_concluded()` and `_should_start_scraping()` gates both held up under real
conditions, not just the pre-event dry runs. Model went 8/12 (66.7%) winner accuracy, in
line with historical performance; all 4 misses were low-confidence (50-56%) calls, none
high-confidence blunders. This closes out the "not yet exercised against a real card
conclusion" caveat noted in the last few entries.

---

## 2026-07-18 — Gated live-results scraping on closing-odds signal, dropped poll interval to 15 min

**What:** asked myself (proactively, not user-reported) whether the frequent-polling
design was actually well-tuned — it was scraping ufcstats.com every 5 min starting at
12:00 UTC Saturday regardless of when the card actually started, hours of pointless hits
(each one solving the site's bot-challenge) before anything was happening. User agreed to
tighten it.

**Design:** `_should_start_scraping()` in `live_results_poll.py` gates the real scrape
behind whether closing odds have already been captured for this event — `BettingOdds`
rows with `is_closing=True` for any of the event's fights. That signal is already free:
`closing_odds_poll.yml` sets it at T-60 min before the first fight, so reusing it here
costs zero extra Odds API calls (as opposed to re-deriving first-fight time via the Odds
API a second time, which would have roughly doubled quota usage against the existing
500/month cap). Added a time-based fallback (`FALLBACK_SCRAPE_HOUR_UTC = 20`, or any time
Sunday) for the case where closing-odds capture never fires for an event (API
down/quota exhausted that week) — better to scrape unnecessarily a few times than to
silently never track a card because one upstream signal failed. Also dropped the cron
interval from every 5 min to every 15 min, since the only time-sensitive thing left is
the one end-of-card summary email, not per-fight promptness.

**Verified:** ran locally against the real in-progress event (0 `BettingOdds.is_closing`
rows for it yet, confirmed via direct query) — correctly skipped the scrape this run
instead of hitting ufcstats.com, logging the reason clearly.

---

## 2026-07-18 — Live results email switched from per-fight to end-of-card summary

**What:** user tried the per-fight version live and didn't want an email after every
individual fight — wants one email once the whole card is over, listing every pick vs.
actual outcome.

**Design:** `scripts/live_results_poll.py` still polls every 5 min and updates the DB
fight-by-fight the same way (same `_is_genuinely_concluded()` safety gate, unchanged),
but no longer emails inside that loop. Added `_event_fully_resolved()` — checks
`finish_round IS NULL` across all of the event's fights, not `winner_id IS NULL` (a draw/
NC never gets a `winner_id`, so keying off that would make the script think a card with a
draw on it never finishes). Once a run's DB update leaves the card fully resolved, sends
one summary (`_format_event_summary_email()`: every fight's pick vs. actual, plus an
overall N/M-correct tally) and marks the event emailed in the new
`data/predictions/.emailed_event_ids.txt` (replaces the old per-fight
`.emailed_fight_ids.txt`, no longer used).

Handled the crash-before-email edge case: if a run resolves the card's last fight but
dies before the email send (network blip, workflow timeout), the next run's primary query
(events with an unresolved fight) would find nothing, since the DB already shows it fully
resolved — silently losing the summary forever. `main()` falls back to checking the
single most recent `Event` overall when the primary query comes up empty; if it's fully
resolved and not yet in `.emailed_event_ids.txt`, sends the summary then instead.

**Verified:** script runs clean against the real in-progress event (still mid-card at
edit time) — no fights fully resolved yet, so no email sent this run, matching expected
behavior. The "send exactly one summary when the last fight resolves" and "recover a
missed send on the next run" paths are both new logic, not yet exercised against a real
card conclusion — first real test is whenever tonight's card actually finishes.

---

## 2026-07-18 — Per-fight live results email + free cloud automation completed

**What:** user asked for the results email to fire after each individual fight during
a live card (comparing model prediction to actual outcome), not just once at the end of
the whole event like the existing `email_report.py`/`daily_pipeline.yml` does.

**Design:** new `scripts/live_results_poll.py`, meant to run frequently (every 5 min)
via a new `.github/workflows/live_results_poll.yml` during the Sat 12:00 UTC – Sun 06:00
UTC live-event window (same reasoning as `closing_odds_poll.yml`'s schedule). Scrapes the
current event's page via the existing `fight_scraper.get_event_fights()`, and for each
fight not yet resolved in the DB, checks a hard safety gate before trusting the scraped
result at all: `_is_genuinely_concluded()` requires both `finish_round is not None` and a
non-empty `finish_time`. This exists because `get_event_fights()` was originally built
only for fully-completed event pages — an unfought bout's empty method cell gets silently
defaulted to `"Decision"` by `_normalize_method()`, and winner defaults to `"fighter_a"`
with no explicit "hasn't happened" signal. Without this gate, polling mid-card would
write fabricated results into the DB for fights that haven't happened yet. Verified the
gate against the real, in-progress "UFC Fight Night: Du Plessis vs. Usman" event page
(12 fights, card not yet started): correctly identified 0 as concluded, 0 false writes.

Per-fight email is idempotent via `data/predictions/.emailed_fight_ids.txt` (fight ID
appended after a successful send, checked before re-sending on the next poll). Method
comparison uses an explicit `{"KO/TKO": "KO_TKO", ...}` label map against
`fight.method`'s already-normalized value, rather than fuzzy string matching. Handles the
edge case of a concluded fight with no stored `Prediction` row (fresh/edge-case matchup)
by emailing a "result in, no prediction on file" notice instead of crashing or skipping
silently.

**Also fixed along the way:** `run_pipeline.py::step_predict_next_event`'s get-or-create
Event logic (added earlier this session) never set the new `Event`'s `url` field — found
because `live_results_poll.py`'s first live test failed with "no URL on file" for the
actual in-flight event (id=781). Manually backfilled that row's URL via
`get_upcoming_events()` and fixed the creation code so future auto-created events aren't
missing it.

**Verified:** ran live against the real in-progress event page (see above) — safety gate
correct, 0 false positives. Per-fight email formatting/sending path is built and code-
reviewed but not yet exercised against a real concluded fight (card hadn't started at
build time) — first real test happens when tonight's prelims start concluding.

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
