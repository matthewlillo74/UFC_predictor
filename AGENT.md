# AGENT.md — Operating Instructions for AI Coding Agents

This file is for any AI agent (Claude Code or otherwise) working in this repo. Read it
before making changes.

## Required reading order

1. **This file** — environment setup, conventions, gotchas.
2. **`SESSION_LOG.md`** — chronological log of what's been changed and why. Read the top
   (most recent) entries to know what state the repo is actually in — it corrects
   anything stale in the docs below.
3. **`AGENT_HANDOFF.md`** — a point-in-time catch-up narrative written after a long
   inactive period. Useful for historical context and design rationale, but individual
   facts in it (feature counts, "already fixed" claims, metrics) can be stale — cross-check
   against `SESSION_LOG.md` and the actual code before relying on anything it states as
   current fact.
4. **`README.md`** — user-facing overview. Same caveat: verify numbers against `config.py`
   / the DB before quoting them, it drifts.

## Update `SESSION_LOG.md` after every change

Append an entry (don't rewrite history) after any non-trivial change: what changed, which
files, why, and how you verified it. This is the durable record other sessions rely on —
without it, the "already fixed" trap (see below) repeats indefinitely.

**Known failure mode this exists to prevent:** `AGENT_HANDOFF.md` claimed the duplicate-Fight-row
bug and several other things were fixed. They weren't — the fix was written and discussed but
never landed. An agent reading only the narrative doc had no way to tell intent from reality.
Always verify a claimed fix against the actual source before trusting it, and always log what
you actually did (not just what you intended) so the next session doesn't repeat this.

## Environment — read before running anything

- **Two virtualenvs exist, and they are NOT interchangeable — pick based on whether the
  task touches the network:**
  - `venv/` is a **WSL-Linux** venv, Python 3.12 (`pyvenv.cfg` shows `home = /usr/bin`).
    Cannot run from native Windows — the binaries won't execute. **WSL itself cannot reach
    ufcstats.com from this machine** (connection times out — a WSL networking/DNS issue on
    this box, separate from the anti-bot challenge below, not root-caused). But it has the
    exact package versions (`sklearn==1.4.2`, `xgboost==2.0.3`, `shap==0.51.0`) that the
    committed model pickles were originally trained against and are proven to work together.
  - `venv_win/` is a native Windows venv, **Python 3.10** (the only readily-available
    interpreter on this machine besides 3.13 — no 3.12 install exists here). Network works
    fine from here. But `shap==0.51.0` requires Python ≥3.11, so this venv **cannot exactly
    match `venv/`'s package versions** — it's currently pinned to `shap==0.49.1` (latest
    available for 3.10) alongside `sklearn==1.4.2`/`xgboost==2.0.3`, which does work, but if
    you hit another SHAP/sklearn/xgboost version error here, that's why: this venv is a
    best-effort approximation, not an exact copy.
  - **Rule of thumb: anything that scrapes (fight_scraper.py, run_pipeline.py's scrape/enrich
    steps, odds_scraper.py) → `venv_win`. Anything purely local (train_model.py,
    backtest scripts, one-off DB queries/fixes) → `venv/` via WSL**, since it's the
    better-tested environment and doesn't need network.
- **Native Windows console is cp1252** and can't print the Unicode box-drawing banner
  characters some scripts use. Set `$env:PYTHONIOENCODING = "utf-8"` (PowerShell) before
  running anything natively that prints them, or the run dies immediately with a
  `UnicodeEncodeError` on the very first `print`.
- To run the scrape/enrich pipeline natively:
  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  .\venv_win\Scripts\python.exe scripts\run_pipeline.py
  ```
- To retrain (no network needed, prefer WSL):
  ```bash
  wsl bash -c "cd /mnt/c/users/matth/UFC_predictor && source venv/bin/activate && rm -f data/processed/training_dataset.csv && python scripts/train_model.py"
  ```

## ufcstats.com bot protection

The site added an Anubis-style JS proof-of-work challenge ("Checking your browser…") at
some point. `src/ingestion/fight_scraper.py::_get()` solves it transparently (parses the
nonce/difficulty from the challenge page's inline JS, brute-forces the SHA-256 preimage,
POSTs to `/__c`, reuses the session cookie). This was a deliberate choice, discussed with
and approved by the project owner — it is not a config toggle, it's load-bearing for any
scraping to work at all. If the site changes its challenge mechanism, this will start
silently returning 0 events again (no exception raised) — check `get_all_events()` returns
a non-zero, non-suspiciously-round count before trusting a pipeline run that touched it.

## Core invariant: zero data leakage

Every feature must be computable from data strictly before the fight date. This has been
violated before in non-obvious ways (see `SESSION_LOG.md` 2026-07-15, item 9 — a backfill
script propagated career-end stats backward onto historical snapshots). Before adding any
feature or data-loading step, ask: *could this value reflect anything that happened at or
after the fight in question?* If yes, it's leakage, full stop, no matter how small the
window seems.

## Standard workflows

See `README.md` → "Workflow" section for the day-to-day commands (pre-event, post-event,
retrain, backtest). They're accurate as of the last docs pass — check `SESSION_LOG.md` for
anything more recent that hasn't been folded into the README yet.

Gotchas worth repeating here because they're easy to trip on:
- **Always `rm data/processed/training_dataset.csv` before retraining** — a stale cached
  copy silently ignores DB changes.
- **`git add -f` for `data/` and `models_saved/`** — likely gitignored by default, but this
  project commits the DB and model weights directly (Streamlit Cloud reads straight from
  the repo, no separate deploy step for data).
- **The Odds API free tier is 500 requests/month** — don't run the pipeline repeatedly in
  a day for no reason.

## Verification expectations

Before reporting a fix as done: read the actual code path, don't trust a doc's claim that
it's already fixed. Where feasible, run the affected code path (even a small isolated
repro) rather than reasoning about it purely by inspection — several bugs here (the
`UnboundLocalError`, the Anubis block, the event_id mismatch) would have been caught
immediately by actually running the code and were otherwise invisible from a read-through
alone, since failures were silently swallowed.
