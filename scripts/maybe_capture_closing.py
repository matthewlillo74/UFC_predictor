"""
scripts/maybe_capture_closing.py
───────────────────────────────────
Designed to run on a frequent schedule (e.g. every 15 min via GitHub Actions
cron) and do nothing on almost every run. Checks whether "now" falls in the
closing-line capture window for the next upcoming UFC card and, if so,
captures it. Otherwise exits quietly. Safe to run as often as you like;
costs 0 extra Odds API quota on runs where it doesn't capture (the odds
fetch used to check timing IS the same fetch used for the capture itself if
the window is hit).

WINDOW — WIDENED 2026-07-19 after the first live run of this automation
missed its capture entirely: the original design used a precise T-60±12min
(24-minute) target window on the theory that a 15-min cron would reliably
land inside it. In production, GitHub Actions' `schedule` trigger turned
out to be far less reliable than that — real gaps between runs that
weekend were 1-3 hours, not 15 minutes (documented GitHub behavior: the
schedule trigger is best-effort and can be delayed under load, worse at
round-number times and worse with multiple frequent-cron workflows
competing in the same repo). No amount of cron-offset tuning fixes a
scheduler that's inherently best-effort — the fix is redundancy, not
precision:

  - WINDOW_START_MINUTES_BEFORE = 90, WINDOW_DEADLINE_MINUTES_BEFORE = 30:
    a full 60-minute window (not 24) to capture in under normal conditions.
  - Past the deadline (T-30) with still no capture, this treats it as a
    fallback: capture immediately regardless, right up until first fight
    time, rather than silently missing the card because the "clean" window
    was never hit. Logged distinctly (see `late` below) so it's obvious
    from the logs which case fired — a late-fallback capture is a slightly
    worse approximation of "closing" than an in-window one, worth knowing
    if you're ever debugging an odd CLV number.
  - This script is also invoked a second time, independently, from
    daily_pipeline.yml's closing-odds-fallback job at fixed times later on
    event day — a structurally separate trigger (different workflow file,
    different cron registration) so a scheduling failure in
    closing_odds_poll.yml specifically doesn't silently take out capture
    for the whole event. Both call sites run this exact same function, so
    "already captured" idempotency (below) means calling it twice is free.

First-fight time comes from The Odds API's commence_time field (already
parsed by odds_scraper.py) — not from Event.date in our own DB, which only
stores a calendar date with no time-of-day. Only counts fights that match
an actual upcoming (unresolved) Fight row in our DB, so noise from other
MMA promotions in the same API response doesn't skew the window.

Idempotent: won't double-capture if it happens to fire more than once
inside the window (checks for an existing is_closing=True row first).

Usage (intended for scheduled/unattended use, but safe to run manually):
    python scripts/maybe_capture_closing.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from loguru import logger

from src.database import init_db, get_session, Fight, BettingOdds
from src.ingestion.odds_scraper import fetch_mma_odds, parse_odds_response, match_odds_to_db_fighters, store_odds

# See module docstring for why this is a wide window with a deadline
# fallback rather than a precise T-60 target.
WINDOW_START_MINUTES_BEFORE = 90     # any earlier isn't really a "closing" line
WINDOW_DEADLINE_MINUTES_BEFORE = 30  # past here with no capture yet, fall back to capturing immediately


def _confirmed_upcoming_fight_ids(session, matched_odds: list[dict]) -> dict:
    """Cross-reference name-matched odds against real upcoming Fight rows,
    filtering out noise from other MMA promotions in the same API response.
    Returns {fight_odds_index: Fight} for confirmed matches."""
    confirmed = {}
    for i, fo in enumerate(matched_odds):
        if not fo.get("matched"):
            continue
        fa_id, fb_id = fo["fighter_a_id"], fo["fighter_b_id"]
        fight = (
            session.query(Fight)
            .filter(
                Fight.winner_id.is_(None),
                ((Fight.fighter_a_id == fa_id) & (Fight.fighter_b_id == fb_id))
                | ((Fight.fighter_a_id == fb_id) & (Fight.fighter_b_id == fa_id)),
            )
            .first()
        )
        if fight:
            confirmed[i] = fight
    return confirmed


def main():
    init_db()
    session = get_session()

    raw = fetch_mma_odds()
    if not raw:
        logger.info("No odds data available (API down, quota exhausted, or no upcoming events) — nothing to do")
        return

    parsed = parse_odds_response(raw)
    matched = match_odds_to_db_fighters(parsed, session)
    confirmed = _confirmed_upcoming_fight_ids(session, matched)

    if not confirmed:
        logger.info("No matched odds correspond to a confirmed upcoming fight in our DB — nothing to do")
        return

    commence_times = [matched[i]["commence_time"] for i in confirmed if matched[i]["commence_time"]]
    if not commence_times:
        logger.warning("Confirmed upcoming fights found but none have a parseable commence_time — nothing to do")
        return

    first_fight_time = min(commence_times)
    if first_fight_time.tzinfo is None:
        first_fight_time = first_fight_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    window_start = first_fight_time - timedelta(minutes=WINDOW_START_MINUTES_BEFORE)
    deadline = first_fight_time - timedelta(minutes=WINDOW_DEADLINE_MINUTES_BEFORE)

    logger.info(f"First fight: {first_fight_time.isoformat()}  |  capture window: "
                f"{window_start.isoformat()} .. {first_fight_time.isoformat()}  |  "
                f"deadline for a clean in-window capture: {deadline.isoformat()}  |  now: {now.isoformat()}")

    if now < window_start:
        logger.info("Too early for the capture window yet — nothing to do")
        return
    if now > first_fight_time:
        logger.info("First fight has already started — capture window has fully passed, nothing to do")
        return

    late = now > deadline

    fight_ids = [f.id for f in confirmed.values()]
    already_captured = (
        session.query(BettingOdds)
        .filter(BettingOdds.fight_id.in_(fight_ids), BettingOdds.is_closing == True)
        .count()
    )
    if already_captured > 0:
        logger.info(f"Already captured closing odds for this event ({already_captured} rows exist) — nothing to do")
        return

    if late:
        logger.warning("Past the T-30 deadline with no capture yet — capturing now as a late fallback "
                        "(a slightly worse approximation of 'closing' than an in-window capture, but far "
                        "better than missing the card entirely)")
    else:
        logger.info("In capture window and not yet captured — capturing closing odds now")

    stored = store_odds(matched, session, is_closing=True)
    logger.success(f"Closing snapshot captured: {stored} rows stored" + (" (late fallback)" if late else ""))

    session.close()


if __name__ == "__main__":
    main()
