"""
scripts/maybe_capture_closing.py
───────────────────────────────────
Designed to run on a frequent schedule (e.g. every 15 min via GitHub Actions
cron) and do nothing on almost every run. Checks whether "now" falls in the
closing-line capture window for the next upcoming UFC card — T-60 minutes
before its first fight, per the operational definition in
scripts/capture_closing_odds.py — and if so, captures it. Otherwise exits
quietly. Safe to run as often as you like; costs 0 extra Odds API quota on
runs where it doesn't capture (the odds fetch used to check timing IS the
same fetch used for the capture itself if the window is hit).

Why polling instead of one precisely-scheduled job: cron can't compute "60
minutes before a time that changes every event" in a single static
schedule. Running a cheap check periodically and acting only when the
condition is met is the standard way to turn a dynamic-time requirement
into something a fixed scheduler can express.

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

# Target: capture at T-60 minutes before first fight. Window is wider than
# the polling interval so a 15-min cron cadence can't skip over it entirely.
TARGET_MINUTES_BEFORE = 60
WINDOW_HALF_WIDTH_MINUTES = 12  # window = [T-72, T-48]


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
    target = first_fight_time - timedelta(minutes=TARGET_MINUTES_BEFORE)
    window_start = target - timedelta(minutes=WINDOW_HALF_WIDTH_MINUTES)
    window_end = target + timedelta(minutes=WINDOW_HALF_WIDTH_MINUTES)

    logger.info(f"First fight: {first_fight_time.isoformat()}  |  target capture window: "
                f"{window_start.isoformat()} .. {window_end.isoformat()}  |  now: {now.isoformat()}")

    if not (window_start <= now <= window_end):
        logger.info("Not in the capture window yet (or window has passed) — nothing to do")
        return

    fight_ids = [f.id for f in confirmed.values()]
    already_captured = (
        session.query(BettingOdds)
        .filter(BettingOdds.fight_id.in_(fight_ids), BettingOdds.is_closing == True)
        .count()
    )
    if already_captured > 0:
        logger.info(f"Already captured closing odds for this event ({already_captured} rows exist) — nothing to do")
        return

    logger.info("In capture window and not yet captured — capturing closing odds now")
    stored = store_odds(matched, session, is_closing=True)
    logger.success(f"Closing snapshot captured: {stored} rows stored")

    session.close()


if __name__ == "__main__":
    main()
