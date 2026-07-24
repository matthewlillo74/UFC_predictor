"""
scripts/is_ufc_weekend.py
───────────────────────────
Cheap gate used by the weekend-only polling workflows (closing_odds_poll.yml,
live_results_poll.yml, daily_pipeline.yml's closing-odds-fallback job) to
skip real work on Saturdays/Sundays that don't actually have a UFC card.
UFC doesn't run every week — without this, those workflows poll all weekend
regardless, burning Odds API quota and hitting ufcstats.com for nothing on
an off week.

Pure DB read, no network calls — safe to run as the first, cheapest step of
a job. Relies on Event.date being the REAL scraped fight date, which as of
2026-07-23 it reliably is (see SESSION_LOG.md for the get_upcoming_events()
date-parsing bug this depended on fixing first — before that fix, Event.date
for freshly-auto-created events was silently the row's creation timestamp,
not the actual fight date, which would have made this gate wrong).

Exit code 0 = there's a UFC event this weekend, downstream steps should run.
Exit code 1 = no event this weekend, safe to skip everything else.

Usage (in a workflow, used for its exit code — see the three workflows
above for the `if python scripts/is_ufc_weekend.py; then ... fi` pattern):
    python scripts/is_ufc_weekend.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from loguru import logger

from src.database import init_db, get_session, Event, Fight


def is_ufc_weekend() -> bool:
    init_db()
    session = get_session()

    now = datetime.utcnow()
    today = datetime(now.year, now.month, now.day)
    # weekday(): Mon=0 .. Sat=5, Sun=6. On Sunday, "this weekend" means
    # YESTERDAY's Saturday (a card that started Saturday can still be
    # resolving into Sunday UTC) — walking forward to the *next* Saturday
    # here would skip right past the card that's actually in progress. On
    # any other day (including Saturday itself), walk forward to the next
    # Saturday, so this also works standalone for "is there a game this
    # upcoming weekend" when run on, say, a Thursday.
    if now.weekday() == 6:
        this_saturday = today - timedelta(days=1)
    else:
        this_saturday = today + timedelta(days=(5 - now.weekday()) % 7)
    window_end = this_saturday + timedelta(days=2)  # covers all of Saturday + Sunday

    event = (
        session.query(Event)
        .join(Fight, Fight.event_id == Event.id)
        .filter(Event.date >= this_saturday, Event.date < window_end)
        .first()
    )
    session.close()

    if event:
        logger.info(f"UFC event this weekend: {event.name} ({event.date.date()})")
        return True

    logger.info(f"No UFC event found for the weekend of {this_saturday.date()} — off week, skip polling")
    return False


if __name__ == "__main__":
    sys.exit(0 if is_ufc_weekend() else 1)
