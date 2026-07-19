"""
scripts/capture_closing_odds.py
─────────────────────────────────
Captures a closing-line odds snapshot for the upcoming card, tagged
is_closing=True in BettingOdds.

WHY THIS EXISTS: the project has no real historical closing-line data —
`is_closing` was never set to True anywhere before this script (confirmed
2026-07-16), and only 57 unlabeled odds snapshots exist across 8,771 fights.
Closing-Line Value (CLV) analysis is meaningless without real closes, and a
close can only be captured going forward, not reconstructed after the fact.
This script is the start of that data collection.

OPERATIONAL DEFINITION OF "CLOSING" (be consistent — this is what makes
future snapshots comparable to each other):

    Run this once, at T-60 minutes before the first fight of the card
    (i.e. 60 minutes before the prelims' scheduled start time).

Do not run this at any other time and call it a closing snapshot — an
"opening" or mid-week snapshot is a different (and already-supported) thing,
use the regular pipeline's odds fetch for that. If the actual capture time
drifts from T-60 for a given event (e.g. you ran it late), note the actual
time in your own records — don't silently treat it as directly comparable to
a precisely-timed one.

(This precision is realistic for a manual run, where a human picks the
moment. The automated path — scripts/maybe_capture_closing.py — can't get
that precise in practice, since GitHub Actions' schedule trigger is best-
effort and can drift by hours under load; it uses a deliberately wider
T-90-to-T-30 window plus a late-fallback instead of chasing an exact minute
it structurally can't hit reliably. See that script's docstring.)

Usage (run manually — this is not automated/scheduled):
    python scripts/capture_closing_odds.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from loguru import logger

from src.database import init_db, get_session
from src.ingestion.odds_scraper import fetch_and_store_odds


def main():
    init_db()
    session = get_session()

    logger.info(f"Capturing CLOSING odds snapshot at {datetime.utcnow().isoformat()} UTC")
    logger.info("Operational definition: T-60 min before first fight — "
                "make sure that's actually when you're running this.")

    matched = fetch_and_store_odds(session, is_closing=True)

    logger.success(f"Closing snapshot captured: {len(matched)} fights matched and stored.")
    if not matched:
        logger.warning("Zero fights matched — check that odds are available for "
                        "this event yet (books sometimes post lines late) and that "
                        "fighter names are resolving correctly.")

    session.close()


if __name__ == "__main__":
    main()
