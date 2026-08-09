"""
scripts/check_data_integrity.py
──────────────────────────────────
Periodic health check for the automated pipeline. Doesn't fix anything —
fixing duplicate/stuck fight rows needs verification against the live
ufcstats.com event page first (see SESSION_LOG.md 2026-08-09 for why:
blindly deleting/merging without checking live data risks discarding the
wrong row). This only detects and reports loudly, so a human notices
instead of the pipeline reporting "success" every day while the data
underneath is quietly wrong — which is exactly what happened across three
separate events before this check existed: 25 Fight rows for a 12-fight
card, and one event stuck at 2/16 resolved for over a week, both while
every workflow run showed a green checkmark.

Checks:
  1. Duplicate fight groups — more than one Fight row for the same
     event_id + fighter pair. Should never happen given the event_id +
     fighter-pair matching in data_loader._load_fight(), but that's
     exactly the invariant that broke silently once already (a stale
     Fight.fight_date-based match, since fixed).
  2. Past events stuck unresolved — any event more than
     STUCK_THRESHOLD_DAYS old that still has an unresolved fight. Normal
     processing takes at most a day or two; anything stuck longer means
     the pipeline isn't actually finishing that event (root cause last
     time: a cursor bug that abandoned an event once a later one got any
     result in — also since fixed, but this check exists so a *different*
     future bug doesn't cause the same silent failure).

Exit code 0 = clean. Exit code 1 = found something — fails the job loudly
in GitHub Actions and triggers the default failure-notification email,
the same mechanism daily_pipeline.yml's other steps already rely on.

Usage:
    python scripts/check_data_integrity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datetime import datetime, timedelta

from loguru import logger

from src.database import init_db, get_session, Event, Fight

STUCK_THRESHOLD_DAYS = 5


def check_duplicate_fights(session) -> list[str]:
    problems = []
    events = session.query(Event).filter(Event.date.isnot(None)).all()
    for ev in events:
        fights = session.query(Fight).filter_by(event_id=ev.id).all()
        groups = defaultdict(list)
        for f in fights:
            key = tuple(sorted([f.fighter_a_id, f.fighter_b_id]))
            groups[key].append(f)
        for key, rows in groups.items():
            if len(rows) > 1:
                ids = ", ".join(str(r.id) for r in rows)
                problems.append(
                    f"Event {ev.id} '{ev.name}': duplicate fight rows for "
                    f"fighter pair {key} (fight_ids: {ids})"
                )
    return problems


def check_stuck_events(session) -> list[str]:
    problems = []
    cutoff = datetime.utcnow() - timedelta(days=STUCK_THRESHOLD_DAYS)
    events = session.query(Event).filter(Event.date.isnot(None), Event.date < cutoff).all()
    for ev in events:
        fights = session.query(Fight).filter_by(event_id=ev.id).all()
        if not fights:
            continue
        unresolved = [f for f in fights if f.winner_id is None and f.finish_round is None]
        if unresolved:
            days_stuck = (datetime.utcnow() - ev.date).days
            problems.append(
                f"Event {ev.id} '{ev.name}' ({ev.date.date()}, {days_stuck}d ago): "
                f"{len(unresolved)}/{len(fights)} fights still unresolved"
            )
    return problems


def main() -> bool:
    init_db()
    session = get_session()

    problems = check_duplicate_fights(session) + check_stuck_events(session)
    session.close()

    if not problems:
        logger.success("Data integrity check passed — no duplicate fights, no events stuck unresolved")
        return True

    logger.error(f"Data integrity check found {len(problems)} issue(s):")
    for p in problems:
        logger.error(f"  {p}")
    logger.error(
        "Fixing these needs verification against the live ufcstats.com event page first "
        "(see SESSION_LOG.md 2026-08-09) — don't blindly delete/merge without checking."
    )
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
