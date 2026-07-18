"""
scripts/live_results_poll.py
───────────────────────────────
Designed to run frequently (e.g. every 5 min via GitHub Actions cron)
during a live UFC card. Checks the current event's fight-by-fight results
as they land on ufcstats.com and updates the DB as each fight concludes.
Once every fight on the card has concluded, emails ONE summary — every
pick vs. actual outcome for the whole card — rather than a message per
fight.

SAFETY — READ BEFORE MODIFYING: fight_scraper.get_event_fights() was built
for fully-completed events, and has no explicit "not yet fought" state for
a row. If a fight hasn't happened yet, its method cell is empty, which
_normalize_method() silently defaults to "Decision", and winner defaults
to "fighter_a" (ufcstats always lists the winner first when there IS a
result, but this scraper has no way to know there ISN'T one from the
method field alone). Trusting the winner blindly during a live, partially-
complete card would write FALSE results into the DB for fights that
haven't happened yet.

The reliable signal this script actually uses: finish_time and
finish_round are only ever populated for genuinely concluded fights
(int()/strip() on an empty cell cleanly produces None / "" with no
fallback, unlike method). A fight_data row is only trusted as a real
result if BOTH finish_round is not None AND finish_time is non-empty.
Do not weaken this check without re-verifying against a real in-progress
event page.

"Fully resolved" for a card is judged off finish_round (set for every
concluded fight, including draws/NCs), not winner_id (which stays NULL
for a draw/NC even after the fight is over — using winner_id here would
make the script think a card with a draw on it never finishes).

Summary email is idempotent via data/predictions/.emailed_event_ids.txt
(event ID appended after a successful send). If this run resolves the
card's last fight but crashes/loses network before the email goes out,
the next run's primary query (which only looks for events with an
unresolved fight) would find nothing — so main() falls back to checking
whether the single most recent event is fully resolved and not yet
emailed, and sends the summary then instead of losing it silently.

Usage:
    python scripts/live_results_poll.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smtplib
from email.mime.text import MIMEText

from loguru import logger

from src.database import init_db, get_session, Event, Fight, Fighter, Prediction
from src.ingestion.fight_scraper import get_event_fights
from src.ingestion.data_loader import get_or_create_fighter
from config import PREDICTIONS_DIR

EMAILED_EVENTS_PATH = PREDICTIONS_DIR / ".emailed_event_ids.txt"


def _load_emailed_event_ids() -> set:
    if not EMAILED_EVENTS_PATH.exists():
        return set()
    return {line.strip() for line in EMAILED_EVENTS_PATH.read_text().splitlines() if line.strip()}


def _mark_event_emailed(event_id: int):
    EMAILED_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMAILED_EVENTS_PATH, "a") as f:
        f.write(f"{event_id}\n")


def _is_genuinely_concluded(fight_data: dict) -> bool:
    """The one check this whole script depends on — see module docstring."""
    return fight_data.get("finish_round") is not None and bool((fight_data.get("finish_time") or "").strip())


def _event_fully_resolved(session, event: Event) -> bool:
    total = session.query(Fight).filter(Fight.event_id == event.id).count()
    unresolved = session.query(Fight).filter(Fight.event_id == event.id, Fight.finish_round.is_(None)).count()
    return total > 0 and unresolved == 0


def _find_matching_fight(session, event: Event, fighter_a: Fighter, fighter_b: Fighter):
    return (
        session.query(Fight)
        .filter(
            Fight.event_id == event.id,
            Fight.finish_round.is_(None),
            ((Fight.fighter_a_id == fighter_a.id) & (Fight.fighter_b_id == fighter_b.id))
            | ((Fight.fighter_a_id == fighter_b.id) & (Fight.fighter_b_id == fighter_a.id)),
        )
        .first()
    )


def _send_email(subject: str, body: str):
    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    to_address = os.getenv("REPORT_EMAIL_TO", gmail_address)

    if not gmail_address or not gmail_app_password:
        logger.warning("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — skipping email, "
                        "printing instead:\n" + body)
        return

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to_address

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.send_message(msg)

    logger.success(f"Card summary emailed: {subject}")


def _fighter_name_by_id(fighter_id, fa: Fighter, fb: Fighter) -> str:
    if fighter_id == fa.id:
        return fa.name
    if fighter_id == fb.id:
        return fb.name
    return "Unknown"


# fight.method is already normalized to KO_TKO/Submission/Decision/NC/Draw
# (fight_scraper._normalize_method) — map our own label the same way rather
# than fuzzy-matching strings, which was fragile.
_METHOD_LABEL_TO_NORMALIZED = {"KO/TKO": "KO_TKO", "Submission": "Submission", "Decision": "Decision"}


def _format_fight_line(session, fight: Fight) -> tuple[str, bool | None]:
    """Returns (one-line summary, winner_correct or None if no prediction/no winner)."""
    fa = session.query(Fighter).get(fight.fighter_a_id)
    fb = session.query(Fighter).get(fight.fighter_b_id)
    winner = fa if fight.winner_id == fa.id else fb if fight.winner_id == fb.id else None
    winner_name = winner.name if winner else "Draw/No Contest"

    pred = session.query(Prediction).filter_by(fight_id=fight.id).first()
    if pred is None:
        return f"{fa.name} vs {fb.name}: {winner_name} by {fight.method} — no prediction on file", None

    predicted_winner = _fighter_name_by_id(pred.predicted_winner_id, fa, fb)
    confidence = max(pred.prob_fighter_a, pred.prob_fighter_b)
    winner_correct = (winner is not None) and (pred.predicted_winner_id == fight.winner_id)

    prob_ko = pred.prob_ko_tko or 0.0
    prob_sub = pred.prob_submission or 0.0
    prob_dec = pred.prob_decision or 0.0
    method_probs = {"KO/TKO": prob_ko, "Submission": prob_sub, "Decision": prob_dec}
    predicted_method = max(method_probs, key=lambda k: method_probs[k])
    method_correct = fight.method == _METHOD_LABEL_TO_NORMALIZED.get(predicted_method)

    status = "CORRECT" if winner_correct else "WRONG" if winner else "N/A"
    tag = "[MAIN EVENT] " if fight.is_main_event else ""
    line = (
        f"{tag}{fa.name} vs {fb.name}\n"
        f"  Picked: {predicted_winner} ({confidence:.0%}, {predicted_method}) — {status}\n"
        f"  Actual: {winner_name} by {fight.method}, R{fight.finish_round} {fight.finish_time}"
        f"  (method {'correct' if method_correct else 'wrong'})"
    )
    return line, winner_correct


def _format_event_summary_email(session, event: Event) -> tuple[str, str]:
    fights = session.query(Fight).filter(Fight.event_id == event.id).order_by(Fight.id.asc()).all()

    lines, correct, graded = [], 0, 0
    for fight in fights:
        line, winner_correct = _format_fight_line(session, fight)
        lines.append(line)
        if winner_correct is not None:
            graded += 1
            correct += int(winner_correct)

    record = f"{correct}/{graded} correct" if graded else "no graded predictions"
    subject = f"UFC Results: {event.name} — {record}"
    body = f"{event.name}\n{record}\n\n" + "\n\n".join(lines)
    return subject, body


def main():
    init_db()
    session = get_session()
    emailed_events = _load_emailed_event_ids()

    event = (
        session.query(Event)
        .join(Fight, Fight.event_id == Event.id)
        .filter(Fight.finish_round.is_(None))
        .order_by(Event.date.desc())
        .first()
    )

    if event is None:
        # Nothing currently in progress. Cover the case where the previous run
        # resolved the card's last fight but crashed/lost network before the
        # summary email went out — check the most recent event overall.
        event = session.query(Event).order_by(Event.date.desc()).first()
        if event is None or str(event.id) in emailed_events or not _event_fully_resolved(session, event):
            logger.info("No in-progress card and nothing pending to email — nothing to do")
            session.close()
            return
        logger.info(f"'{event.name}' was already fully resolved from a prior run — sending summary now")
        subject, body = _format_event_summary_email(session, event)
        _send_email(subject, body)
        _mark_event_emailed(event.id)
        session.close()
        return

    if not event.url:
        logger.warning(f"Event '{event.name}' has no URL on file — can't scrape live results")
        session.close()
        return

    logger.info(f"Polling live results for: {event.name}")
    scraped_fights = get_event_fights(event.url)
    if not scraped_fights:
        logger.info("No fights returned from scrape — nothing to do")
        session.close()
        return

    newly_resolved = 0
    for fd in scraped_fights:
        if not _is_genuinely_concluded(fd):
            continue

        fa = get_or_create_fighter(session, fd["fighter_a_name"], fd.get("fighter_a_url", ""))
        fb = get_or_create_fighter(session, fd["fighter_b_name"], fd.get("fighter_b_url", ""))
        if fa is None or fb is None:
            continue

        fight = _find_matching_fight(session, event, fa, fb)
        if fight is None:
            continue  # already resolved earlier, or not on this card

        if fd["winner"] == "fighter_a":
            fight.winner_id = fa.id
        elif fd["winner"] == "fighter_b":
            fight.winner_id = fb.id
        # else "draw" — leave winner_id as None but still record method/round/time below

        fight.method = fd["method"]
        fight.finish_round = fd["finish_round"]
        fight.finish_time = fd["finish_time"]
        session.flush()
        newly_resolved += 1

    session.commit()
    logger.info(f"{newly_resolved} newly-resolved fight(s) this run.")

    if str(event.id) not in emailed_events and _event_fully_resolved(session, event):
        logger.info(f"'{event.name}' is fully resolved — sending card summary")
        subject, body = _format_event_summary_email(session, event)
        _send_email(subject, body)
        _mark_event_emailed(event.id)

    session.close()


if __name__ == "__main__":
    main()
