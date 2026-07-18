"""
scripts/live_results_poll.py
───────────────────────────────
Designed to run frequently (e.g. every 10 min via GitHub Actions cron)
during a live UFC card. Checks the current event's fight-by-fight results
as they land on ufcstats.com, and for each fight that just concluded (since
the last check), updates the DB and emails a prediction-vs-actual
comparison for THAT fight — not a once-per-event summary.

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

Emails one message per newly-concluded fight (not per poll — idempotent,
tracked in data/predictions/.emailed_fight_ids.txt). If a fight has no
stored Prediction (shouldn't normally happen, but a fresh/edge-case
matchup could lack one), still emails a "result in, no prediction on
file" notice rather than silently skipping it.

Usage:
    python scripts/live_results_poll.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smtplib
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger

from src.database import init_db, get_session, Event, Fight, Fighter, Prediction
from src.ingestion.fight_scraper import get_event_fights
from src.ingestion.data_loader import get_or_create_fighter
from config import PREDICTIONS_DIR

EMAILED_FIGHTS_PATH = PREDICTIONS_DIR / ".emailed_fight_ids.txt"


def _load_emailed_ids() -> set:
    if not EMAILED_FIGHTS_PATH.exists():
        return set()
    return {line.strip() for line in EMAILED_FIGHTS_PATH.read_text().splitlines() if line.strip()}


def _mark_emailed(fight_id: int):
    EMAILED_FIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMAILED_FIGHTS_PATH, "a") as f:
        f.write(f"{fight_id}\n")


def _is_genuinely_concluded(fight_data: dict) -> bool:
    """The one check this whole script depends on — see module docstring."""
    return fight_data.get("finish_round") is not None and bool((fight_data.get("finish_time") or "").strip())


def _find_matching_fight(session, event: Event, fighter_a: Fighter, fighter_b: Fighter):
    return (
        session.query(Fight)
        .filter(
            Fight.event_id == event.id,
            Fight.winner_id.is_(None),
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

    logger.success(f"Per-fight result emailed: {subject}")


def _format_fight_email(fight: Fight, fa: Fighter, fb: Fighter, pred: Prediction | None) -> tuple[str, str]:
    winner = fa if fight.winner_id == fa.id else fb if fight.winner_id == fb.id else None
    winner_name = winner.name if winner else "Draw/No Contest"
    loser_name = (fb.name if winner is fa else fa.name) if winner else None

    result_line = (
        f"{winner_name} def. {loser_name} by {fight.method}, "
        f"R{fight.finish_round} {fight.finish_time}"
        if winner else f"{fa.name} vs {fb.name} — {fight.method}"
    )

    if pred is None:
        subject = f"UFC Result: {fa.name} vs {fb.name} (no prediction on file)"
        body = (
            f"{result_line}\n\n"
            f"No stored prediction was found for this fight — can't compare. "
            f"This shouldn't normally happen; worth checking why if it keeps occurring."
        )
        return subject, body

    predicted_winner = _fighter_name_by_id(pred.predicted_winner_id, fa, fb)
    winner_correct = (winner is not None) and (pred.predicted_winner_id == fight.winner_id)
    confidence = max(pred.prob_fighter_a, pred.prob_fighter_b)

    prob_ko = pred.prob_ko_tko or 0.0
    prob_sub = pred.prob_submission or 0.0
    prob_dec = pred.prob_decision or 0.0
    method_probs = {"KO/TKO": prob_ko, "Submission": prob_sub, "Decision": prob_dec}
    predicted_method = max(method_probs, key=lambda k: method_probs[k])
    # fight.method is already normalized to KO_TKO/Submission/Decision/NC/Draw
    # (fight_scraper._normalize_method) — map our own label the same way rather
    # than fuzzy-matching strings, which was fragile.
    method_label_to_normalized = {"KO/TKO": "KO_TKO", "Submission": "Submission", "Decision": "Decision"}
    method_correct = fight.method == method_label_to_normalized.get(predicted_method)

    status = "CORRECT" if winner_correct else "WRONG"
    subject = f"UFC Result: {fa.name} vs {fb.name} — model was {status}"

    body_lines = [
        result_line,
        "",
        f"Our prediction: {predicted_winner} {confidence:.1%}"
        + (" (favored)" if confidence > 0.5 else ""),
        f"Actual winner: {winner_name}  {'✅ CORRECT' if winner_correct else '❌ WRONG' if winner else '(no winner — draw/NC)'}",
        "",
        f"Method predicted: KO/TKO {prob_ko:.0%} | Submission {prob_sub:.0%} | "
        f"Decision {prob_dec:.0%}  (favored: {predicted_method})",
        f"Actual method: {fight.method}  {'✅ CORRECT' if method_correct else '❌ WRONG'}",
    ]
    return subject, "\n".join(body_lines)


def _fighter_name_by_id(fighter_id, fa: Fighter, fb: Fighter) -> str:
    if fighter_id == fa.id:
        return fa.name
    if fighter_id == fb.id:
        return fb.name
    return "Unknown"


def main():
    init_db()
    session = get_session()

    event = (
        session.query(Event)
        .join(Fight, Fight.event_id == Event.id)
        .filter(Fight.winner_id.is_(None))
        .order_by(Event.date.desc())
        .first()
    )
    if not event:
        logger.info("No event with unresolved fights — nothing to poll")
        return

    if not event.url:
        logger.warning(f"Event '{event.name}' has no URL on file — can't scrape live results")
        return

    logger.info(f"Polling live results for: {event.name}")
    scraped_fights = get_event_fights(event.url)
    if not scraped_fights:
        logger.info("No fights returned from scrape — nothing to do")
        return

    emailed_ids = _load_emailed_ids()
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

        if str(fight.id) in emailed_ids:
            continue  # already emailed this fight (shouldn't hit given the check above, defense in depth)

        pred = session.query(Prediction).filter_by(fight_id=fight.id).first()
        subject, body = _format_fight_email(fight, fa, fb, pred)
        _send_email(subject, body)
        _mark_emailed(fight.id)

    session.commit()
    session.close()
    logger.info(f"Done. {newly_resolved} newly-resolved fight(s) this run.")


if __name__ == "__main__":
    main()
