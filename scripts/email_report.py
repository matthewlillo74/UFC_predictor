"""
scripts/email_report.py
──────────────────────────
Emails a summary of the most recently scored event's results via Gmail SMTP.

Reuses log_live_results.py's print_report() for all the actual analysis
(winner/method/round accuracy, P&L, misses) — this script just captures that
output and mails it, scoped to the single most recent event
(last_n_events=1). Only sends once per event: tracks the last-emailed event
name in a small state file so re-running this (e.g. the daily pipeline
running on non-event days) doesn't re-send the same report.

SETUP (one-time, can't be done from code):
1. Enable 2-Step Verification on your Gmail account (required for App
   Passwords) — myaccount.google.com/security.
2. myaccount.google.com/apppasswords → generate one for "Mail". This is a
   16-character password used ONLY for this SMTP connection, separate from
   your real Google password — safe to put in a GitHub secret. Full OAuth
   Gmail API is overkill for one-way "send me a report" use; SMTP + App
   Password needs no Google Cloud project, no token refresh.
3. Add GitHub Actions secrets (Settings → Secrets and variables → Actions):
     GMAIL_ADDRESS       — the Gmail account sending the report
     GMAIL_APP_PASSWORD  — the 16-character App Password from step 2
     REPORT_EMAIL_TO     — where to send it (can be the same address)
   For local testing, put the same three in .env instead.

Usage:
    python scripts/email_report.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import smtplib
import importlib.util
from contextlib import redirect_stdout
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger

from src.database import init_db, get_session
from config import PREDICTIONS_DIR

STATE_PATH = PREDICTIONS_DIR / ".last_emailed_event.txt"


def _load_log_live_results_module():
    # Same dynamic-import pattern already used in run_pipeline.py's
    # step_auto_score_live_results — avoids a circular/awkward package import
    # for a sibling script.
    script_path = os.path.join(os.path.dirname(__file__), "log_live_results.py")
    spec = importlib.util.spec_from_file_location("log_live_results", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _send_email(subject: str, body: str):
    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    to_address = os.getenv("REPORT_EMAIL_TO", gmail_address)

    if not gmail_address or not gmail_app_password:
        logger.warning("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — skipping email, "
                        "printing report instead:\n" + body)
        return

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to_address

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.send_message(msg)

    logger.success(f"Report emailed to {to_address}")


def main():
    init_db()
    session = get_session()
    llr = _load_log_live_results_module()

    log = llr.get_or_create_log()
    if not log:
        logger.info("No live results logged yet — nothing to email")
        return

    events_in_order = []
    for row in log:
        name = (row.get("event") or "").strip()
        if name and name not in events_in_order:
            events_in_order.append(name)
    if not events_in_order:
        logger.info("No events found in the live log — nothing to email")
        return

    latest_event = events_in_order[-1]

    already_emailed = STATE_PATH.read_text().strip() if STATE_PATH.exists() else ""
    if already_emailed == latest_event:
        logger.info(f"Already emailed a report for '{latest_event}' — nothing new")
        return

    buf = io.StringIO()
    with redirect_stdout(buf):
        llr.print_report(session, last_n_events=1)
    body = buf.getvalue()

    if not body.strip():
        logger.warning(f"print_report produced no output for '{latest_event}' — skipping email")
        return

    _send_email(subject=f"UFC Predictor results: {latest_event}", body=body)

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(latest_event)

    session.close()


if __name__ == "__main__":
    main()
