"""
src/ingestion/odds_scraper.py
──────────────────────────────
Pulls betting odds from The Odds API (https://the-odds-api.com).
Free tier: 500 requests/month — enough for every UFC event.

Sign up at the-odds-api.com, get a free key, put it in .env as ODDS_API_KEY.

Tracks opening and closing lines so we can detect line movement.
Stores all odds in the BettingOdds table for historical tracking.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import Optional
import requests
from loguru import logger
from rapidfuzz import process, fuzz

from config import ODDS_API_KEY, ODDS_API_BASE_URL
from src.betting.value_detector import american_to_prob, remove_vig, analyze_fight_value


SPORT_KEY = "mma_mixed_martial_arts"


# ── API Fetching ──────────────────────────────────────────────────────────────

def fetch_mma_odds(regions: str = "us", markets: str = "h2h", api_key: str = None) -> list[dict]:
    """
    Fetch current MMA moneyline odds from The Odds API.

    Args:
        api_key: Optional override. If not provided, falls back to ODDS_API_KEY
                 from config (set via .env or Streamlit secrets). Pass this when
                 users supply their own key via the dashboard UI.

    Returns raw API response list. Each item is one event with odds
    from multiple sportsbooks.

    Free tier reminder: each call costs 1 request from your monthly quota.
    Check remaining: the response headers include x-requests-remaining.
    """
    key = api_key or ODDS_API_KEY
    if not key:
        logger.warning("ODDS_API_KEY not set — skipping odds fetch")
        return []

    url = f"{ODDS_API_BASE_URL}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey":      key,
        "regions":     regions,
        "markets":     markets,
        "oddsFormat":  "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        logger.info(f"Odds fetched. API quota: {used} used, {remaining} remaining")
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            logger.error("Odds API: Invalid API key (401)")
            raise ValueError("Invalid Odds API key — check your key and try again.")
        logger.error(f"Odds API error: {e}")
        return []
    except requests.RequestException as e:
        logger.error(f"Odds API error: {e}")
        return []


def parse_odds_response(raw_events: list[dict]) -> list[dict]:
    """
    Parse the raw Odds API response into clean fight odds dicts.

    The API returns events with multiple bookmakers. We take the
    best available line (highest implied prob for each fighter) 
    or DraftKings/FanDuel as preferred books.

    Returns list of dicts:
        {
            fighter_a: str,
            fighter_b: str,
            odds_a: int,       # American odds
            odds_b: int,
            fair_prob_a: float,
            fair_prob_b: float,
            bookmaker: str,
            commence_time: datetime,
        }
    """
    fights = []
    preferred_books = {"draftkings", "fanduel", "betmgm", "caesars"}

    for event in raw_events:
        if not event.get("bookmakers"):
            continue

        commence = event.get("commence_time", "")
        try:
            fight_date = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        except Exception:
            fight_date = None

        # Pick best bookmaker — prefer major US books
        bookmakers = event["bookmakers"]
        selected_book = None
        for book in bookmakers:
            if book["key"] in preferred_books:
                selected_book = book
                break
        if not selected_book:
            selected_book = bookmakers[0]  # fallback to first available

        # Extract h2h market
        h2h = next((m for m in selected_book.get("markets", []) if m["key"] == "h2h"), None)
        if not h2h or len(h2h.get("outcomes", [])) < 2:
            continue

        outcomes = h2h["outcomes"]
        fighter_a_name = outcomes[0]["name"]
        fighter_b_name = outcomes[1]["name"]
        odds_a = int(outcomes[0]["price"])
        odds_b = int(outcomes[1]["price"])

        raw_prob_a = american_to_prob(odds_a)
        raw_prob_b = american_to_prob(odds_b)
        fair_prob_a, fair_prob_b = remove_vig(raw_prob_a, raw_prob_b)

        fights.append({
            "fighter_a":    fighter_a_name,
            "fighter_b":    fighter_b_name,
            "odds_a":       odds_a,
            "odds_b":       odds_b,
            "fair_prob_a":  round(fair_prob_a, 4),
            "fair_prob_b":  round(fair_prob_b, 4),
            "bookmaker":    selected_book["key"],
            "commence_time": fight_date,
        })

    logger.info(f"Parsed {len(fights)} fights from odds response")
    return fights


# ── Name Matching ─────────────────────────────────────────────────────────────

def match_odds_to_db_fighters(
    odds_fights: list[dict],
    db_session,
    score_cutoff: int = 70,
) -> list[dict]:
    """
    Match fighter names from the odds API to fighters in our database.

    The odds API uses different name formats than ufcstats.
    Example: "Max Holloway" vs "Maxwell Holloway"

    Returns the odds_fights list with added fighter_a_id and fighter_b_id.
    Fights where we can't match both fighters are flagged with matched=False.
    """
    from src.database import Fighter
    from src.ingestion.data_loader import normalize_name

    all_fighters = db_session.query(Fighter).all()
    name_to_id = {normalize_name(f.name): f.id for f in all_fighters}
    normalized_names = list(name_to_id.keys())

    def find_id(name: str) -> Optional[int]:
        norm = normalize_name(name)
        match = process.extractOne(norm, normalized_names, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
        if match:
            return name_to_id[match[0]]
        return None

    matched = []
    for fight in odds_fights:
        id_a = find_id(fight["fighter_a"])
        id_b = find_id(fight["fighter_b"])
        matched.append({
            **fight,
            "fighter_a_id": id_a,
            "fighter_b_id": id_b,
            "matched": id_a is not None and id_b is not None,
        })
        if not (id_a and id_b):
            logger.debug(f"Could not match: {fight['fighter_a']} vs {fight['fighter_b']}")

    matched_count = sum(1 for f in matched if f["matched"])
    logger.info(f"Matched {matched_count}/{len(matched)} fights to DB fighters")
    return matched


def store_odds(odds_fights: list[dict], db_session, is_opening: bool = False) -> int:
    """
    Store parsed odds into the BettingOdds table.

    Args:
        odds_fights:  Output from match_odds_to_db_fighters()
        db_session:   DB session
        is_opening:   True if these are opening lines

    Returns number of rows stored.
    """
    from src.database import BettingOdds, Fight
    from sqlalchemy import and_

    stored = 0
    for fight_odds in odds_fights:
        if not fight_odds.get("matched"):
            continue

        # Find the fight in our DB
        fight = (
            db_session.query(Fight)
            .filter(
                and_(
                    Fight.fighter_a_id == fight_odds["fighter_a_id"],
                    Fight.fighter_b_id == fight_odds["fighter_b_id"],
                    Fight.winner_id == None,  # upcoming fights only
                )
            )
            .first()
        )

        # Also try reversed (fighter order might differ)
        if not fight:
            fight = (
                db_session.query(Fight)
                .filter(
                    and_(
                        Fight.fighter_a_id == fight_odds["fighter_b_id"],
                        Fight.fighter_b_id == fight_odds["fighter_a_id"],
                        Fight.winner_id == None,
                    )
                )
                .first()
            )

        if not fight:
            logger.debug(f"No upcoming fight found for {fight_odds['fighter_a']} vs {fight_odds['fighter_b']}")
            continue

        odds_row = BettingOdds(
            fight_id=fight.id,
            sportsbook=fight_odds.get("bookmaker", "unknown"),
            recorded_at=datetime.utcnow(),
            is_opening=is_opening,
            is_closing=False,
            odds_fighter_a=fight_odds["odds_a"],
            odds_fighter_b=fight_odds["odds_b"],
            implied_prob_a=fight_odds["fair_prob_a"],
            implied_prob_b=fight_odds["fair_prob_b"],
        )
        db_session.add(odds_row)
        stored += 1

    db_session.commit()
    logger.info(f"Stored {stored} odds records")
    return stored


# ── Convenience: Full Fetch + Store Pipeline ──────────────────────────────────

def fetch_and_store_odds(db_session, is_opening: bool = False) -> list[dict]:
    """
    One-call function: fetch odds from API, parse, match to DB, store.
    Returns the matched odds list for immediate use in predictions.
    """
    raw = fetch_mma_odds()
    if not raw:
        return []

    parsed = parse_odds_response(raw)
    matched = match_odds_to_db_fighters(parsed, db_session)
    store_odds(matched, db_session, is_opening=is_opening)
    return [f for f in matched if f["matched"]]


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.database import init_db, get_session

    init_db()
    session = get_session()

    print("\nFetching current MMA odds...")
    odds = fetch_and_store_odds(session)

    if odds:
        print(f"\nFound {len(odds)} matched fights with odds:\n")
        for f in odds:
            print(f"  {f['fighter_a']} ({f['odds_a']:+d}) vs {f['fighter_b']} ({f['odds_b']:+d})")
            print(f"  Fair probs: {f['fair_prob_a']:.1%} / {f['fair_prob_b']:.1%}")
            print()
    else:
        print("No odds returned. Check ODDS_API_KEY in .env")

    session.close()
