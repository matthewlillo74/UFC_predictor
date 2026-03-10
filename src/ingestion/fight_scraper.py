"""
src/ingestion/fight_scraper.py
──────────────────────────────
Scrapes ufcstats.com for:
  - All historical UFC events
  - Fight results per event
  - Fighter profiles and stats

ufcstats.com is the official UFC data source. It has clean HTML,
no JS rendering required, and covers the full UFC history.

Usage:
    scraper = UFCStatsScraper()

    # Get all events
    events = scraper.get_all_events()

    # Get fights for a specific event
    fights = scraper.get_event_fights(event_url)

    # Get fighter profile + stats
    fighter = scraper.get_fighter(fighter_url)
"""

import time
import re
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ufcstats.com doesn't need fake UA but we respect rate limits
BASE_URL = "http://www.ufcstats.com"
DELAY = 1.5  # seconds between requests — be polite


def _get(url: str) -> Optional[BeautifulSoup]:
    """Fetch a page and return parsed BeautifulSoup, or None on failure."""
    try:
        time.sleep(DELAY)
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        logger.error(f"Request failed: {url} — {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def get_all_events() -> list[dict]:
    """
    Scrape the full UFC event list from ufcstats.com.

    Returns list of dicts:
        [{"name": "UFC 300", "date": datetime, "url": "http://..."}]
    """
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    soup = _get(url)
    if not soup:
        return []

    events = []
    rows = soup.select("tr.b-statistics__table-row")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 1:
            continue

        link = cells[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        event_url = link.get("href", "").strip()

        # ufcstats puts name and date in cell[0] separated by whitespace/span
        # e.g. "UFC 6: Clash of the Titans  July 14, 1995"
        # Extract by getting full cell text and removing the link text
        cell0_full = cells[0].get_text(separator=" ", strip=True)
        date_str = cell0_full.replace(name, "").strip()

        # Fallback: try cells[1]
        if not date_str and len(cells) > 1:
            date_str = cells[1].get_text(strip=True)

        date = None
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                date = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue

        if name and event_url:
            events.append({
                "name": name,
                "date": date,
                "url": event_url,
            })

    logger.info(f"Found {len(events)} events")
    return events


# ══════════════════════════════════════════════════════════════════════════════
# FIGHTS PER EVENT
# ══════════════════════════════════════════════════════════════════════════════

def get_event_fights(event_url: str) -> list[dict]:
    """
    Scrape all fights from a specific UFC event page.

    Returns list of fight dicts, each containing:
        fighter_a, fighter_b, fighter_a_url, fighter_b_url,
        winner, method, round, time, weight_class, is_title_fight
    """
    soup = _get(event_url)
    if not soup:
        return []

    fights = []
    rows = soup.select("tr.b-fight-details__table-row")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        # Fighter names and URLs
        fighter_links = cells[1].find_all("a") if len(cells) > 1 else []
        if len(fighter_links) < 2:
            continue

        fighter_a_name = fighter_links[0].get_text(strip=True)
        fighter_b_name = fighter_links[1].get_text(strip=True)
        fighter_a_url = fighter_links[0].get("href", "").strip()
        fighter_b_url = fighter_links[1].get("href", "").strip()

        method_cell = cells[7].get_text(strip=True) if len(cells) > 7 else ""
        round_cell = cells[8].get_text(strip=True) if len(cells) > 8 else ""
        time_cell = cells[9].get_text(strip=True) if len(cells) > 9 else ""
        weight_class = cells[6].get_text(strip=True) if len(cells) > 6 else ""

        is_title = "title" in weight_class.lower()

        # ufcstats always lists winner first
        winner = "fighter_a"
        if method_cell.upper() in ("DRAW", "NC", "NO CONTEST"):
            winner = "draw"

        fight_link = row.find("a")
        fight_url = fight_link.get("href", "").strip() if fight_link else ""

        fights.append({
            "fighter_a_name": fighter_a_name,
            "fighter_b_name": fighter_b_name,
            "fighter_a_url": fighter_a_url,
            "fighter_b_url": fighter_b_url,
            "winner": winner,
            "method": _normalize_method(method_cell),
            "finish_round": _safe_int(round_cell),
            "finish_time": time_cell,
            "weight_class": weight_class.replace("UFC ", "").strip(),
            "is_title_fight": is_title,
            "fight_url": fight_url,
        })

    logger.debug(f"Found {len(fights)} fights at {event_url}")
    return fights


# ══════════════════════════════════════════════════════════════════════════════
# FIGHTER PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def get_fighter(fighter_url: str) -> Optional[dict]:
    """
    Scrape a fighter's profile page on ufcstats.com.

    Returns dict with physical attributes and career stats.
    """
    soup = _get(fighter_url)
    if not soup:
        return None

    data = {"url": fighter_url}

    # Name
    name_el = soup.find("span", class_="b-content__title-highlight")
    data["name"] = name_el.get_text(strip=True) if name_el else ""

    # Physical + career stats (label: value pairs)
    stat_items = soup.select("li.b-list__box-list-item")
    for item in stat_items:
        text = item.get_text(separator="|", strip=True)
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        label = parts[0].lower().rstrip(":")
        value = parts[1]

        if "height" in label:
            data["height_cm"] = _height_to_cm(value)
        elif "reach" in label:
            data["reach_cm"] = _reach_to_cm(value)
        elif "stance" in label:
            data["stance"] = value
        elif "dob" in label or "date of birth" in label:
            data["date_of_birth"] = _parse_date(value)
        elif "slpm" in label:
            data["slpm"] = _safe_float(value)
        elif "str. acc" in label:
            data["strike_accuracy"] = _safe_float(value.replace("%", "")) / 100 if value else None
        elif "sapm" in label:
            data["sapm"] = _safe_float(value)
        elif "str. def" in label:
            data["strike_defense"] = _safe_float(value.replace("%", "")) / 100 if value else None
        elif "td avg" in label:
            data["td_avg"] = _safe_float(value)
        elif "td acc" in label:
            data["td_accuracy"] = _safe_float(value.replace("%", "")) / 100 if value else None
        elif "td def" in label:
            data["td_defense"] = _safe_float(value.replace("%", "")) / 100 if value else None
        elif "sub. avg" in label:
            data["sub_avg"] = _safe_float(value)

    # Win/loss record
    record_el = soup.find("span", class_="b-content__title-record")
    if record_el:
        data.update(_parse_record(record_el.get_text(strip=True)))

    return data


# ══════════════════════════════════════════════════════════════════════════════
# FIGHTER SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_fighter(name: str) -> Optional[str]:
    """
    Search ufcstats.com for a fighter by name.
    Returns the fighter's profile URL or None if not found.
    """
    last_initial = name.strip().split()[-1][0].lower()
    url = f"{BASE_URL}/statistics/fighters?char={last_initial}&page=all"
    soup = _get(url)
    if not soup:
        return None

    rows = soup.select("tr.b-statistics__table-row")
    candidates = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link = cells[0].find("a")
        if not link:
            continue
        first = link.get_text(strip=True)
        last_cell = cells[1].find("a") or cells[1]
        last = last_cell.get_text(strip=True) if last_cell else ""
        full_name = f"{first} {last}".strip()
        fighter_url = link.get("href", "").strip()
        candidates.append((full_name, fighter_url))

    if not candidates:
        return None

    from rapidfuzz import process, fuzz
    match = process.extractOne(
        name,
        [c[0] for c in candidates],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=70,
    )

    if match:
        matched_name = match[0]
        for cname, curl in candidates:
            if cname == matched_name:
                logger.debug(f"Matched '{name}' → '{cname}' ({match[1]:.0f}%)")
                return curl

    logger.warning(f"No match found for fighter: '{name}'")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# UPCOMING CARD
# ══════════════════════════════════════════════════════════════════════════════

def get_upcoming_events() -> list[dict]:
    """Scrape upcoming UFC events from ufcstats.com."""
    url = f"{BASE_URL}/statistics/events/upcoming?page=all"
    soup = _get(url)
    if not soup:
        return []

    events = []
    rows = soup.select("tr.b-statistics__table-row")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link = cells[0].find("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        event_url = link.get("href", "").strip()
        date_str = cells[1].get_text(strip=True)
        try:
            date = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            date = None
        if name and event_url:
            events.append({"name": name, "date": date, "url": event_url})

    logger.info(f"Found {len(events)} upcoming events")
    return events


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_method(raw: str) -> str:
    """Normalize all method variants to 5 clean categories."""
    raw = raw.strip().upper()
    # Overturned/DQ results — treat as NC for modeling purposes
    if "OVERTURNED" in raw or raw.startswith("DQ"):
        return "NC"
    if any(x in raw for x in ["KO", "TKO", "DOCTOR"]):
        return "KO_TKO"
    elif "SUB" in raw or any(x in raw for x in [
        "CHOKE", "ARMBAR", "TRIANGLE", "KIMURA", "GUILLOTINE",
        "HEEL HOOK", "ANKLE", "REAR NAKED", "D'ARCE", "DARCE"
    ]):
        return "Submission"
    elif "DEC" in raw:
        return "Decision"
    elif "DRAW" in raw:
        return "Draw"
    elif "NC" in raw or "NO CONTEST" in raw:
        return "NC"
    # Default anything unknown to Decision
    return "Decision"


def _height_to_cm(height_str: str) -> Optional[float]:
    try:
        match = re.match(r"(\d+)'?\s*(\d+)\"?", height_str)
        if match:
            feet, inches = int(match.group(1)), int(match.group(2))
            return round((feet * 12 + inches) * 2.54, 1)
    except Exception:
        pass
    return None


def _reach_to_cm(reach_str: str) -> Optional[float]:
    try:
        inches = float(re.sub(r"[^\d.]", "", reach_str))
        return round(inches * 2.54, 1)
    except Exception:
        return None


def _parse_date(date_str: str) -> Optional[datetime]:
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_record(record_str: str) -> dict:
    try:
        parts = re.findall(r"\d+", record_str)
        if len(parts) >= 2:
            return {
                "wins": int(parts[0]),
                "losses": int(parts[1]),
                "draws": int(parts[2]) if len(parts) > 2 else 0,
            }
    except Exception:
        pass
    return {"wins": 0, "losses": 0, "draws": 0}


def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None
