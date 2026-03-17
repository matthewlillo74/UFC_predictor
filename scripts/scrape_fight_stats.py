"""
scripts/scrape_fight_stats.py
──────────────────────────────
Scrapes per-fight knockdown and significant strike data from ufcstats fight
detail pages, then backfills kd_landed_per_fight, kd_absorbed_per_fight,
and kd_ratio into FighterStats snapshots.

This replaces the proxy durability score with real measured knockdown data.

ufcstats fight detail page (e.g. ufcstats.com/fight-details/abc123) has a
"Totals" table with columns:
  Fighter | KD | Sig. Str. | Sig. Str. % | Total Str. | Td | Td % | Sub. Att | Rev. | Ctrl

Usage:
    # Scrape all fights (slow — ~8500 pages, 2 sec each = ~5 hours)
    python scripts/scrape_fight_stats.py --all

    # Scrape only fights missing from fight_stats table (resume-safe)
    python scripts/scrape_fight_stats.py --missing

    # Scrape last N events only (for incremental updates)
    python scripts/scrape_fight_stats.py --events 10

    # After scraping, backfill KD features into FighterStats snapshots
    python scripts/scrape_fight_stats.py --backfill

    # Full workflow: scrape missing + backfill
    python scripts/scrape_fight_stats.py --missing --backfill
"""

import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
from loguru import logger

from src.database import init_db, get_session, Fight, Fighter, FightStats, FighterStats, Event
from config import SCRAPE_DELAY_SECONDS


UA = "Mozilla/5.0 (compatible; UFC-Predictor/1.0)"


def _get(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=12)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.debug(f"Failed to fetch {url}: {e}")
    return None


def _parse_ctrl(ctrl_str: str) -> int:
    """Parse '2:34' or '--' control time string into seconds."""
    if not ctrl_str or ctrl_str.strip() in ("--", ""):
        return 0
    m = re.match(r"(\d+):(\d+)", ctrl_str.strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def _safe_int(val: str) -> int:
    """Parse 'N of M' or plain integer strings."""
    if not val or val.strip() in ("--", ""):
        return 0
    # Handle "N of M" format (e.g. "3 of 7")
    m = re.match(r"(\d+)\s+of\s+(\d+)", val.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        return int(val.strip()), None
    except Exception:
        return 0, None


def scrape_fight_detail(fight_url: str) -> dict | None:
    """
    Scrape fight stats from a ufcstats fight detail page.

    VERIFIED table structure (from live page inspection):
      classed[0]   (b-fight-details__table): Fight-level TOTALS
                   Cols: Fighter|KD|SigStr|SigStr%|TotalStr|TD|TD%|SubAtt|Rev|Ctrl
      unclassed[0]: Fight-level SIG STRIKES by location
                   Cols: Fighter|SigStr|SigStr%|Head|Body|Leg|Distance|Clinch|Ground
      classed[1]   (b-fight-details__table): Round-by-round TOTALS (same cols as classed[0])
      unclassed[1]: Round-by-round SIG STRIKES by location (same cols as unclassed[0])

    Each data row has ONE <tr> with TWO <p> tags per <td> — one per fighter.
    """
    soup = _get(fight_url)
    if not soup:
        return None

    all_tables = soup.find_all("table")
    if not all_tables:
        return None

    classed   = [t for t in all_tables if t.get("class")]
    unclassed = [t for t in all_tables if not t.get("class")]

    if not classed:
        return None

    # VERIFIED table structure across all fight pages:
    #   classed[0]:   10 cells = fight-level TOTALS (Fighter|KD|Sig|Sig%|Total|TD|TD%|Sub|Rev|Ctrl)
    #   unclassed[1]: 9 cells, 1 row = fight-level SIG STRIKE LOCATIONS (Fighter|Sig|Sig%|Head|Body|Leg|Dist|Clinch|Ground)
    #   classed[1]:   9 cells, N rows = round-by-round SIG STRIKE LOCATIONS (same format as unclassed[1])
    #   unclassed[0]: 10 cells, 1 row = duplicate of fight totals (ignored)
    # NOTE: There is NO separate round-level totals table. Only round-level sig strikes exist.

    def get_p(row, cell_idx, fighter_idx):
        """Get text value for one fighter from a cell's p-tags."""
        cells = row.find_all("td")
        if cell_idx >= len(cells):
            return ""
        ps = cells[cell_idx].find_all("p")
        if fighter_idx < len(ps):
            return ps[fighter_idx].get_text(strip=True)
        return cells[cell_idx].get_text(strip=True) if fighter_idx == 0 else ""

    def parse_la(val):
        """Parse 'N of M' → (N, M) or plain int → (N, 0)."""
        if not val or val.strip() in ("--", "---", ""):
            return 0, 0
        m = re.match(r"(\d+)\s+of\s+(\d+)", val.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        try:
            return int(val.strip()), 0
        except Exception:
            return 0, 0

    def parse_totals_rows(table):
        """
        Parse a TOTALS table (classed).
        Cols: Fighter|KD|SigStr|SigStr%|TotalStr|TD|TD%|SubAtt|Rev|Ctrl
        Returns list of dicts, one per (row, fighter) combination.
        """
        rows = [r for r in table.find_all("tr") if r.find("td")]
        results = []
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            # Get fighter names from cell 0
            ps = cells[0].find_all("p")
            names = [p.get_text(strip=True) for p in ps] if ps else [cells[0].get_text(strip=True)]
            if not names or not names[0]:
                continue
            for fi, name in enumerate(names[:2]):
                kd_raw = get_p(row, 1, fi)
                kd = int(kd_raw) if kd_raw.isdigit() else 0
                sig_l, sig_a = parse_la(get_p(row, 2, fi))
                total_l, _   = parse_la(get_p(row, 4, fi))
                td_l, td_a   = parse_la(get_p(row, 5, fi))
                sub_raw = get_p(row, 7, fi)
                sub_att = int(sub_raw) if sub_raw.isdigit() else 0
                rev_raw = get_p(row, 8, fi)
                reversals = int(rev_raw) if rev_raw.isdigit() else 0
                ctrl_secs = _parse_ctrl(get_p(row, 9, fi))
                results.append({
                    "fighter_name": name, "kd": kd,
                    "sig_landed": sig_l, "sig_attempted": sig_a,
                    "total_landed": total_l, "td_landed": td_l,
                    "td_attempted": td_a, "sub_att": sub_att,
                    "reversals": reversals, "ctrl_secs": ctrl_secs,
                })
        return results

    def parse_location_rows(table):
        """
        Parse a SIG STRIKES LOCATION table (unclassed).
        Cols: Fighter|SigStr|SigStr%|Head|Body|Leg|Distance|Clinch|Ground
        Returns list of dicts, one per (row, fighter) combination.
        """
        rows = [r for r in table.find_all("tr") if r.find("td")]
        results = []
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            ps = cells[0].find_all("p")
            names = [p.get_text(strip=True) for p in ps] if ps else [cells[0].get_text(strip=True)]
            if not names or not names[0]:
                continue
            for fi, name in enumerate(names[:2]):
                sig_l, sig_a = parse_la(get_p(row, 1, fi))   # sig strikes at col 1
                head_l, head_a = parse_la(get_p(row, 3, fi))
                body_l, body_a = parse_la(get_p(row, 4, fi))
                leg_l,  leg_a  = parse_la(get_p(row, 5, fi))
                dist_l, _   = parse_la(get_p(row, 6, fi))
                clinch_l, _ = parse_la(get_p(row, 7, fi))
                ground_l, _ = parse_la(get_p(row, 8, fi))
                results.append({
                    "fighter_name": name,
                    "sig_landed": sig_l, "sig_attempted": sig_a,
                    "head_landed": head_l, "head_attempted": head_a,
                    "body_landed": body_l, "body_attempted": body_a,
                    "leg_landed":  leg_l,  "leg_attempted":  leg_a,
                    "distance_landed": dist_l,
                    "clinch_landed":   clinch_l,
                    "ground_landed":   ground_l,
                })
        return results

    # ── Parse tables ───────────────────────────────────────────────────────

    # Fight-level totals (classed[0], 10 cells)
    fight_totals = parse_totals_rows(classed[0])
    if len(fight_totals) < 2:
        return None

    # Fight-level sig strike locations (unclassed[1], 9 cells, 1 row)
    fight_locs = parse_location_rows(unclassed[1]) if len(unclassed) > 1 else []
    loc_by_name = {}
    for r in fight_locs:
        name = r.get("fighter_name", "").strip()
        if name:
            loc_by_name[name] = r
    loc_by_first = {name.split()[0]: r for name, r in loc_by_name.items()}

    # Build combined totals
    fa = fight_totals[0]
    fb = fight_totals[1]
    fa["kd_absorbed"]  = fb["kd"]
    fb["kd_absorbed"]  = fa["kd"]
    fa["sig_absorbed"] = fb["sig_landed"]
    fb["sig_absorbed"] = fa["sig_landed"]
    for entry in [fa, fb]:
        name = entry["fighter_name"]
        loc = loc_by_name.get(name) or loc_by_first.get(name.split()[0] if name else "")
        if loc:
            entry.update({k: v for k, v in loc.items() if k != "fighter_name"})
    totals = [fa, fb]

    # Round-by-round sig strike locations (classed[1], 9 cells, N rows = N rounds)
    # NOTE: No round-level totals table exists — only sig strike locations per round.
    # Cardio decay is computed from round sig_landed (col [1] on 9-cell format).
    rounds = []
    if len(classed) > 1:
        round_locs_list = parse_location_rows(classed[1])
        # pair up: every 2 entries = one round
        for i in range(0, len(round_locs_list) - 1, 2):
            rnum = (i // 2) + 1
            for entry in [round_locs_list[i], round_locs_list[i + 1]]:
                rounds.append({"round_num": rnum, **entry})

    return {"totals": totals, "rounds": rounds}
    soup = _get(fight_url)
    if not soup:
        return None

    all_tables = soup.find_all("table")
    if len(all_tables) < 2:
        return None

    unclassed = [t for t in all_tables if not t.get("class")]
    classed   = [t for t in all_tables if t.get("class")]

    if not unclassed:
        return None

    def cell_values(row, idx):
        """Get [fighter_a_value, fighter_b_value] from a cell's two <p> tags."""
        cells = row.find_all("td")
        if idx >= len(cells):
            return ["", ""]
        ps = cells[idx].find_all("p")
        if len(ps) >= 2:
            return [ps[0].get_text(strip=True), ps[1].get_text(strip=True)]
        elif len(ps) == 1:
            return [ps[0].get_text(strip=True), ""]
        else:
            return [cells[idx].get_text(strip=True), ""]

    def fighter_names_from_row(row):
        """Extract both fighter names from cell 0's <a> tags or <p> tags."""
        cells = row.find_all("td")
        if not cells:
            return None, None
        links = cells[0].find_all("a")
        if len(links) >= 2:
            return links[0].get_text(strip=True), links[1].get_text(strip=True)
        ps = cells[0].find_all("p")
        if len(ps) >= 2:
            return ps[0].get_text(strip=True), ps[1].get_text(strip=True)
        return None, None

    def parse_landed_attempted(val):
        """Parse 'N of M' → (N, M), plain int → (N, 0)."""
        if not val or val.strip() in ("--", ""):
            return 0, 0
        m = re.match(r"(\d+)\s+of\s+(\d+)", val.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        try:
            return int(val.strip()), 0
        except Exception:
            return 0, 0

    def parse_totals_table(table):
        """
        Parse a totals table. Returns list of 2 dicts (one per fighter).
        Each row has 2 p-tags per cell for the two fighters.
        """
        rows = [r for r in table.find_all("tr") if r.find("td")]
        if not rows:
            return []

        # For fight-level: just one data row with 2 p-tags per cell
        # For round-level: one data row per round, each with 2 p-tags per cell
        results_a = []
        results_b = []

        for row in rows:
            name_a, name_b = fighter_names_from_row(row)
            if not name_a:
                continue

            def v(idx, fighter_idx):
                vals = cell_values(row, idx)
                return vals[fighter_idx] if fighter_idx < len(vals) else ""

            for fi, name in [(0, name_a), (1, name_b)]:
                kd_raw = v(1, fi)
                kd = int(kd_raw) if kd_raw.isdigit() else 0
                sig_l, sig_a = parse_landed_attempted(v(2, fi))
                total_l, _ = parse_landed_attempted(v(4, fi))
                td_l, td_a = parse_landed_attempted(v(5, fi))
                sub_raw = v(7, fi)
                sub_att = int(sub_raw) if sub_raw.isdigit() else 0
                rev_raw = v(8, fi)
                reversals = int(rev_raw) if rev_raw.isdigit() else 0
                ctrl_secs = _parse_ctrl(v(9, fi))
                entry = {
                    "fighter_name": name, "kd": kd,
                    "sig_landed": sig_l, "sig_attempted": sig_a,
                    "total_landed": total_l, "td_landed": td_l,
                    "td_attempted": td_a, "sub_att": sub_att,
                    "reversals": reversals, "ctrl_secs": ctrl_secs,
                }
                if fi == 0:
                    results_a.append(entry)
                else:
                    results_b.append(entry)

        return results_a, results_b

    def parse_location_table(table):
        """Parse a sig strikes location table. Returns dict: fighter_name → {head, body, leg, ...}"""
        rows = [r for r in table.find_all("tr") if r.find("td")]
        if not rows:
            return {}

        result = {}
        for row in rows:
            name_a, name_b = fighter_names_from_row(row)
            if not name_a:
                continue
            for fi, name in [(0, name_a), (1, name_b)]:
                def v(idx): return cell_values(row, idx)[fi]
                head_l, head_a = parse_landed_attempted(v(3))
                body_l, body_a = parse_landed_attempted(v(4))
                leg_l, leg_a   = parse_landed_attempted(v(5))
                dist_l, _  = parse_landed_attempted(v(6))
                clinch_l, _ = parse_landed_attempted(v(7))
                ground_l, _ = parse_landed_attempted(v(8))
                result[name] = {
                    "head_landed": head_l, "head_attempted": head_a,
                    "body_landed": body_l, "body_attempted": body_a,
                    "leg_landed":  leg_l,  "leg_attempted": leg_a,
                    "distance_landed": dist_l,
                    "clinch_landed":   clinch_l,
                    "ground_landed":   ground_l,
                }
        return result

    # ── Parse all 4 tables ─────────────────────────────────────────────────

    # Fight-level totals
    fight_a, fight_b = parse_totals_table(unclassed[0]) if unclassed else ([], [])
    if not fight_a or not fight_b:
        return None

    # Fight-level sig strike locations
    fight_locs = parse_location_table(classed[0]) if classed else {}

    # Build totals for both fighters (fight-level = first entry only)
    totals = []
    for entries, other_entries in [(fight_a, fight_b), (fight_b, fight_a)]:
        if not entries:
            continue
        entry = dict(entries[0])  # fight-level = one entry
        name = entry["fighter_name"]
        # Cross-assign absorbed stats
        opp = other_entries[0] if other_entries else {}
        entry["kd_absorbed"]  = opp.get("kd", 0)
        entry["sig_absorbed"] = opp.get("sig_landed", 0)
        # Merge location data
        if name in fight_locs:
            entry.update(fight_locs[name])
        totals.append(entry)

    # Round-by-round totals
    rounds = []
    if len(unclassed) > 1:
        round_locs = parse_location_table(classed[1]) if len(classed) > 1 else {}
        round_a_list, round_b_list = parse_totals_table(unclassed[1])
        for rnum, (ra, rb) in enumerate(zip(round_a_list, round_b_list), start=1):
            for entry in [ra, rb]:
                row_dict = {"round_num": rnum, **entry}
                name = entry["fighter_name"]
                if name in round_locs:
                    row_dict.update(round_locs[name])
                rounds.append(row_dict)

    return {"totals": totals, "rounds": rounds}


def store_fight_stats(session, fight, scraped: dict):
    """Store scraped fight stats (totals + rounds + locations) into DB."""
    from rapidfuzz import process, fuzz
    from src.ingestion.data_loader import normalize_name
    from src.database import RoundStats

    fighters = [
        session.get(Fighter, fight.fighter_a_id),
        session.get(Fighter, fight.fighter_b_id),
    ]
    fighter_names = {normalize_name(f.name): f for f in fighters if f}

    def match_fighter(name):
        norm = normalize_name(name)
        m = process.extractOne(norm, list(fighter_names.keys()),
                               scorer=fuzz.token_sort_ratio, score_cutoff=80)
        return fighter_names[m[0]] if m else None

    stored = 0

    # ── Store fight-level totals + location data ───────────────────────────
    totals = scraped.get("totals", [])
    for row in totals:
        fighter = match_fighter(row["fighter_name"])
        if not fighter:
            continue
        existing = session.query(FightStats).filter_by(
            fight_id=fight.id, fighter_id=fighter.id
        ).first()
        if existing:
            # Update with location data if not already stored
            if existing.head_landed is None and row.get("head_landed") is not None:
                existing.head_landed    = row.get("head_landed")
                existing.head_attempted = row.get("head_attempted")
                existing.body_landed    = row.get("body_landed")
                existing.body_attempted = row.get("body_attempted")
                existing.leg_landed     = row.get("leg_landed")
                existing.leg_attempted  = row.get("leg_attempted")
                existing.distance_landed = row.get("distance_landed")
                existing.clinch_landed  = row.get("clinch_landed")
                existing.ground_landed  = row.get("ground_landed")
            continue

        fs = FightStats(
            fight_id=fight.id, fighter_id=fighter.id,
            knockdowns=row.get("kd", 0),
            knockdowns_absorbed=row.get("kd_absorbed", 0),
            sig_strikes_landed=row.get("sig_landed"),
            sig_strikes_attempted=row.get("sig_attempted"),
            sig_strikes_absorbed=row.get("sig_absorbed"),
            total_strikes_landed=row.get("total_landed"),
            takedowns_landed=row.get("td_landed"),
            takedowns_attempted=row.get("td_attempted"),
            submission_attempts=row.get("sub_att"),
            reversals=row.get("reversals"),
            control_time_secs=row.get("ctrl_secs"),
            head_landed=row.get("head_landed"),
            head_attempted=row.get("head_attempted"),
            body_landed=row.get("body_landed"),
            body_attempted=row.get("body_attempted"),
            leg_landed=row.get("leg_landed"),
            leg_attempted=row.get("leg_attempted"),
            distance_landed=row.get("distance_landed"),
            clinch_landed=row.get("clinch_landed"),
            ground_landed=row.get("ground_landed"),
            scraped_at=datetime.utcnow(),
        )
        session.add(fs)
        stored += 1

    # ── Store round-level data ─────────────────────────────────────────────
    rounds = scraped.get("rounds", [])
    for row in rounds:
        fighter = match_fighter(row["fighter_name"])
        if not fighter:
            continue
        rnum = row.get("round_num", 1)
        existing = session.query(RoundStats).filter_by(
            fight_id=fight.id, fighter_id=fighter.id, round_num=rnum
        ).first()
        if existing:
            continue
        rs = RoundStats(
            fight_id=fight.id, fighter_id=fighter.id, round_num=rnum,
            knockdowns=row.get("kd", 0),
            sig_strikes_landed=row.get("sig_landed"),
            sig_strikes_attempted=row.get("sig_attempted"),
            total_strikes_landed=row.get("total_landed"),
            takedowns_landed=row.get("td_landed"),
            takedowns_attempted=row.get("td_attempted"),
            submission_attempts=row.get("sub_att"),
            reversals=row.get("reversals"),
            control_time_secs=row.get("ctrl_secs"),
            head_landed=row.get("head_landed"),
            body_landed=row.get("body_landed"),
            leg_landed=row.get("leg_landed"),
            distance_landed=row.get("distance_landed"),
            clinch_landed=row.get("clinch_landed"),
            ground_landed=row.get("ground_landed"),
            scraped_at=datetime.utcnow(),
        )
        session.add(rs)
        stored += 1

    if stored:
        session.commit()
    return stored


def backfill_fight_urls(session, delay: float = 1.0):
    """
    Backfill fight_url for existing Fight rows that don't have it.
    Step 1: Populate event.url by scraping the ufcstats event list pages.
    Step 2: Re-scrape each event page to get individual fight detail URLs.
    This only needs to run once after adding the fight_url column.
    """
    from src.ingestion.fight_scraper import get_all_events, get_event_fights
    from src.database import Event as EventModel

    # Step 1: populate event URLs we don't have yet
    events_missing_url = session.query(EventModel).filter(
        (EventModel.url == None) | (EventModel.url == "")
    ).all()

    if events_missing_url:
        logger.info(f"Fetching event URLs for {len(events_missing_url)} events...")
        try:
            all_scraped_events = get_all_events()
            # Build name → url map
            name_url_map = {e["name"].strip().lower(): e["url"] for e in all_scraped_events}
            filled = 0
            for ev in events_missing_url:
                key = ev.name.strip().lower()
                if key in name_url_map:
                    ev.url = name_url_map[key]
                    filled += 1
            session.commit()
            logger.success(f"Populated URLs for {filled} events")
        except Exception as e:
            logger.error(f"Could not fetch event list: {e}")

    # Step 2: for each event with a URL, scrape fight detail URLs
    fights_missing = session.query(Fight).filter(
        Fight.winner_id.isnot(None),
        (Fight.fight_url == None) | (Fight.fight_url == "")
    ).all()

    if not fights_missing:
        logger.info("All fights already have URLs")
        return

    logger.info(f"Backfilling fight_url for {len(fights_missing)} fights...")

    event_ids = list({f.event_id for f in fights_missing if f.event_id})
    events = session.query(EventModel).filter(
        EventModel.id.in_(event_ids),
        EventModel.url != "",
        EventModel.url.isnot(None),
    ).all()

    fight_lookup = {(f.fighter_a_id, f.fighter_b_id): f for f in fights_missing}
    fight_lookup.update({(f.fighter_b_id, f.fighter_a_id): f for f in fights_missing})

    from src.database import Fighter
    from src.ingestion.data_loader import normalize_name
    from rapidfuzz import process, fuzz

    all_fighters = session.query(Fighter).all()
    name_map = {normalize_name(f.name): f for f in all_fighters}

    updated = 0
    for event in tqdm(events, desc="Backfilling fight URLs"):
        scraped_fights = get_event_fights(event.url)
        if not scraped_fights:
            time.sleep(delay)
            continue

        for sf in scraped_fights:
            if not sf.get("fight_url"):
                continue
            fa_norm = normalize_name(sf["fighter_a_name"])
            fb_norm = normalize_name(sf["fighter_b_name"])
            fa_match = process.extractOne(fa_norm, list(name_map.keys()), scorer=fuzz.token_sort_ratio, score_cutoff=85)
            fb_match = process.extractOne(fb_norm, list(name_map.keys()), scorer=fuzz.token_sort_ratio, score_cutoff=85)
            if not fa_match or not fb_match:
                continue
            fa = name_map[fa_match[0]]
            fb = name_map[fb_match[0]]
            fight = fight_lookup.get((fa.id, fb.id)) or fight_lookup.get((fb.id, fa.id))
            if fight and not fight.fight_url:
                fight.fight_url = sf["fight_url"]
                updated += 1

        time.sleep(delay)

    session.commit()
    logger.success(f"Backfilled fight_url for {updated} fights")


def scrape_all(session, fights_to_scrape_list: list, delay: float = SCRAPE_DELAY_SECONDS):
    """Scrape fight detail pages and store results."""
    stored_total = 0
    failed = 0

    for fight in tqdm(fights_to_scrape_list, desc="Scraping fight stats"):
        fight_url = getattr(fight, "fight_url", None)
        if not fight_url:
            continue

        scraped = scrape_fight_detail(fight.fight_url)
        if scraped:
            n = store_fight_stats(session, fight, scraped)
            stored_total += n
        else:
            failed += 1

        time.sleep(delay)

    logger.success(f"Scraped {stored_total} fight-fighter stat rows. Failed: {failed}")
    return stored_total


def backfill_kd_features(session):
    """
    Recompute kd_landed_per_fight, kd_absorbed_per_fight, kd_ratio
    for all FighterStats snapshots using fight_stats data.
    """
    logger.info("Backfilling knockdown durability features into FighterStats...")
    fighters = session.query(Fighter).all()
    updated = 0

    for fighter in tqdm(fighters, desc="KD features"):
        fight_stats_rows = (
            session.query(FightStats, Fight)
            .join(Fight, FightStats.fight_id == Fight.id)
            .filter(FightStats.fighter_id == fighter.id)
            .filter(Fight.fight_date.isnot(None))
            .order_by(Fight.fight_date)
            .all()
        )
        if not fight_stats_rows:
            continue

        snapshots = (
            session.query(FighterStats)
            .filter_by(fighter_id=fighter.id)
            .order_by(FighterStats.as_of_date)
            .all()
        )

        for snap in snapshots:
            if not snap.as_of_date:
                continue
            prior = [fs for fs, fight in fight_stats_rows if fight.fight_date < snap.as_of_date]
            if not prior:
                continue
            n = len(prior)
            kd_landed   = sum(fs.knockdowns or 0 for fs in prior) / n
            kd_absorbed = sum(fs.knockdowns_absorbed or 0 for fs in prior) / n
            kd_ratio    = kd_landed / (kd_absorbed + 0.1)
            snap.kd_landed_per_fight  = round(kd_landed, 4)
            snap.kd_absorbed_per_fight = round(kd_absorbed, 4)
            snap.kd_ratio             = round(kd_ratio, 4)
            updated += 1

    session.commit()
    logger.success(f"KD features backfilled for {updated} snapshots")


def backfill_strike_and_cardio_features(session):
    """
    Compute strike location rates and cardio decay from scraped data
    and store in FighterStats snapshots.
    """
    from src.database import RoundStats

    logger.info("Backfilling strike location and cardio decay features...")
    fighters = session.query(Fighter).all()
    updated = 0

    for fighter in tqdm(fighters, desc="Strike/cardio features"):
        # Get all fight stats for this fighter ordered by fight date
        fight_stats_rows = (
            session.query(FightStats, Fight)
            .join(Fight, FightStats.fight_id == Fight.id)
            .filter(FightStats.fighter_id == fighter.id)
            .filter(Fight.fight_date.isnot(None))
            .order_by(Fight.fight_date)
            .all()
        )
        round_stats_rows = (
            session.query(RoundStats, Fight)
            .join(Fight, RoundStats.fight_id == Fight.id)
            .filter(RoundStats.fighter_id == fighter.id)
            .filter(Fight.fight_date.isnot(None))
            .order_by(Fight.fight_date, RoundStats.round_num)
            .all()
        )

        if not fight_stats_rows:
            continue

        snapshots = (
            session.query(FighterStats)
            .filter_by(fighter_id=fighter.id)
            .order_by(FighterStats.as_of_date)
            .all()
        )

        for snap in snapshots:
            if not snap.as_of_date:
                continue

            prior_fs = [fs for fs, fight in fight_stats_rows if fight.fight_date < snap.as_of_date]
            prior_rs = [(rs, fight) for rs, fight in round_stats_rows if fight.fight_date < snap.as_of_date]

            if not prior_fs:
                continue

            # ── Strike location rates ──────────────────────────────────────
            total_sig = sum(fs.sig_strikes_landed or 0 for fs in prior_fs)
            if total_sig > 0:
                snap.head_strike_rate    = round(sum(fs.head_landed or 0 for fs in prior_fs) / total_sig, 4)
                snap.body_strike_rate    = round(sum(fs.body_landed or 0 for fs in prior_fs) / total_sig, 4)
                snap.leg_strike_rate     = round(sum(fs.leg_landed  or 0 for fs in prior_fs) / total_sig, 4)
                snap.ground_strike_share = round(sum(fs.ground_landed or 0 for fs in prior_fs) / total_sig, 4)

            # ── Cardio decay ───────────────────────────────────────────────
            # Average (round3_sig / round1_sig) across all fights that went 3+ rounds
            decay_values = []
            # Group round stats by fight
            from collections import defaultdict
            by_fight = defaultdict(list)
            for rs, fight in prior_rs:
                by_fight[fight.id].append(rs)

            for fight_id, rounds in by_fight.items():
                r1 = next((r.sig_strikes_landed for r in rounds if r.round_num == 1), None)
                r3 = next((r.sig_strikes_landed for r in rounds if r.round_num == 3), None)
                if r1 and r1 > 0 and r3 is not None:
                    decay_values.append(r3 / r1)

            if decay_values:
                snap.cardio_decay = round(sum(decay_values) / len(decay_values), 4)

            # Early output share: avg fraction of strikes thrown in round 1
            early_values = []
            for fight_id, rounds in by_fight.items():
                r1 = next((r.sig_strikes_landed for r in rounds if r.round_num == 1), None)
                total = sum(r.sig_strikes_landed or 0 for r in rounds)
                if r1 and total > 0:
                    early_values.append(r1 / total)
            if early_values:
                snap.early_output_share = round(sum(early_values) / len(early_values), 4)

            updated += 1

    session.commit()
    logger.success(f"Strike/cardio features backfilled for {updated} snapshots")
    """
    Recompute kd_landed_per_fight, kd_absorbed_per_fight, kd_ratio
    for all FighterStats snapshots using fight_stats data.

    For each snapshot (fighter, as_of_date), aggregate all fight_stats
    for that fighter from fights BEFORE as_of_date.
    """
    logger.info("Backfilling knockdown durability features into FighterStats...")

    fighters = session.query(Fighter).all()
    updated = 0

    for fighter in tqdm(fighters, desc="KD features"):
        # Get all fight stats for this fighter, ordered by fight date
        fight_stats_rows = (
            session.query(FightStats, Fight)
            .join(Fight, FightStats.fight_id == Fight.id)
            .filter(FightStats.fighter_id == fighter.id)
            .filter(Fight.fight_date.isnot(None))
            .order_by(Fight.fight_date)
            .all()
        )

        if not fight_stats_rows:
            continue

        # For each snapshot, compute cumulative KD stats up to that date
        snapshots = (
            session.query(FighterStats)
            .filter_by(fighter_id=fighter.id)
            .order_by(FighterStats.as_of_date)
            .all()
        )

        for snap in snapshots:
            if not snap.as_of_date:
                continue

            prior = [
                fs for fs, fight in fight_stats_rows
                if fight.fight_date < snap.as_of_date
            ]

            if not prior:
                continue

            n = len(prior)
            kd_landed = sum(fs.knockdowns or 0 for fs in prior) / n
            kd_absorbed = sum(fs.knockdowns_absorbed or 0 for fs in prior) / n
            kd_ratio = kd_landed / (kd_absorbed + 0.1)  # avoid div/0

            snap.kd_landed_per_fight = round(kd_landed, 4)
            snap.kd_absorbed_per_fight = round(kd_absorbed, 4)
            snap.kd_ratio = round(kd_ratio, 4)
            updated += 1

    session.commit()
    logger.success(f"KD features backfilled for {updated} snapshots")


def main():
    parser = argparse.ArgumentParser(description="Scrape fight stats and backfill durability features")
    parser.add_argument("--all",           action="store_true", help="Scrape all fights (slow)")
    parser.add_argument("--missing",       action="store_true", help="Scrape only fights not yet in fight_stats")
    parser.add_argument("--rescrape",      action="store_true", help="Re-scrape fights missing round/location data")
    parser.add_argument("--events",        type=int, default=None, help="Scrape last N events only")
    parser.add_argument("--backfill",      action="store_true", help="Backfill KD features into FighterStats")
    parser.add_argument("--backfill-all",  action="store_true", help="Backfill KD + strike location + cardio decay")
    parser.add_argument("--backfill-urls", action="store_true", help="Backfill fight_url for existing Fight rows")
    parser.add_argument("--delay",         type=float, default=SCRAPE_DELAY_SECONDS, help="Seconds between requests")
    args = parser.parse_args()

    init_db()
    session = get_session()

    if args.backfill_urls:
        backfill_fight_urls(session, delay=args.delay)

    if args.all or args.missing or args.rescrape or args.events:
        query = session.query(Fight).filter(Fight.winner_id.isnot(None))

        if args.events:
            events = (session.query(Event).order_by(Event.date.desc()).limit(args.events).all())
            event_ids = [e.id for e in events]
            query = query.filter(Fight.event_id.in_(event_ids))

        fights = query.order_by(Fight.fight_date.desc()).all()

        if args.missing:
            already_scraped = {fs.fight_id for fs in session.query(FightStats.fight_id).all()}
            fights = [f for f in fights if f.id not in already_scraped]
            logger.info(f"Fights missing from fight_stats: {len(fights)}")
        elif args.rescrape:
            # Fights that exist in fight_stats but have no round data
            from src.database import RoundStats
            has_rounds = {rs.fight_id for rs in session.query(RoundStats.fight_id).all()}
            fights = [f for f in fights if f.id not in has_rounds]
            logger.info(f"Fights missing round data: {len(fights)}")
        else:
            logger.info(f"Fights to scrape: {len(fights)}")

        if fights:
            scrape_all(session, fights, delay=args.delay)
        else:
            logger.info("Nothing to scrape")

    if args.backfill or args.backfill_all:
        backfill_kd_features(session)

    if args.backfill_all:
        backfill_strike_and_cardio_features(session)
        logger.success("Next: rm data/processed/training_dataset.csv && python scripts/train_model.py")

    session.close()


if __name__ == "__main__":
    main()
