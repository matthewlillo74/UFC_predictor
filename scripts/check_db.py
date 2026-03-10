"""
scripts/check_db.py
────────────────────
Quick sanity check on what's in the database.
Run after loading historical data to verify everything looks right.

Usage:
    python scripts/check_db.py
    python scripts/check_db.py --fighter "Islam Makhachev"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.database import init_db, get_session, Fighter, Fight, Event, EloRating, FighterStats
from src.features.elo_calculator import EloCalculator

def main(fighter_name: str = None):
    init_db()
    session = get_session()

    print("\n" + "═" * 50)
    print("  UFC PREDICTOR — DATABASE STATUS")
    print("═" * 50)

    event_count = session.query(Event).count()
    fight_count = session.query(Fight).count()
    fighter_count = session.query(Fighter).count()
    elo_count = session.query(EloRating).count()
    stats_count = session.query(FighterStats).count()

    print(f"\n  Events:           {event_count:,}")
    print(f"  Fights:           {fight_count:,}")
    print(f"  Fighters:         {fighter_count:,}")
    print(f"  Elo ratings:      {elo_count:,}")
    print(f"  Stats snapshots:  {stats_count:,}")

    # Most recent event
    latest_event = session.query(Event).order_by(Event.date.desc()).first()
    if latest_event:
        print(f"\n  Latest event: {latest_event.name} ({latest_event.date.strftime('%Y-%m-%d') if latest_event.date else 'N/A'})")

    # Elo leaderboard
    calc = EloCalculator(session)
    leaderboard = calc.get_leaderboard(top_n=10)
    if leaderboard:
        print("\n  TOP 10 BY ELO RATING")
        print("  " + "─" * 40)
        for entry in leaderboard:
            print(f"  {entry['rank']:2}. {entry['name']:<25} {entry['elo']:.0f}")

    # Fighter lookup
    if fighter_name:
        print(f"\n  FIGHTER LOOKUP: {fighter_name}")
        print("  " + "─" * 40)
        from rapidfuzz import process, fuzz
        all_fighters = session.query(Fighter).all()
        match = process.extractOne(
            fighter_name.lower(),
            [f.name_normalized for f in all_fighters],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=60,
        )
        if match:
            fighter = next(f for f in all_fighters if f.name_normalized == match[0])
            elo = calc.get_rating(fighter.id)
            fights = session.query(Fight).filter(
                (Fight.fighter_a_id == fighter.id) | (Fight.fighter_b_id == fighter.id)
            ).order_by(Fight.fight_date.desc()).limit(5).all()

            print(f"  Name:  {fighter.name}")
            print(f"  Elo:   {elo:.0f}")
            print(f"  Last 5 fights:")
            for fight in fights:
                opponent_id = fight.fighter_b_id if fight.fighter_a_id == fighter.id else fight.fighter_a_id
                opponent = session.query(Fighter).get(opponent_id)
                result = "W" if fight.winner_id == fighter.id else ("D" if fight.winner_id is None else "L")
                date = fight.fight_date.strftime("%Y-%m-%d") if fight.fight_date else "?"
                print(f"    {date}  {result}  vs {opponent.name if opponent else '?'}  ({fight.method})")
        else:
            print(f"  Fighter not found: {fighter_name}")

    print("\n" + "═" * 50 + "\n")
    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fighter", type=str, default=None)
    args = parser.parse_args()
    main(fighter_name=args.fighter)
