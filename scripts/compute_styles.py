"""
scripts/compute_styles.py
──────────────────────────
Computes style fingerprints, recent form scores, and weight-class
percentile rankings for every fighter in the DB.

Run after enrich_fighters.py:
    python scripts/compute_styles.py

This populates the style_* and momentum_score columns in fighter_stats,
which the model then uses as matchup features to detect:
  - Wrestler vs striker mismatches
  - Pressure fighter vs counter-striker dynamics
  - Hot streaks and momentum shifts
  - Weight class context (a 5 TD/15min avg means different things at HW vs FW)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from loguru import logger
from sqlalchemy.orm import Session

from src.database import init_db, get_session, FighterStats, Fight, Fighter


def compute_style_fingerprint(stats: FighterStats) -> dict:
    """
    Derive style label scores from raw performance stats.

    These are NOT labels (we don't classify fighters as 'wrestler' or 'striker')
    — they're continuous scores on each dimension so the model learns the
    degree of style mismatch, not a binary category.
    """
    slpm = stats.slpm or 0.0
    sapm = stats.sapm or 0.0
    td_avg = stats.td_avg or 0.0
    td_def = stats.td_defense or 0.5
    sub_avg = stats.sub_avg or 0.0
    finish_rate = stats.finish_rate or 0.0
    wins_ko = stats.wins_ko_tko or 0
    wins_sub = stats.wins_sub or 0
    wins_dec = stats.wins_decision or 0
    total_wins = max(wins_ko + wins_sub + wins_dec, 1)

    # Pressure fighter: volume striking + takedown threat + forward pace
    # High value = dangerous going forward
    style_pressure = (slpm / 6.0) * 0.4 + (td_avg / 5.0) * 0.3 + (1 - sapm / 6.0) * 0.3
    style_pressure = float(np.clip(style_pressure, 0, 1))

    # Wrestling reliance: how much does this fighter lean on grappling?
    # High value = primarily a grappler
    style_wrestling = td_avg / (slpm + td_avg + 0.01)
    style_wrestling = float(np.clip(style_wrestling, 0, 1))

    # Striking reliance: inverse of wrestling reliance
    style_striker = slpm / (slpm + td_avg + 0.01)
    style_striker = float(np.clip(style_striker, 0, 1))

    # Finisher score: KO% and sub% weighted by how they actually finish
    ko_rate = wins_ko / total_wins
    sub_rate = wins_sub / total_wins
    style_finisher = (ko_rate * 0.6 + sub_rate * 0.4) * finish_rate
    style_finisher = float(np.clip(style_finisher, 0, 1))

    # Grappling defense: ability to stay standing + resist submissions
    grappling_defense = (td_def * 0.7 + (1 - sub_avg / 3.0) * 0.3)
    grappling_defense = float(np.clip(grappling_defense, 0, 1))

    return {
        "style_pressure":    round(style_pressure, 4),
        "style_wrestling":   round(style_wrestling, 4),
        "style_striker":     round(style_striker, 4),
        "style_finisher":    round(style_finisher, 4),
        "grappling_defense": round(grappling_defense, 4),
    }


def compute_recent_form(fighter_id: int, as_of_date, session: Session) -> dict:
    """
    Compute recent form metrics using last 3 fights.
    Recent fights are weighted more than older ones.

    Momentum score: +1 for win, -1 for loss, weighted by recency
    Weights: most recent = 3x, second = 2x, third = 1x
    """
    recent_fights = (
        session.query(Fight)
        .filter(
            ((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id)),
            Fight.fight_date < as_of_date,
            Fight.winner_id.isnot(None),
        )
        .order_by(Fight.fight_date.desc())
        .limit(3)
        .all()
    )

    if not recent_fights:
        return {"momentum_score": 0.0, "recent_finish_rate": 0.0}

    weights = [3, 2, 1][:len(recent_fights)]
    total_weight = sum(weights)
    momentum = 0.0
    finishes = 0

    for fight, weight in zip(recent_fights, weights):
        won = fight.winner_id == fighter_id
        momentum += weight * (1 if won else -1)
        if won and fight.method in ("KO_TKO", "Submission"):
            finishes += 1

    momentum_score = momentum / total_weight  # -1 to +1
    recent_finish_rate = finishes / len(recent_fights)

    return {
        "momentum_score":     round(float(momentum_score), 4),
        "recent_finish_rate": round(float(recent_finish_rate), 4),
    }


def compute_weight_class_percentiles(session: Session):
    """
    For each weight class, compute where each fighter's slpm and td_avg
    rank among all fighters in that class.

    A 4.0 slpm at Flyweight is elite; at Heavyweight it's normal.
    Percentile ranking captures this context.
    """
    from src.database import Fighter as FighterModel
    import pandas as pd

    # Get latest stats per fighter with their weight class
    all_fighters = session.query(FighterModel).all()
    rows = []
    for f in all_fighters:
        latest = (
            session.query(FighterStats)
            .filter_by(fighter_id=f.id)
            .order_by(FighterStats.as_of_date.desc())
            .first()
        )
        if latest and latest.slpm is not None:
            rows.append({
                "fighter_id": f.id,
                "stats_id": latest.id,
                "weight_class": f.weight_class or "Unknown",
                "slpm": latest.slpm or 0,
                "td_avg": latest.td_avg or 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Compute percentiles within each weight class
    def pctile(series):
        return series.rank(pct=True)

    df["slpm_pctile"] = df.groupby("weight_class")["slpm"].transform(pctile)
    df["td_avg_pctile"] = df.groupby("weight_class")["td_avg"].transform(pctile)

    # Write back to DB
    for _, row in df.iterrows():
        stats = session.query(FighterStats).get(int(row["stats_id"]))
        if stats:
            stats.slpm_pctile = round(float(row["slpm_pctile"]), 4)
            stats.td_avg_pctile = round(float(row["td_avg_pctile"]), 4)

    session.commit()
    logger.success(f"Percentiles computed for {len(df)} fighters across {df['weight_class'].nunique()} weight classes")


def compute_rolling_style(fighter_id: int, as_of_date, session: Session, window: int) -> dict:
    """
    Compute style fingerprint from the last N fights' stat snapshots.
    Uses the FighterStats snapshot closest to each recent fight date.

    This captures style evolution — a fighter who switched from wrestling
    to striking over the last 3 fights will have different l3 vs career scores.
    """
    recent_fights = (
        session.query(Fight)
        .filter(
            ((Fight.fighter_a_id == fighter_id) | (Fight.fighter_b_id == fighter_id)),
            Fight.fight_date < as_of_date,
            Fight.winner_id.isnot(None),
        )
        .order_by(Fight.fight_date.desc())
        .limit(window)
        .all()
    )

    if len(recent_fights) < max(2, window // 2):
        return {}  # not enough fights for meaningful window

    # Get stats snapshots just before each fight
    snap_stats = []
    for fight in recent_fights:
        snap = (
            session.query(FighterStats)
            .filter(
                FighterStats.fighter_id == fighter_id,
                FighterStats.as_of_date < fight.fight_date,
            )
            .order_by(FighterStats.as_of_date.desc())
            .first()
        )
        if snap and snap.slpm is not None:
            snap_stats.append(snap)

    if not snap_stats:
        return {}

    # Average the raw stats across the window, then compute style scores
    avg_slpm   = sum(s.slpm or 0 for s in snap_stats) / len(snap_stats)
    avg_td_avg = sum(s.td_avg or 0 for s in snap_stats) / len(snap_stats)
    avg_sapm   = sum(s.sapm or 0 for s in snap_stats) / len(snap_stats)

    total = avg_slpm + avg_td_avg + 0.01
    style_pressure  = float(np.clip((avg_slpm / 6.0) * 0.4 + (avg_td_avg / 5.0) * 0.3 + (1 - avg_sapm / 6.0) * 0.3, 0, 1))
    style_wrestling = float(np.clip(avg_td_avg / total, 0, 1))
    style_striker   = float(np.clip(avg_slpm / total, 0, 1))

    suffix = f"_l{window}"
    return {
        f"style_pressure{suffix}":  round(style_pressure, 4),
        f"style_wrestling{suffix}": round(style_wrestling, 4),
        f"style_striker{suffix}":   round(style_striker, 4),
    }


def run():
    init_db()
    session = get_session()

    all_stats = session.query(FighterStats).all()
    logger.info(f"Computing style features for {len(all_stats)} stat snapshots...")

    updated = 0
    for stats in tqdm(all_stats, desc="Style features"):
        try:
            style = compute_style_fingerprint(stats)
            form = compute_recent_form(stats.fighter_id, stats.as_of_date, session)

            stats.style_pressure    = style["style_pressure"]
            stats.style_wrestling   = style["style_wrestling"]
            stats.style_striker     = style["style_striker"]
            stats.style_finisher    = style["style_finisher"]
            stats.grappling_defense = style["grappling_defense"]
            stats.momentum_score    = form["momentum_score"]
            stats.recent_finish_rate = form["recent_finish_rate"]

            # Rolling style windows — last 3 and last 5 fights
            if stats.as_of_date:
                for window in [3, 5]:
                    rolling = compute_rolling_style(stats.fighter_id, stats.as_of_date, session, window)
                    for k, v in rolling.items():
                        setattr(stats, k, v)

            updated += 1
        except Exception as e:
            logger.debug(f"Failed stats {stats.id}: {e}")

    session.commit()
    logger.success(f"Style features computed for {updated} snapshots")

    logger.info("Computing weight class percentiles...")
    compute_weight_class_percentiles(session)

    session.close()
    logger.success("Done. Now retrain: rm data/processed/training_dataset.csv && python scripts/train_model.py")


if __name__ == "__main__":
    run()
