"""
src/database.py
───────────────
SQLAlchemy ORM models — the full schema for the UFC Predictor system.
Every table has a clear purpose. Nothing is bolted on later.
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, ForeignKey, Text, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config import DATABASE_URL

Base = declarative_base()


# ── Fighter ───────────────────────────────────────────────────────────────────

class Fighter(Base):
    """
    One row per unique fighter.
    Static attributes that don't change fight-to-fight.
    """
    __tablename__ = "fighters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    name_normalized = Column(String(100), nullable=False, unique=True)  # lowercase, stripped
    nickname = Column(String(100))
    date_of_birth = Column(DateTime)
    nationality = Column(String(50))
    height_cm = Column(Float)
    reach_cm = Column(Float)
    stance = Column(String(20))           # Orthodox | Southpaw | Switch
    weight_class = Column(String(50))
    url = Column(String(300), default="")    # ufcstats profile URL
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)

    # Relationships
    stats_history = relationship("FighterStats", back_populates="fighter")
    elo_history = relationship("EloRating", back_populates="fighter")
    fights_as_a = relationship("Fight", foreign_keys="Fight.fighter_a_id", back_populates="fighter_a")
    fights_as_b = relationship("Fight", foreign_keys="Fight.fighter_b_id", back_populates="fighter_b")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Fighter {self.name}>"


# ── FighterStats ──────────────────────────────────────────────────────────────

class FighterStats(Base):
    """
    A snapshot of a fighter's cumulative stats AS OF a specific date.
    This is how we avoid data leakage — we store versioned stats, not just current.
    """
    __tablename__ = "fighter_stats"
    __table_args__ = (
        UniqueConstraint("fighter_id", "as_of_date", name="uq_fighter_stats_snapshot"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    fighter_id = Column(Integer, ForeignKey("fighters.id"), nullable=False)
    as_of_date = Column(DateTime, nullable=True)   # stats valid BEFORE this date's fights

    # Career record at this point in time
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    nc = Column(Integer, default=0)               # no contests
    wins_ko_tko = Column(Integer, default=0)
    wins_sub = Column(Integer, default=0)
    wins_decision = Column(Integer, default=0)
    losses_ko_tko = Column(Integer, default=0)

    # UFC stats (per 15 min or percentage)
    slpm = Column(Float)                          # strikes landed per minute
    strike_accuracy = Column(Float)               # 0.0 – 1.0
    sapm = Column(Float)                          # strikes absorbed per minute
    strike_defense = Column(Float)                # 0.0 – 1.0
    td_avg = Column(Float)                        # takedowns per 15 min
    td_accuracy = Column(Float)
    td_defense = Column(Float)
    sub_avg = Column(Float)                       # submission attempts per 15 min

    # Derived / computed
    win_rate = Column(Float)
    finish_rate = Column(Float)
    recent_win_rate = Column(Float)               # last N fights
    win_streak = Column(Integer, default=0)       # positive = wins, negative = losses
    days_since_last_fight = Column(Integer)
    avg_opponent_elo = Column(Float)              # strength of schedule

    fighter = relationship("Fighter", back_populates="stats_history")


# ── EloRating ─────────────────────────────────────────────────────────────────

class EloRating(Base):
    """
    Elo rating for a fighter after each fight.
    Append-only — never update, always insert a new row.
    """
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fighter_id = Column(Integer, ForeignKey("fighters.id"), nullable=False)
    rating = Column(Float, nullable=False)
    after_fight_id = Column(Integer, ForeignKey("fights.id"))
    recorded_at = Column(DateTime, default=datetime.utcnow)

    fighter = relationship("Fighter", back_populates="elo_history")


# ── Fight ─────────────────────────────────────────────────────────────────────

class Fight(Base):
    """
    One row per fight. Result is nullable — used for upcoming fights too.
    """
    __tablename__ = "fights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    fighter_a_id = Column(Integer, ForeignKey("fighters.id"), nullable=False)
    fighter_b_id = Column(Integer, ForeignKey("fighters.id"), nullable=False)
    fight_date = Column(DateTime, nullable=True)
    weight_class = Column(String(50))
    is_title_fight = Column(Boolean, default=False)
    is_main_event = Column(Boolean, default=False)
    scheduled_rounds = Column(Integer, default=3)

    # Result fields — None until fight completes
    winner_id = Column(Integer, ForeignKey("fighters.id"), nullable=True)
    method = Column(String(20))              # KO_TKO | Submission | Decision | NC | Draw
    finish_round = Column(Integer)
    finish_time = Column(String(10))         # e.g. "2:47"

    # Relationships
    event = relationship("Event", back_populates="fights")
    fighter_a = relationship("Fighter", foreign_keys=[fighter_a_id], back_populates="fights_as_a")
    fighter_b = relationship("Fighter", foreign_keys=[fighter_b_id], back_populates="fights_as_b")
    prediction = relationship("Prediction", back_populates="fight", uselist=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Fight {self.fighter_a_id} vs {self.fighter_b_id} on {self.fight_date}>"


# ── Event ─────────────────────────────────────────────────────────────────────

class Event(Base):
    """UFC event (e.g. UFC 300, UFC Fight Night: Nashville)"""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    date = Column(DateTime, nullable=True)
    location = Column(String(200))
    is_ppv = Column(Boolean, default=False)

    fights = relationship("Fight", back_populates="event")


# ── Prediction ────────────────────────────────────────────────────────────────

class Prediction(Base):
    """
    Model prediction for a fight — stored BEFORE the fight happens.
    Linked to the fight so we can measure accuracy after results come in.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey("fights.id"), nullable=False, unique=True)
    model_version = Column(String(20), nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    # Win probability
    prob_fighter_a = Column(Float, nullable=False)
    prob_fighter_b = Column(Float, nullable=False)
    predicted_winner_id = Column(Integer, ForeignKey("fighters.id"))

    # Method probabilities
    prob_ko_tko = Column(Float)
    prob_submission = Column(Float)
    prob_decision = Column(Float)

    # Round predictions
    prob_under_1_5 = Column(Float)
    prob_under_2_5 = Column(Float)
    prob_under_3_5 = Column(Float)
    prob_goes_distance = Column(Float)

    # Confidence & flags
    confidence_score = Column(Float)         # 0.0 – 1.0
    upset_score = Column(Float)              # model_prob - market_implied_prob for underdog

    # Explanation (SHAP output serialized as JSON string)
    shap_explanation = Column(Text)          # JSON: {feature: shap_value, ...}
    prediction_narrative = Column(Text)      # Human-readable reasoning paragraph

    # Outcome tracking — filled in after the fight
    was_correct = Column(Boolean, nullable=True)
    method_correct = Column(Boolean, nullable=True)
    round_correct = Column(Boolean, nullable=True)

    fight = relationship("Fight", back_populates="prediction")


# ── BettingOdds ───────────────────────────────────────────────────────────────

class BettingOdds(Base):
    """
    Odds snapshot for a fight from a sportsbook.
    Multiple rows per fight — we track opening AND closing line.
    """
    __tablename__ = "betting_odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey("fights.id"), nullable=False)
    sportsbook = Column(String(50))          # e.g. DraftKings, FanDuel
    recorded_at = Column(DateTime, default=datetime.utcnow)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)

    # American odds format
    odds_fighter_a = Column(Integer)         # e.g. -220 or +180
    odds_fighter_b = Column(Integer)

    # Implied probabilities (computed at insert time)
    implied_prob_a = Column(Float)
    implied_prob_b = Column(Float)

    # Value gap (model prob - implied prob)
    value_fighter_a = Column(Float)
    value_fighter_b = Column(Float)
    reverse_line_movement = Column(Boolean, default=False)


# ── SentimentSnapshot ─────────────────────────────────────────────────────────

class SentimentSnapshot(Base):
    """NLP sentiment for a fighter around a specific fight."""
    __tablename__ = "sentiment_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fighter_id = Column(Integer, ForeignKey("fighters.id"), nullable=False)
    fight_id = Column(Integer, ForeignKey("fights.id"), nullable=False)
    source = Column(String(50))              # reddit | news | twitter
    recorded_at = Column(DateTime, default=datetime.utcnow)

    sentiment_score = Column(Float)          # -1.0 to +1.0
    mention_count = Column(Integer)
    injury_mentioned = Column(Boolean, default=False)
    short_notice_mentioned = Column(Boolean, default=False)
    camp_change_mentioned = Column(Boolean, default=False)
    raw_summary = Column(Text)               # sample of analyzed text


# ── Engine / Session factory ──────────────────────────────────────────────────

def get_engine():
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create all tables. Safe to call multiple times."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {DATABASE_URL}")


if __name__ == "__main__":
    init_db()
