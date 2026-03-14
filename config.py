"""
config.py
─────────
Central configuration for the UFC Predictor system.
All modules import from here — never read env vars directly in modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models_saved"

# Ensure dirs exist at import time
for _dir in [RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, LOGS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/ufc_predictor.db")

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL: str = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")

# ── Reddit API ────────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "ufc-predictor/1.0")

# ── Scraping ──────────────────────────────────────────────────────────────────
SCRAPE_DELAY_SECONDS: float = float(os.getenv("SCRAPE_DELAY_SECONDS", 2))
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 10))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1")
MIN_FIGHTS_REQUIRED: int = int(os.getenv("MIN_FIGHTS_REQUIRED", 5))

# How many recent fights to use for "recent form" features
RECENT_FIGHTS_WINDOW: int = 5

# Starting Elo for every fighter
ELO_BASE_RATING: float = 1500.0
ELO_K_FACTOR: float = 32.0          # how fast ratings update
ELO_FINISH_BONUS: float = 0.1       # bonus multiplier for finishes vs decisions

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "ufc_predictor.log"))

# ── Feature columns used by the model ─────────────────────────────────────────
# Defined here so features/model stay in sync
FEATURE_COLUMNS = [
    # Physical
    "reach_diff",
    "height_diff",
    "age_diff",
    # Performance diffs (fighter_A - fighter_B, all pre-fight)
    "slpm_diff",              # strikes landed per minute
    "strike_acc_diff",        # striking accuracy
    "sapm_diff",              # strikes absorbed per minute (lower is better)
    "strike_def_diff",        # striking defense
    "td_avg_diff",            # takedowns per 15 min
    "td_acc_diff",            # takedown accuracy
    "td_def_diff",            # takedown defense
    "sub_avg_diff",           # submission attempts per 15 min
    # Record / form
    "win_rate_diff",
    "finish_rate_diff",
    "recent_win_rate_diff",   # last N fights window
    "days_since_last_fight_diff",
    "win_streak_diff",
    # Elo
    "elo_diff",
    "avg_opponent_elo_diff",
    # Style matchup features (continuous scores — model learns degree of mismatch)
    "style_pressure_diff",      # forward pressure style diff
    "style_wrestling_diff",     # wrestling reliance diff
    "style_striker_diff",       # striking reliance diff
    "style_finisher_diff",      # finishing ability diff
    "grappling_defense_diff",   # grappling defense diff
    # Recent form (recency-weighted — combats favorite bias)
    "momentum_score_diff",      # weighted win streak, recent fights count more
    "recent_finish_rate_diff",  # finishing rate in last 3 fights
    # Weight class context
    "slpm_pctile_diff",         # striking volume percentile within weight class
    "td_avg_pctile_diff",       # wrestling volume percentile within weight class
    # UFC experience — debut vs veteran dynamic
    "ufc_fights_diff",          # total UFC fights difference
    "ufc_wins_diff",            # UFC wins difference
    # Fight context flags (per-fight, not per-fighter)
    "is_title_fight",           # 1 if championship fight
    # Stance mismatch — southpaw geometric advantage
    # Encoded asymmetrically: two flags, not a diff, because the effect is directional
    "is_southpaw_a_vs_orthodox_b",  # 1 = fighter A has southpaw edge over orthodox B
    "is_orthodox_a_vs_southpaw_b",  # 1 = fighter A is at southpaw disadvantage
    # Short-notice flags — derived from days_since_last_fight < 21 days
    # Already populated from existing data, no external source needed
    "fighter_a_short_notice",
    "fighter_b_short_notice",
    # Narrative signals
    "sentiment_diff",
    "fighter_a_injury_flag",
    "fighter_b_injury_flag",
]

TARGET_WINNER = "winner"          # binary: 1 = fighter_A wins, 0 = fighter_B wins
TARGET_METHOD = "method"          # KO_TKO | Submission | Decision
TARGET_ROUND = "finish_round"     # 1-5 or None
