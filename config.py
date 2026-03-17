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
    # Age curve by weight class — distance from divisional prime age
    # More meaningful than raw age: a 32-year-old flyweight is past peak,
    # a 32-year-old heavyweight is at peak
    "age_vs_peak_diff",
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
    # Elo dynamics — trajectory matters as much as current level
    "elo_trend_diff",         # Elo change over last 3 fights (rising vs declining)
    "elo_uncertainty_diff",   # how well-known is each fighter (debut vs veteran)
    "elo_vs_peak_diff",       # distance from career peak Elo (decline from prime)
    # Style matchup features (continuous scores — model learns degree of mismatch)
    "style_pressure_diff",
    "style_wrestling_diff",
    "style_striker_diff",
    "style_finisher_diff",
    "grappling_defense_diff",
    # Recent form (recency-weighted — combats favorite bias)
    "momentum_score_diff",
    "recent_finish_rate_diff",
    # Weight class context
    "slpm_pctile_diff",
    "td_avg_pctile_diff",
    # UFC experience — debut vs veteran dynamic
    "ufc_fights_diff",
    "ufc_wins_diff",
    # Durability — derived from fight-level knockdown data
    # These replace the proxy durability score with real measured data
    "durability_diff",             # composite proxy (SAPM + KO loss rate)
    "kd_absorbed_per_fight_diff",  # avg knockdowns absorbed per fight — chin/durability
    "kd_ratio_diff",               # KD landed / absorbed — KO offensive dominance
    # Rolling style windows — last 3 and last 5 fights
    # Captures style evolution; diff between career and recent = style shift
    "style_pressure_l3_diff",
    "style_wrestling_l3_diff",
    "style_striker_l3_diff",
    "style_pressure_l5_diff",
    "style_wrestling_l5_diff",
    "style_striker_l5_diff",
    # Cardio decay — from round-level data (does output hold up late?)
    "cardio_decay_diff",         # round3_output / round1_output (1.0 = no fade)
    "early_output_share_diff",   # fraction of strikes thrown in round 1 (front-loaded)
    # Strike location rates — matchup-specific targeting patterns
    "head_strike_rate_diff",     # head hunter vs body/leg worker
    "leg_strike_rate_diff",      # leg kick specialist signal
    "ground_strike_share_diff",  # striker from top vs pure standup
    # These are matchup interaction features — how does A perform vs B's specific style?
    # Formula: A_winrate_vs_wrestlers * B_style_wrestling (weighted by how much B is a wrestler)
    # Negative diff = A is MORE vulnerable to B's style than B is to A's style
    "style_vuln_wrestling_diff",   # A's wrestler vulnerability vs B's wrestling style
    "style_vuln_striker_diff",     # A's striker vulnerability vs B's striking style
    "style_vuln_pressure_diff",    # A's pressure vulnerability vs B's pressure style
    # XGBoost can learn these but needs ~50k fights; we inject them manually
    "td_success_prob_diff",   # A_td_avg * (1 - B_td_def): actual TD success likelihood
    "striking_edge_diff",     # net effective striking (output*acc - absorbed)
    "grapple_dom_diff",       # td_avg * sub_avg * (1 - opp_td_def): submission via wrestling
    "finish_threat_diff",     # finish_rate * (1 - opp_strike_defense): finishing vs chin
    "reach_strike_diff",      # reach * accuracy: reach only matters if you use it
    # Fight context flags (per-fight, not per-fighter)
    "is_title_fight",
    # Stance mismatch — southpaw geometric advantage (asymmetric, not a diff)
    "is_southpaw_a_vs_orthodox_b",
    "is_orthodox_a_vs_southpaw_b",
    # Short-notice flags
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
