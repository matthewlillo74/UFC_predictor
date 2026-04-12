"""
dashboard/app.py
─────────────────
Streamlit dashboard for UFC fight predictions.

Run with:
    streamlit run dashboard/app.py

Features:
  - Live predictions for upcoming events
  - Value bet highlighting
  - Fighter comparison tool
  - Model performance tracker
  - Historical accuracy stats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UFC Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;800&family=Barlow:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
    }

    .main { background-color: #0a0a0a; }
    .block-container { padding-top: 1.5rem; }

    h1, h2, h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .fight-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .fighter-name {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .winner-name { color: #e94560; }
    .loser-name  { color: #888; }

    .prob-big {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }

    .value-badge {
        background: #e94560;
        color: white;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .upset-badge {
        background: #f5a623;
        color: #0a0a0a;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-box {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a2a4e;
    }

    .metric-value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #e94560;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .factor-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        font-size: 0.85rem;
        border-bottom: 1px solid #1a1a2e;
    }

    .stSelectbox > div > div {
        background: #1a1a2e;
        border-color: #e94560;
    }

    .sidebar .sidebar-content { background: #0f0f1a; }

    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── AI Summary ───────────────────────────────────────────────────────────────

def generate_ai_summary(pred_dict: dict, odds_data: dict = None) -> str:
    """
    Call Claude API to generate a plain-English fight analysis.
    Only called on-demand (button press) to control costs (~$0.001/call).
    """
    import json, requests

    fa = pred_dict["fighter_a"]
    fb = pred_dict["fighter_b"]
    prob_a = pred_dict["prob_fighter_a"]
    prob_b = pred_dict["prob_fighter_b"]
    winner = pred_dict["predicted_winner"]
    methods = pred_dict.get("method_probabilities", {})
    rounds = pred_dict.get("round_probabilities", {})
    explanation = pred_dict.get("explanation", {})
    factors_a = explanation.get(f"factors_favoring_{fa}", [])
    factors_b = explanation.get(f"factors_favoring_{fb}", [])

    odds_context = ""
    if odds_data:
        raw_a = odds_data.get("odds_a", 0)
        raw_b = odds_data.get("odds_b", 0)
        fair_a = odds_data.get("fair_prob_a", 0.5)
        edge_a = prob_a - fair_a
        odds_context = f"""
Sportsbook odds: {fa} ({'+' if raw_a > 0 else ''}{raw_a}) vs {fb} ({'+' if raw_b > 0 else ''}{raw_b})
Market implied probabilities: {fa} {fair_a:.0%} / {fb} {1-fair_a:.0%}
Model vs market edge for {fa}: {edge_a:+.0%}
"""

    prompt = f"""You are an MMA analyst. Explain in 3 concise sentences why the prediction model favors {winner} in this fight. Be specific about the key factors driving the prediction. Do NOT just restate the numbers — explain what they mean about fighting style, physical advantages, and momentum.

FIGHT: {fa} vs {fb}
Model probabilities: {fa} {prob_a:.0%} / {fb} {prob_b:.0%}
Predicted winner: {winner}
Method: KO/TKO {methods.get('ko_tko',0):.0%} / Submission {methods.get('submission',0):.0%} / Decision {methods.get('decision',0):.0%}
Round: Under 2.5 {rounds.get('under_2_5',0):.0%} / Over 2.5 {rounds.get('over_2_5',0):.0%}

Key factors favoring {fa}: {', '.join(factors_a[:4]) if factors_a else 'none significant'}
Key factors favoring {fb}: {', '.join(factors_b[:4]) if factors_b else 'none significant'}
{odds_context}

Write 3 sentences max. Be direct and analytical. Mention if there is betting value or an upset angle."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15,
        )
        data = resp.json()
        if data.get("content"):
            return data["content"][0]["text"].strip()
        return f"API error: {data.get('error', {}).get('message', 'unknown')}"
    except Exception as e:
        return f"Could not generate summary: {e}"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    from src.database import init_db, get_session
    init_db()
    return get_session()


@st.cache_data(ttl=300)  # refresh every 5 min
def load_upcoming_card():
    from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights
    upcoming = get_upcoming_events()
    if not upcoming:
        return None, []
    event = upcoming[0]
    fights = get_event_fights(event["url"])
    return event, fights


@st.cache_data(ttl=60)
def load_predictions(_session):
    """Load all stored predictions with fight + fighter info."""
    from src.database import Prediction, Fight, Fighter
    rows = (
        _session.query(Prediction, Fight, Fighter, Fighter)
        .join(Fight, Prediction.fight_id == Fight.id)
        .join(Fighter, Prediction.predicted_winner_id == Fighter.id)
        .filter(Fight.winner_id == None)  # upcoming only
        .order_by(Fight.fight_date.desc())
        .all()
    )
    return rows


@st.cache_data(ttl=3600)
def load_accuracy_history(_session):
    """Load historical prediction accuracy by month."""
    from src.database import Prediction, Fight
    import sqlalchemy as sa

    rows = (
        _session.query(Prediction, Fight)
        .join(Fight)
        .filter(Prediction.was_correct != None)
        .order_by(Fight.fight_date)
        .all()
    )

    if not rows:
        return pd.DataFrame()

    data = []
    for pred, fight in rows:
        data.append({
            "date": fight.fight_date,
            "correct": int(pred.was_correct),
            "confidence": pred.confidence_score or 0.5,
        })

    df = pd.DataFrame(data)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        accuracy=("correct", "mean"),
        count=("correct", "count"),
    ).reset_index()
    monthly["accuracy_pct"] = monthly["accuracy"] * 100
    return monthly


def run_live_prediction(fighter_a_name: str, fighter_b_name: str, session):
    """Run a fresh prediction for two fighters."""
    from src.database import Fighter
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from src.ingestion.data_loader import normalize_name
    from rapidfuzz import process, fuzz

    all_fighters = session.query(Fighter).all()
    name_map = {normalize_name(f.name): f for f in all_fighters}
    norms = list(name_map.keys())

    def find(name):
        m = process.extractOne(normalize_name(name), norms,
                               scorer=fuzz.token_sort_ratio, score_cutoff=60)
        return name_map[m[0]] if m else None

    fa = find(fighter_a_name)
    fb = find(fighter_b_name)
    if not fa or not fb:
        return None

    builder = FeatureBuilder(session)
    features = builder.build_matchup_features(fa.id, fb.id, datetime.utcnow())

    predictor = UFCPredictor()
    predictor.load()
    return predictor.predict(features, fa.name, fb.name)


# ── Components ────────────────────────────────────────────────────────────────

def render_fight_card(pred_dict: dict, odds_data: dict = None, weight_class: str = ""):
    """Render a single fight prediction card."""
    fa = pred_dict["fighter_a"]
    fb = pred_dict["fighter_b"]
    prob_a = pred_dict["prob_fighter_a"]
    prob_b = pred_dict["prob_fighter_b"]
    winner = pred_dict["predicted_winner"]
    methods = pred_dict.get("method_probabilities", {})
    rounds = pred_dict.get("round_probabilities", {})
    explanation = pred_dict.get("explanation", {})

    is_value = False
    edge = 0.0
    market_prob_a = 0.5
    market_prob_b = 0.5
    odds_a_str = ""
    odds_b_str = ""
    if odds_data:
        market_prob_a = odds_data.get("fair_prob_a", 0.5)
        market_prob_b = odds_data.get("fair_prob_b", 0.5)
        edge = prob_a - market_prob_a
        is_value = abs(edge) > 0.06
        raw_a = odds_data.get("odds_a", 0)
        raw_b = odds_data.get("odds_b", 0)
        odds_a_str = f"+{raw_a}" if raw_a > 0 else str(raw_a)
        odds_b_str = f"+{raw_b}" if raw_b > 0 else str(raw_b)

    # Determine which side is the value pick
    value_side = None
    if is_value:
        value_side = fa if edge > 0 else fb

    with st.container():
        st.markdown(f"""
        <div class="fight-card">
            <div style="color:#888;font-size:0.75rem;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.5rem">
                {weight_class}
                {"&nbsp;&nbsp;<span class='value-badge'>⚡ VALUE — " + value_side + "</span>" if is_value else ""}
            </div>
        """, unsafe_allow_html=True)

        col1, col_vs, col2 = st.columns([5, 1, 5])

        with col1:
            color = "#e94560" if winner == fa else "#aaa"
            edge_a = prob_a - market_prob_a
            edge_a_str = f"<span style='color:{'#00c47d' if edge_a > 0.06 else '#e94560' if edge_a < -0.06 else '#888'};font-size:0.8rem'>{edge_a:+.0%} edge</span>" if odds_data else ""
            odds_display_a = f"<div style='color:#888;font-size:0.85rem'>{odds_a_str} &nbsp;·&nbsp; mkt {market_prob_a:.0%} &nbsp;{edge_a_str}</div>" if odds_data else ""
            st.markdown(f"""
                <div class="fighter-name" style="color:{color}">{fa}</div>
                <div class="prob-big" style="color:{color}">{prob_a:.0%}</div>
                {odds_display_a}
            """, unsafe_allow_html=True)

        with col_vs:
            st.markdown("""
                <div style="text-align:center;padding-top:0.8rem;
                            color:#555;font-family:'Barlow Condensed';
                            font-size:1.2rem;font-weight:800">VS</div>
            """, unsafe_allow_html=True)

        with col2:
            color = "#e94560" if winner == fb else "#aaa"
            edge_b = prob_b - market_prob_b
            edge_b_str = f"<span style='color:{'#00c47d' if edge_b > 0.06 else '#e94560' if edge_b < -0.06 else '#888'};font-size:0.8rem'>{edge_b:+.0%} edge</span>" if odds_data else ""
            odds_display_b = f"<div style='color:#888;font-size:0.85rem;text-align:right'>{odds_b_str} &nbsp;·&nbsp; mkt {market_prob_b:.0%} &nbsp;{edge_b_str}</div>" if odds_data else ""
            st.markdown(f"""
                <div class="fighter-name" style="color:{color};text-align:right">{fb}</div>
                <div class="prob-big" style="color:{color};text-align:right">{prob_b:.0%}</div>
                {odds_display_b}
            """, unsafe_allow_html=True)

        # Probability bar
        fig = go.Figure(go.Bar(
            x=[prob_a * 100],
            orientation='h',
            marker_color='#e94560',
            width=0.4,
        ))
        fig.add_bar(
            x=[prob_b * 100],
            orientation='h',
            marker_color='#2a2a5e',
            width=0.4,
        )
        fig.update_layout(
            barmode='stack', height=30, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100]),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                        key=f"prob_bar_{fa}_{fb}")

        # One-line summary
        winner_name = fa if winner == fa else fb
        loser_name = fb if winner == fa else fa
        winner_prob = prob_a if winner == fa else prob_b
        methods = pred_dict.get("method_probabilities", {})
        top_method = max(methods, key=methods.get) if methods else "Decision"
        method_label = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision"}.get(top_method, top_method)
        rounds_val = pred_dict.get("round_probabilities", {})
        consistency = pred_dict.get("consistency", {})

        # Only show round prediction in summary if models agree
        if consistency.get("status") == "consistent":
            ending = "early finish" if rounds_val.get("under_2_5", 0) > 0.6 else "likely decision"
            ending_str = f" — {ending}"
        elif consistency.get("status") == "warning":
            ending_str = ""  # suppress conflicting round info from summary
        else:
            ending_str = ""  # hard contradiction — omit entirely

        confidence_word = "strongly" if winner_prob > 0.65 else "narrowly" if winner_prob < 0.55 else "comfortably"
        summary = f"Model {confidence_word} favors **{winner_name}** ({winner_prob:.0%}) via {method_label}{ending_str}."
        st.markdown(f"*{summary}*")
        st.markdown("")

        # AI Narrative button — calls Claude API on demand (costs ~$0.001 per fight)
        ai_key = f"ai_{fa}_{fb}"
        col_sum, col_ai = st.columns([6, 2])
        with col_ai:
            if st.button("🤖 AI Analysis", key=f"btn_{ai_key}"):
                st.session_state[ai_key] = None  # trigger generation

        if ai_key in st.session_state:
            if st.session_state[ai_key] is None:
                with st.spinner("Generating analysis..."):
                    st.session_state[ai_key] = generate_ai_summary(pred_dict, odds_data)
            if st.session_state[ai_key]:
                st.info(st.session_state[ai_key])

        # Method + round breakdown
        col_m, col_r = st.columns(2)
        with col_m:
            st.markdown("**Method**")
            for label, val in [("KO/TKO", methods.get("ko_tko", 0)),
                               ("Submission", methods.get("submission", 0)),
                               ("Decision", methods.get("decision", 0))]:
                st.markdown(f"""
                    <div class="factor-row">
                        <span style="color:#aaa">{label}</span>
                        <span style="color:#e94560;font-weight:600">{val:.0%}</span>
                    </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.markdown("**Rounds**")
            rounds = pred_dict.get("round_probabilities", {})
            # Label switches to O/U 3.5 for 5-round (title) fights
            if "under_3_5" in rounds:
                round_pairs = [("Under 3.5", rounds.get("under_3_5", 0)),
                               ("Over 3.5",  rounds.get("over_3_5",  0))]
            else:
                round_pairs = [("Under 2.5", rounds.get("under_2_5", 0)),
                               ("Over 2.5",  rounds.get("over_2_5",  0))]
            for label, val in round_pairs:
                st.markdown(f"""
                    <div class="factor-row">
                        <span style="color:#aaa">{label}</span>
                        <span style="color:#e94560;font-weight:600">{val:.0%}</span>
                    </div>
                """, unsafe_allow_html=True)

        # Consistency indicator — shown only when models disagree
        consistency = pred_dict.get("consistency", {})
        c_status  = consistency.get("status", "consistent")
        c_message = consistency.get("message", "")
        if c_status == "contradiction" and c_message:
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;border-left:3px solid #e94560;"
                f"background:#1a0a0a;border-radius:4px;font-size:0.8rem;color:#e94560'>"
                f"⚠️ Model conflict: {c_message}</div>",
                unsafe_allow_html=True
            )
        elif c_status == "warning" and c_message:
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;border-left:3px solid #f5a623;"
                f"background:#1a1400;border-radius:4px;font-size:0.8rem;color:#f5a623'>"
                f"⚡ {c_message}</div>",
                unsafe_allow_html=True
            )

        # Key factors expander
        with st.expander("Key factors"):
            col_fa, col_fb = st.columns(2)
            factors_a = explanation.get(f"factors_favoring_{fa}", [])
            factors_b = explanation.get(f"factors_favoring_{fb}", [])
            with col_fa:
                st.markdown(f"**{fa}**")
                for f_ in factors_a[:5]:
                    st.markdown(f"+ {f_}")
            with col_fb:
                st.markdown(f"**{fb}**")
                for f_ in factors_b[:5]:
                    st.markdown(f"+ {f_}")

            if odds_data and abs(edge) > 0.01:
                st.divider()
                market_p = odds_data.get("fair_prob_a", 0.5)
                st.markdown(f"**Betting edge ({fa}):** Model `{prob_a:.1%}` vs Market `{market_p:.1%}` → `{edge:+.1%}`")

        st.markdown("</div>", unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_upcoming_event(session, odds_lookup: dict):
    st.markdown("# 🥊 Upcoming Event")

    event_data, fights_raw = load_upcoming_card()
    if not event_data:
        st.error("No upcoming events found on ufcstats.com")
        return

    event_name = event_data.get("name", "Unknown Event")
    event_date = event_data.get("date")

    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown(f"## {event_name}")
    with col_date:
        if event_date:
            st.markdown(f"**{event_date.strftime('%B %d, %Y')}**")

    if not fights_raw:
        st.warning("Could not load fight card")
        return

    # Odds passed in from session_state — no API call here
    live_odds = odds_lookup

    # Run predictions
    from src.ingestion.data_loader import get_or_create_fighter, normalize_name
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from rapidfuzz import fuzz

    try:
        predictor = UFCPredictor()
        predictor.load()
    except Exception as e:
        st.error(f"Model not loaded. Run `python scripts/train_model.py` first.\n\n{e}")
        return

    builder = FeatureBuilder(session)
    fight_date = event_date or datetime.utcnow()

    predictions = []
    with st.spinner("Running predictions..."):
        for f in fights_raw:
            try:
                fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
                fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))
                if fa is None or fb is None:
                    # Fighter not in DB and DB is read-only (cloud) — skip silently
                    logger.warning(f"Skipping unknown fighter(s): {f.get('fighter_a_name')} vs {f.get('fighter_b_name')}")
                    continue
                features = builder.build_matchup_features(fa.id, fb.id, fight_date)
                pred = predictor.predict(features, fa.name, fb.name)

                # Match odds — try fuzzy key matching
                fight_odds = None
                fa_norm = normalize_name(fa.name)
                fb_norm = normalize_name(fb.name)
                key = fa_norm + "|" + fb_norm
                if key in live_odds:
                    fight_odds = live_odds[key]
                else:
                    # Fuzzy fallback
                    for ok, ov in live_odds.items():
                        parts = ok.split("|")
                        if len(parts) == 2:
                            if fuzz.token_sort_ratio(fa_norm, parts[0]) > 75 and fuzz.token_sort_ratio(fb_norm, parts[1]) > 75:
                                fight_odds = ov
                                break

                predictions.append((pred, fight_odds, f.get("weight_class", "")))
            except Exception as e:
                st.warning(f"Could not predict {f.get('fighter_a_name')} vs {f.get('fighter_b_name')}: {e}")

    if not predictions:
        st.error("No predictions generated")
        return

    # Value picks summary banner
    value_picks = [(p, o) for p, o, w in predictions
                   if o and abs(p["prob_fighter_a"] - o.get("fair_prob_a", 0.5)) > 0.07]
    if value_picks:
        st.markdown("### ⚡ Value Picks This Card")
        vcols = st.columns(min(len(value_picks), 4))
        for i, (vp, vo) in enumerate(value_picks[:4]):
            market_p = vo.get("fair_prob_a", 0.5)
            edge = vp["prob_fighter_a"] - market_p
            fighter = vp["fighter_a"] if edge > 0 else vp["fighter_b"]
            model_p = vp["prob_fighter_a"] if edge > 0 else vp["prob_fighter_b"]
            with vcols[i]:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{model_p:.0%}</div>
                        <div style="font-family:'Barlow Condensed';font-weight:700;
                                    font-size:1rem;color:#fff;margin:4px 0">{fighter}</div>
                        <div class="metric-label">+{abs(edge):.1%} edge</div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("")

    # Fight cards
    for pred, odds, wc in predictions:
        render_fight_card(pred, odds, wc)


def page_fighter_compare(session):
    st.markdown("# ⚔️ Fighter Matchup Tool")
    st.markdown("Simulate any matchup — current or hypothetical")

    from src.database import Fighter as FighterModel
    from src.ingestion.data_loader import normalize_name

    all_fighters = session.query(FighterModel).order_by(FighterModel.name).all()
    fighter_names = [f.name for f in all_fighters]

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        fa_name = st.selectbox("Fighter A", fighter_names,
                               index=fighter_names.index("Islam Makhachev") if "Islam Makhachev" in fighter_names else 0,
                               key="fa")
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-family:\"Barlow Condensed\";font-size:2rem;font-weight:800;color:#e94560'>VS</div>", unsafe_allow_html=True)
    with col2:
        fb_name = st.selectbox("Fighter B", fighter_names,
                               index=fighter_names.index("Charles Oliveira") if "Charles Oliveira" in fighter_names else 1,
                               key="fb")

    if st.button("🔮 Predict", type="primary", width="stretch"):
        if fa_name == fb_name:
            st.error("Select two different fighters")
            return

        with st.spinner("Analyzing matchup..."):
            pred = run_live_prediction(fa_name, fb_name, session)

        if pred:
            st.divider()
            render_fight_card(pred, weight_class="Simulated Matchup")

            # Elo comparison
            from src.database import EloRating, Fighter as FM
            fa_obj = session.query(FM).filter_by(name=fa_name).first()
            fb_obj = session.query(FM).filter_by(name=fb_name).first()

            if fa_obj and fb_obj:
                def get_elo(fid):
                    r = session.query(EloRating).filter_by(fighter_id=fid).order_by(EloRating.recorded_at.desc()).first()
                    return r.rating if r else 1500

                elo_a = get_elo(fa_obj.id)
                elo_b = get_elo(fb_obj.id)

                st.markdown("### Elo Rating Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(fa_name, f"{elo_a:.0f}", delta=f"{elo_a - elo_b:+.0f} vs opponent")
                with col2:
                    st.metric(fb_name, f"{elo_b:.0f}", delta=f"{elo_b - elo_a:+.0f} vs opponent")
        else:
            st.error("Could not find one or both fighters in the database")


def page_performance(session):
    st.markdown("# 📊 Model Performance")

    from src.evaluation.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker(session)
    summary = tracker.get_summary()

    if "error" in summary:
        st.info("No resolved predictions yet. After an event completes, run:\n```\npython scripts/run_pipeline.py --post-event\n```")

        # Show what we do have — training set stats
        st.markdown("### Training Set Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box"><div class="metric-value">60.5%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><div class="metric-value">1,286</div><div class="metric-label">Test Fights</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><div class="metric-value">+10.6%</div><div class="metric-label">vs Baseline</div></div>', unsafe_allow_html=True)

        st.markdown("### Feature Importance")
        features = {
            "Elo Diff": 0.1639,
            "Age Diff": 0.1601,
            "Avg Opp Elo Diff": 0.1133,
            "Reach Diff": 0.1091,
            "Win Rate Diff": 0.1007,
            "Finish Rate Diff": 0.0924,
            "Win Streak Diff": 0.0882,
            "Days Since Fight Diff": 0.0865,
            "Height Diff": 0.0856,
        }
        fig = px.bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation='h',
            color=list(features.values()),
            color_continuous_scale=[[0, '#2a2a5e'], [1, '#e94560']],
        )
        fig.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
            font_color='#aaa', showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=320,
        )
        fig.update_xaxes(showgrid=False, tickformat=".0%")
        st.plotly_chart(fig, width="stretch", key="accuracy_calibration_chart")

    # Live stats
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Winner Accuracy", f"{summary['winner_accuracy']:.1%}"),
        ("Total Predictions", str(summary["total_predictions"])),
        ("Avg Confidence", f"{summary['avg_confidence']:.1%}"),
        ("Method Accuracy", f"{summary.get('method_accuracy', 0):.1%}" if summary.get("method_accuracy") else "N/A"),
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    # Monthly accuracy chart
    monthly_df = load_accuracy_history(session)
    if not monthly_df.empty:
        st.markdown("### Monthly Accuracy")
        fig = px.line(
            monthly_df, x="month", y="accuracy_pct",
            markers=True,
            color_discrete_sequence=["#e94560"],
        )
        fig.add_hline(y=50, line_dash="dash", line_color="#555", annotation_text="Baseline (50%)")
        fig.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
            font_color='#aaa', yaxis_title="Accuracy %",
            xaxis_title="", margin=dict(l=0, r=0, t=20, b=0), height=300,
        )
        st.plotly_chart(fig, width="stretch", key="monthly_accuracy_chart")
    st.markdown("# 🏆 Elo Leaderboard")

    from src.database import Fighter as FighterModel, EloRating
    import sqlalchemy as sa

    weight_classes = [
        "All", "Heavyweight", "Light Heavyweight", "Middleweight",
        "Welterweight", "Lightweight", "Featherweight", "Bantamweight",
        "Flyweight", "Women's Featherweight", "Women's Bantamweight",
        "Women's Flyweight", "Women's Strawweight",
    ]
    wc = st.selectbox("Weight Class", weight_classes)

    # Get top fighters by Elo
    fighters = session.query(FighterModel).all()
    rows = []
    for f in fighters:
        latest_elo = (
            session.query(EloRating)
            .filter_by(fighter_id=f.id)
            .order_by(EloRating.recorded_at.desc())
            .first()
        )
        if latest_elo:
            rows.append({
                "Fighter": f.name,
                "Elo": round(latest_elo.rating),
                "Wins": f.wins or 0,
                "Losses": f.losses or 0,
            })

    df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    df.index += 1
    df["W-L"] = df["Wins"].astype(str) + "-" + df["Losses"].astype(str)

    st.dataframe(
        df[["Fighter", "Elo", "W-L"]].head(50),
        width="stretch",
        column_config={
            "Elo": st.column_config.NumberColumn(format="%d"),
        }
    )


# ── Sidebar + routing ─────────────────────────────────────────────────────────

def page_props(session, odds_lookup: dict):
    """
    Props tab — method and round O/U predictions with market edge detection.

    Shows for every fight:
      - Method props: KO/TKO, Submission, Decision with model prob vs market line
      - Round O/U: model prob vs market line
      - Edge flagged when model prob exceeds market implied prob by threshold
      - Backtest accuracy shown so user knows how much to trust each prop type
    """
    st.markdown("# 🎯 Prop Bets")
    st.markdown("Method and round predictions with market edge — verify lines before betting")

    from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights
    from src.ingestion.data_loader import get_or_create_fighter, normalize_name
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from rapidfuzz import fuzz

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Model not loaded. Run `python scripts/train_model.py` first.\n\n{e}")
        return

    # ── Load card ─────────────────────────────────────────────────────────────
    upcoming = get_upcoming_events()
    if not upcoming:
        st.error("No upcoming events found")
        return
    event = upcoming[0]
    fights_raw = get_event_fights(event["url"])
    if not fights_raw:
        st.error("No fights found for upcoming event")
        return

    builder = FeatureBuilder(session)
    fight_date = event.get("date") or datetime.utcnow()

    # ── Calibration warning banner ────────────────────────────────────────────
    st.info(
        "**Backtest accuracy (holdout test set):** "
        "Method ~55% | Round O/U ~40-50% | "
        "Live data still limited (4 events). "
        "Use props as supplementary signals, not primary bets."
    )

    # ── Prop odds reference (simulated — replace with real API when available) ─
    # These are approximate average UFC prop lines for reference.
    # Real prop odds are not available from The Odds API free tier.
    PROP_ODDS_REF = {
        "ko_tko":     {"label": "KO/TKO",     "approx_odds": -115, "mkt_prob": 0.535},
        "submission": {"label": "Submission",  "approx_odds": +230, "mkt_prob": 0.303},
        "decision":   {"label": "Decision",    "approx_odds": -140, "mkt_prob": 0.583},
        "under_2_5":  {"label": "Under 2.5",  "approx_odds": -110, "mkt_prob": 0.524},
        "over_2_5":   {"label": "Over 2.5",   "approx_odds": -110, "mkt_prob": 0.524},
        "under_3_5":  {"label": "Under 3.5",  "approx_odds": -130, "mkt_prob": 0.565},
        "over_3_5":   {"label": "Over 3.5",   "approx_odds": -110, "mkt_prob": 0.524},
    }

    def edge_color(edge: float) -> str:
        if edge > 0.08:   return "#00c47d"
        if edge > 0.04:   return "#7ecb8e"
        if edge < -0.04:  return "#e94560"
        return "#888"

    def american_fmt(odds: int) -> str:
        return f"+{odds}" if odds > 0 else str(odds)

    st.markdown(f"### {event['name']}")
    st.markdown("")

    # ── Generate predictions ──────────────────────────────────────────────────
    all_props = []
    with st.spinner("Generating prop predictions..."):
        for f in fights_raw:
            try:
                fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
                fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))
                if fa is None or fb is None:
                    continue
                features = builder.build_matchup_features(fa.id, fb.id, fight_date)
                pred = predictor.predict(features, fa.name, fb.name)

                # Match moneyline odds for context
                fa_norm = normalize_name(fa.name)
                fb_norm = normalize_name(fb.name)
                fight_odds = None
                key = fa_norm + "|" + fb_norm
                if key in odds_lookup:
                    fight_odds = odds_lookup[key]
                else:
                    for ok, ov in odds_lookup.items():
                        parts = ok.split("|")
                        if len(parts) == 2:
                            if fuzz.token_sort_ratio(fa_norm, parts[0]) > 75 and \
                               fuzz.token_sort_ratio(fb_norm, parts[1]) > 75:
                                fight_odds = ov
                                break

                all_props.append({
                    "pred": pred,
                    "odds": fight_odds,
                    "weight_class": f.get("weight_class", ""),
                    "is_title": f.get("is_title_fight", False),
                })
            except Exception:
                pass

    if not all_props:
        st.error("No predictions generated")
        return

    # ── Render each fight ─────────────────────────────────────────────────────
    MIN_METHOD_EDGE = 0.08   # flag method prop if model > market by this much
    MIN_ROUND_EDGE  = 0.06   # flag round prop if model > market by this much

    for item in all_props:
        pred       = item["pred"]
        fight_odds = item["odds"]
        wc         = item["weight_class"]
        is_title   = item["is_title"]
        fa         = pred["fighter_a"]
        fb         = pred["fighter_b"]
        methods    = pred.get("method_probabilities", {})
        rounds     = pred.get("round_probabilities", {})

        ko_prob  = methods.get("ko_tko", 0)
        sub_prob = methods.get("submission", 0)
        dec_prob = methods.get("decision", 0)

        if is_title or "under_3_5" in rounds:
            round_key   = "under_3_5"
            round_label = "O/U 3.5"
            under_prob  = rounds.get("under_3_5", 0)
            over_prob   = rounds.get("over_3_5", 0)
        else:
            round_key   = "under_2_5"
            round_label = "O/U 2.5"
            under_prob  = rounds.get("under_2_5", 0)
            over_prob   = rounds.get("over_2_5", 0)

        # Compute edges vs simulated market
        ko_edge    = ko_prob  - PROP_ODDS_REF["ko_tko"]["mkt_prob"]
        sub_edge   = sub_prob - PROP_ODDS_REF["submission"]["mkt_prob"]
        dec_edge   = dec_prob - PROP_ODDS_REF["decision"]["mkt_prob"]
        under_edge = under_prob - PROP_ODDS_REF[round_key]["mkt_prob"]
        over_edge  = over_prob  - PROP_ODDS_REF[round_key.replace("under", "over")]["mkt_prob"]

        # Flag if any prop has real edge
        has_method_edge = max(abs(ko_edge), abs(sub_edge), abs(dec_edge)) > MIN_METHOD_EDGE
        has_round_edge  = max(abs(under_edge), abs(over_edge)) > MIN_ROUND_EDGE
        has_any_edge    = has_method_edge or has_round_edge

        # Fight header
        badge = " &nbsp;<span style='background:#e94560;color:#fff;font-size:0.7rem;padding:2px 7px;border-radius:4px'>⚡ PROP VALUE</span>" if has_any_edge else ""
        wc_display = f"{wc}{badge}" if wc else f"Fight{badge}"
        st.markdown(
            f"<div style='color:#888;font-size:0.75rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-top:1.2rem;margin-bottom:0.4rem'>{wc_display}</div>",
            unsafe_allow_html=True,
        )

        winner     = pred["predicted_winner"]
        confidence = pred["confidence"]
        col_title, col_conf = st.columns([4, 1])
        with col_title:
            st.markdown(f"**{fa}** vs **{fb}**")
        with col_conf:
            st.markdown(
                f"<div style='text-align:right;color:#888;font-size:0.85rem'>"
                f"Winner: {winner} ({confidence:.0%})</div>",
                unsafe_allow_html=True,
            )

        # ── Method props ──────────────────────────────────────────────────────
        col_ko, col_sub, col_dec, col_round = st.columns(4)

        def render_prop_card(col, label, model_prob, edge, approx_odds, is_flagged):
            with col:
                color = edge_color(edge)
                flag  = "⚡ " if is_flagged else ""
                st.markdown(
                    f"""<div style='background:#1a1a2e;border-radius:8px;padding:0.75rem;
                                   border:1px solid {"#e94560" if is_flagged else "#333"};
                                   text-align:center'>
                        <div style='color:#aaa;font-size:0.7rem;text-transform:uppercase;
                                    letter-spacing:0.05em;margin-bottom:4px'>{flag}{label}</div>
                        <div style='font-size:1.4rem;font-weight:700;color:#fff'>{model_prob:.0%}</div>
                        <div style='font-size:0.75rem;color:{color};margin-top:2px'>
                            {edge:+.0%} vs mkt ({american_fmt(approx_odds)})
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        render_prop_card(col_ko,  "KO/TKO",     ko_prob,    ko_edge,    PROP_ODDS_REF["ko_tko"]["approx_odds"],    abs(ko_edge)    > MIN_METHOD_EDGE)
        render_prop_card(col_sub, "Submission",  sub_prob,   sub_edge,   PROP_ODDS_REF["submission"]["approx_odds"], abs(sub_edge)   > MIN_METHOD_EDGE)
        render_prop_card(col_dec, "Decision",    dec_prob,   dec_edge,   PROP_ODDS_REF["decision"]["approx_odds"],  abs(dec_edge)   > MIN_METHOD_EDGE)

        # Round O/U
        under_flagged = abs(under_edge) > MIN_ROUND_EDGE
        over_flagged  = abs(over_edge)  > MIN_ROUND_EDGE
        with col_round:
            u_color = edge_color(under_edge)
            o_color = edge_color(over_edge)
            u_flag  = "⚡ " if under_flagged else ""
            o_flag  = "⚡ " if over_flagged  else ""
            st.markdown(
                f"""<div style='background:#1a1a2e;border-radius:8px;padding:0.75rem;
                               border:1px solid {"#e94560" if (under_flagged or over_flagged) else "#333"}'>
                    <div style='color:#aaa;font-size:0.7rem;text-transform:uppercase;
                                letter-spacing:0.05em;margin-bottom:6px;text-align:center'>{round_label}</div>
                    <div style='display:flex;justify-content:space-between;font-size:0.85rem'>
                        <div>
                            <span style='color:#aaa'>{u_flag}Under</span>
                            <span style='color:#fff;font-weight:700;margin-left:4px'>{under_prob:.0%}</span>
                            <span style='color:{u_color};font-size:0.7rem;margin-left:3px'>{under_edge:+.0%}</span>
                        </div>
                        <div>
                            <span style='color:#aaa'>{o_flag}Over</span>
                            <span style='color:#fff;font-weight:700;margin-left:4px'>{over_prob:.0%}</span>
                            <span style='color:{o_color};font-size:0.7rem;margin-left:3px'>{over_edge:+.0%}</span>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Consistency warning if method/round disagree
        consistency = pred.get("consistency", {})
        if consistency.get("status") in ("warning", "contradiction"):
            st.caption(f"⚠️ {consistency['message']}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='color:#666;font-size:0.75rem'>"
        "Market odds shown are approximate averages — actual sportsbook lines vary. "
        "Edge calculation uses simulated baseline. Always verify real odds before betting. "
        "Method accuracy ~55% and round O/U ~45% on test set. "
        "Run <code>python scripts/backtest_props.py</code> for full historical analysis."
        "</div>",
        unsafe_allow_html=True,
    )


def page_value_bets(session, parsed_odds: list, odds_lookup: dict):
    st.markdown("# ⚡ Value Bets & Upsets")
    st.markdown("Fights where the model meaningfully disagrees with the market")

    from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights
    from src.ingestion.data_loader import get_or_create_fighter, normalize_name
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from src.models.calibrate import find_value_bets, find_upset_candidates
    from rapidfuzz import fuzz

    # Odds passed in — no API call
    if not parsed_odds:
        st.warning("No odds available. Add ODDS_API_KEY to .env")
        return

    # Build odds map
    odds_map = {}
    for o in parsed_odds:
        odds_map[normalize_name(o["fighter_a"]) + "|" + normalize_name(o["fighter_b"])] = o

    upcoming = get_upcoming_events()
    if not upcoming:
        st.error("No upcoming events")
        return

    event = upcoming[0]
    fights_raw = get_event_fights(event["url"])

    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Model not loaded: {e}")
        return

    builder = FeatureBuilder(session)
    fight_date = event.get("date") or datetime.utcnow()

    all_preds = []
    for f in fights_raw:
        try:
            fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
            fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))
            if fa is None or fb is None:
                continue
            features = builder.build_matchup_features(fa.id, fb.id, fight_date)
            pred = predictor.predict(features, fa.name, fb.name)
            pred["weight_class"] = f.get("weight_class", "")

            # Match odds
            fa_norm = normalize_name(fa.name)
            fb_norm = normalize_name(fb.name)
            fight_odds = None
            key = fa_norm + "|" + fb_norm
            if key in odds_map:
                fight_odds = odds_map[key]
            else:
                for ok, ov in odds_map.items():
                    parts = ok.split("|")
                    if len(parts) == 2 and fuzz.token_sort_ratio(fa_norm, parts[0]) > 75 and fuzz.token_sort_ratio(fb_norm, parts[1]) > 75:
                        fight_odds = ov
                        break
                if not fight_odds:
                    for ok, ov in odds_map.items():
                        parts = ok.split("|")
                        if len(parts) == 2 and fuzz.token_sort_ratio(fb_norm, parts[0]) > 75 and fuzz.token_sort_ratio(fa_norm, parts[1]) > 75:
                            fight_odds = {"odds_a": ov["odds_b"], "odds_b": ov["odds_a"],
                                         "fair_prob_a": ov["fair_prob_b"], "fair_prob_b": ov["fair_prob_a"],
                                         "fighter_a": fb.name, "fighter_b": fa.name}
                            break

            pred["odds_data"] = fight_odds
            all_preds.append(pred)
        except Exception as e:
            pass

    # ── Value Bets ────────────────────────────────────────────────────────────
    value_bets = find_value_bets(all_preds, min_edge=0.05)
    upsets = find_upset_candidates(all_preds, min_upset_score=0.08)

    st.markdown(f"### {event['name']}")
    st.markdown("")

    if value_bets:
        st.markdown("## 💰 Value Bets")
        st.markdown("*Model probability exceeds market by 5%+*")
        st.markdown("")

        for bet in value_bets:
            odds_str = f"+{bet['odds']}" if bet['odds'] and bet['odds'] > 0 else str(bet.get('odds', 'N/A'))
            underdog_tag = " 🐶 UNDERDOG" if bet["is_underdog"] else ""
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            with col1:
                st.markdown(f"**{bet['fighter']}**{underdog_tag}  \n*vs {bet['opponent']}*")
            with col2:
                st.metric("Model", f"{bet['model_prob']:.0%}")
            with col3:
                st.metric("Market", f"{bet['market_prob']:.0%}")
            with col4:
                st.metric("Edge", f"+{bet['edge']:.0%}", delta=f"+{bet['edge']:.0%}")
            with col5:
                st.metric("Kelly", f"{bet['kelly_pct']:.1f}%",
                          help="Kelly Criterion: suggested % of bankroll. Use half-Kelly in practice.")
            st.markdown(f"Odds: `{odds_str}` · {bet['weight_class']}")
            st.divider()
    else:
        st.info("No strong value bets detected on this card (edge < 5%)")

    # ── Upset Candidates ──────────────────────────────────────────────────────
    if upsets:
        st.markdown("## 🐶 Upset Candidates")
        st.markdown("*Model likes the underdog significantly more than the market does*")
        st.markdown("")

        for u in upsets:
            odds_str = f"+{u['underdog_odds']}" if u['underdog_odds'] and u['underdog_odds'] > 0 else str(u.get('underdog_odds', 'N/A'))
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**{u['underdog']}** to upset **{u['favorite']}**")
                st.caption(u['weight_class'])
            with col2:
                st.metric("Model says", f"{u['model_prob']:.0%}")
            with col3:
                st.metric("Market says", f"{u['market_prob']:.0%}")
            with col4:
                st.metric("Disagreement", f"+{u['upset_score']:.0%}")
            st.markdown(f"Underdog odds: `{odds_str}`")
            st.divider()
    else:
        st.info("No significant upset candidates detected")

    # ── All fights odds table ─────────────────────────────────────────────────
    st.markdown("## 📋 Full Card Odds vs Model")
    rows = []
    for pred in all_preds:
        o = pred.get("odds_data")
        if not o:
            continue
        edge_a = pred["prob_fighter_a"] - o.get("fair_prob_a", 0.5)
        rows.append({
            "Fighter A": pred["fighter_a"],
            "Model A": f"{pred['prob_fighter_a']:.0%}",
            "Market A": f"{o.get('fair_prob_a', 0.5):.0%}",
            "Odds A": f"+{o['odds_a']}" if o.get('odds_a', 0) > 0 else str(o.get('odds_a', '')),
            "Edge A": f"{edge_a:+.0%}",
            "Fighter B": pred["fighter_b"],
            "Model B": f"{pred['prob_fighter_b']:.0%}",
            "Odds B": f"+{o['odds_b']}" if o.get('odds_b', 0) > 0 else str(o.get('odds_b', '')),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")



def page_parlays(session, parsed_odds: list, odds_lookup: dict):
    st.markdown("# 🎰 Parlay Builder")
    st.markdown("Data-backed parlays built from model edge — not just favorites")

    from src.ingestion.fight_scraper import get_upcoming_events, get_event_fights
    from src.ingestion.data_loader import get_or_create_fighter, normalize_name
    from src.features.feature_builder import FeatureBuilder
    from src.models.predict import UFCPredictor
    from src.betting.parlay_builder import build_parlays, build_candidate_legs
    from rapidfuzz import fuzz

    # Odds passed in — no API call
    # Build odds map
    odds_map = {}
    for o in parsed_odds:
        odds_map[normalize_name(o["fighter_a"]) + "|" + normalize_name(o["fighter_b"])] = o

    upcoming = get_upcoming_events()
    if not upcoming:
        st.error("No upcoming events found")
        return

    event = upcoming[0]
    fights_raw = get_event_fights(event["url"])

    predictor = UFCPredictor()
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Model not loaded: {e}")
        return

    builder = FeatureBuilder(session)
    fight_date = event.get("date") or datetime.utcnow()

    all_preds = []
    with st.spinner("Building predictions..."):
        for f in fights_raw:
            try:
                fa = get_or_create_fighter(session, f["fighter_a_name"], f.get("fighter_a_url", ""))
                fb = get_or_create_fighter(session, f["fighter_b_name"], f.get("fighter_b_url", ""))
                if fa is None or fb is None:
                    continue
                features = builder.build_matchup_features(fa.id, fb.id, fight_date)
                pred = predictor.predict(features, fa.name, fb.name)
                pred["weight_class"] = f.get("weight_class", "")

                # Match odds
                fa_norm = normalize_name(fa.name)
                fb_norm = normalize_name(fb.name)
                fight_odds = None
                key = fa_norm + "|" + fb_norm
                if key in odds_map:
                    fight_odds = odds_map[key]
                else:
                    for ok, ov in odds_map.items():
                        parts = ok.split("|")
                        if len(parts) == 2:
                            if fuzz.token_sort_ratio(fa_norm, parts[0]) > 75 and fuzz.token_sort_ratio(fb_norm, parts[1]) > 75:
                                fight_odds = ov
                                break
                            if fuzz.token_sort_ratio(fb_norm, parts[0]) > 75 and fuzz.token_sort_ratio(fa_norm, parts[1]) > 75:
                                fight_odds = {
                                    **ov,
                                    "odds_a": ov["odds_b"], "odds_b": ov["odds_a"],
                                    "fair_prob_a": ov["fair_prob_b"], "fair_prob_b": ov["fair_prob_a"],
                                }
                                break
                pred["odds_data"] = fight_odds
                all_preds.append(pred)
            except Exception:
                pass

    parlays = build_parlays(all_preds)
    legs = build_candidate_legs(all_preds)

    if not parlays and not legs:
        st.warning("Not enough data to build parlays. Make sure odds are loaded.")
        return

    st.markdown(f"### {event['name']}")
    st.markdown(f"*{len(legs)} viable legs identified across {len(all_preds)} fights*")
    st.markdown("")

    # ── Leg quality table ─────────────────────────────────────────────────────
    with st.expander("📋 All viable legs (ranked by model score)", expanded=False):
        rows = []
        for leg in legs:
            odds_str = f"+{leg.american_odds}" if leg.american_odds and leg.american_odds > 0 else str(leg.american_odds or "N/A")
            rows.append({
                "Fighter": ("🐶 " if leg.is_underdog else "") + leg.fighter,
                "vs": leg.opponent,
                "Model": f"{leg.model_prob:.0%}",
                "Market": f"{leg.market_prob:.0%}",
                "Edge": f"{leg.edge:+.0%}",
                "Odds": odds_str,
                "Score": f"{leg.leg_score:.3f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")

    # ── Render each tier ──────────────────────────────────────────────────────
    tier_config = [
        ("safe",  "🟢 Safe Parlay",  "High confidence, positive edge on every leg. Lowest risk tier."),
        ("value", "🔵 Value Parlay", "4-5 legs chosen to maximize expected value. Best bang for buck."),
        ("shot",  "🟡 Shot Parlay",  "Strong favorites + model-backed underdog(s). Enhanced payout."),
        ("super", "🔴 Super Parlay", "Full card parlay — lottery ticket. One for fun."),
    ]

    for tier_key, tier_label, tier_desc in tier_config:
        tier_parlays = parlays.get(tier_key, [])
        if not tier_parlays:
            continue

        st.markdown(f"## {tier_label}")
        st.caption(tier_desc)

        for parlay in tier_parlays:
            ev_color = "#00c47d" if parlay.expected_value > 0 else "#e94560"
            odds_str = f"+{parlay.true_american_odds}" if parlay.true_american_odds > 0 else str(parlay.true_american_odds)

            # Header metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Legs", len(parlay.legs))
            with c2:
                st.metric("Model prob", f"{parlay.combined_model_prob:.1%}")
            with c3:
                st.metric("Payout", odds_str)
            with c4:
                ev_label = f"{parlay.expected_value:+.1%} EV"
                st.metric("Expected value", ev_label,
                          help="Positive EV = profitable long run. Based on model probabilities vs actual payout odds.")

            # Legs
            for i, leg in enumerate(parlay.legs, 1):
                leg_odds = f"+{leg.american_odds}" if leg.american_odds and leg.american_odds > 0 else str(leg.american_odds or "N/A")
                underdog_icon = "🐶" if leg.is_underdog else "✅"
                edge_color = "#00c47d" if leg.edge > 0.05 else "#888"
                st.markdown(
                    f"{underdog_icon} **Leg {i}: {leg.fighter}** vs {leg.opponent} &nbsp;·&nbsp; "
                    f"Model `{leg.model_prob:.0%}` &nbsp;·&nbsp; "
                    f"Odds `{leg_odds}` &nbsp;·&nbsp; "
                    f"<span style='color:{edge_color}'>{leg.edge:+.0%} edge</span>",
                    unsafe_allow_html=True,
                )

            # EV explanation
            if parlay.expected_value > 0:
                st.success(f"✅ Positive EV: for every $100 bet, model expects +${parlay.expected_value*100:.0f} long-run profit")
            else:
                st.info(f"ℹ️ Negative EV ({parlay.expected_value:+.1%}) — model edge partially overcomes vig but doesn't beat it fully")

            st.markdown(f"*Market EV on this parlay: {parlay.market_ev:+.1%} (bookmaker's edge)*")
            st.divider()

    # ── Parlay math explainer ─────────────────────────────────────────────────
    with st.expander("📚 How parlay EV is calculated", expanded=False):
        st.markdown("""
**Expected Value (EV)** tells you whether a bet is mathematically profitable over many repetitions.

```
EV = (Model Probability × Parlay Payout) - 1
```

- **Positive EV** (+5%) → for every $100 bet, you expect to profit $5 long-run
- **Negative EV** (-8%) → for every $100 bet, you expect to lose $8 long-run
- Sportsbooks build ~5-8% vig into every leg, which compounds across a parlay

**Why these parlays are different from random parlays:**
- Every leg is chosen where the model has *higher* probability than the market implies
- This means each leg partially or fully overcomes the vig
- Stringing 4-5 legs with +5% edge each can flip the whole parlay to positive EV

**Kelly Criterion for parlays:** Size your parlay bets at 1-3% of bankroll max.
The variance is extreme even with positive EV.
        """)


@st.cache_data(ttl=1800)
def load_odds_once(api_key: str = ""):
    """
    Fetch odds from the API exactly once per 30-min window (cached).
    All tabs read from session_state — never call fetch_mma_odds() directly in a tab.
    api_key is included in cache key so different users get independent caches.
    """
    try:
        from src.ingestion.odds_scraper import fetch_mma_odds, parse_odds_response
        from src.ingestion.data_loader import normalize_name
        raw = fetch_mma_odds(api_key=api_key or None)
        if not raw:
            return {}, [], None
        parsed = parse_odds_response(raw)
        odds_lookup = {}
        for o in parsed:
            key_fwd = normalize_name(o["fighter_a"]) + "|" + normalize_name(o["fighter_b"])
            key_rev = normalize_name(o["fighter_b"]) + "|" + normalize_name(o["fighter_a"])
            odds_lookup[key_fwd] = o
            odds_lookup[key_rev] = {
                **o,
                "fighter_a": o["fighter_b"], "fighter_b": o["fighter_a"],
                "odds_a": o["odds_b"],       "odds_b": o["odds_a"],
                "fair_prob_a": o["fair_prob_b"], "fair_prob_b": o["fair_prob_a"],
            }
        return odds_lookup, parsed, None   # None = no error
    except ValueError as e:
        return {}, [], str(e)              # pass error message back to UI
    except Exception as e:
        return {}, [], str(e)


def main():
    with st.sidebar:
        st.markdown("## 🥊 UFC Predictor")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Upcoming Event", "⚡ Value Bets", "🎰 Parlays", "🎯 Props", "Fighter Matchup", "Performance", "Elo Leaderboard"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Odds API Key ───────────────────────────────────────────────────────
        # Users supply their own free key from the-odds-api.com
        # It lives in session_state only — never stored or logged
        st.markdown("### 🔑 Odds API Key")

        # Check if a key is baked in via Streamlit secrets (for the host's own deployment)
        import os
        host_key = ""
        try:
            host_key = st.secrets.get("ODDS_API_KEY", "") or os.getenv("ODDS_API_KEY", "")
        except Exception:
            host_key = os.getenv("ODDS_API_KEY", "")

        if host_key:
            # Host has a key configured — use it silently, no input shown
            st.caption("✅ Odds API connected")
            api_key = host_key
        else:
            # Users supply their own key
            user_key = st.text_input(
                "Enter your Odds API key",
                type="password",
                placeholder="Paste key from the-odds-api.com",
                help="Free tier: 500 requests/month. Get yours at the-odds-api.com",
                key="user_odds_api_key",
            )
            api_key = user_key.strip() if user_key else ""
            if not api_key:
                st.caption("No key = predictions only, no odds/edge data")
                st.markdown(
                    "[Get a free key →](https://the-odds-api.com)",
                    unsafe_allow_html=False,
                )
            else:
                st.caption("🔑 Key stored for this session only")

        st.markdown("---")
        st.markdown("### Quick Actions")

        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            for key in ["odds_lookup", "parsed_odds", "odds_error", "all_preds", "odds_api_key_used"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="color:#555;font-size:0.75rem">
            Run pipeline before events:<br>
            <code>python scripts/run_pipeline.py</code><br><br>
            Score after events:<br>
            <code>python scripts/run_pipeline.py --post-event</code>
        </div>
        """, unsafe_allow_html=True)

    session = get_db()

    # ── Load odds once, keyed by api_key so each user gets their own cache ────
    # Re-fetch if the key changed since last load (user pasted a new key)
    key_changed = st.session_state.get("odds_api_key_used") != api_key
    if "odds_lookup" not in st.session_state or key_changed:
        if api_key:
            with st.spinner("Fetching odds (once per session)..."):
                odds_lookup, parsed_odds, odds_error = load_odds_once(api_key)
                st.session_state["odds_lookup"] = odds_lookup
                st.session_state["parsed_odds"] = parsed_odds
                st.session_state["odds_error"]  = odds_error
                st.session_state["odds_api_key_used"] = api_key
        else:
            st.session_state["odds_lookup"] = {}
            st.session_state["parsed_odds"] = []
            st.session_state["odds_error"]  = None
            st.session_state["odds_api_key_used"] = ""

    odds_lookup = st.session_state["odds_lookup"]
    parsed_odds = st.session_state["parsed_odds"]
    odds_error  = st.session_state.get("odds_error")

    # Show error banner if key was invalid
    if odds_error:
        st.error(f"⚠️ Odds API: {odds_error}")

    if page == "Upcoming Event":
        page_upcoming_event(session, odds_lookup)
    elif page == "⚡ Value Bets":
        page_value_bets(session, parsed_odds, odds_lookup)
    elif page == "🎰 Parlays":
        page_parlays(session, parsed_odds, odds_lookup)
    elif page == "🎯 Props":
        page_props(session, odds_lookup)
    elif page == "Fighter Matchup":
        page_fighter_compare(session)
    elif page == "Performance":
        page_performance(session)
    elif page == "Elo Leaderboard":
        page_leaderboard(session)


if __name__ == "__main__":
    main()
