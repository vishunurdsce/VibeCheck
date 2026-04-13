import streamlit as st
import joblib
import os
import time

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Root of the project (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

st.set_page_config(
    page_title="VibeCheck | Sentiment Intelligence",
    page_icon="🔮",
    layout="wide"
)

# ── Inject premium CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Globals ────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #08080f;
    color: #e2e8f0;
}

/* Animated grain overlay for depth */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(99, 102, 241, 0.08) 0%, transparent 70%),
        radial-gradient(ellipse 60% 50% at 80% 20%, rgba(168, 85, 247, 0.06) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 50% 90%, rgba(236, 72, 153, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c18 0%, #0f0f1e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* sidebar nav items */
.sidebar-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 2px 0;
}
.tag-ml  { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
.tag-nlp { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }
.tag-ens { background: rgba(236,72,153,0.15); color: #f472b6; border: 1px solid rgba(236,72,153,0.25); }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero-wrap {
    padding: 2rem 0 1rem 0;
}

.hero-title {
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 25%, #c084fc 50%, #e879f9 75%, #f472b6 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.05;
    margin-bottom: 0;
    animation: gradientShift 4s ease-in-out infinite;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% center; }
    50%      { background-position: 100% center; }
}

.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    margin-top: 6px;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Input Box ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 18px 24px !important;
    font-size: 1.05rem !important;
    color: #e2e8f0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(12px);
}

.stTextInput > div > div > input:focus {
    border-color: rgba(129, 140, 248, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.1), 0 8px 32px rgba(129, 140, 248, 0.08) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #475569 !important;
}

/* ── Glass Card ─────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 32px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    animation: cardIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(129,140,248,0.2), rgba(192,132,252,0.1), transparent, transparent);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

/* ── Result Badge ───────────────────────────────────────────────────────── */
.result-section {
    text-align: center;
    padding: 20px 0 10px 0;
}

.result-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 4px;
    color: #475569;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 12px;
}

.result-badge {
    display: inline-block;
    padding: 12px 40px;
    border-radius: 100px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 6px;
    text-transform: uppercase;
    animation: badgePulse 2s ease-in-out infinite;
}

@keyframes badgePulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.02); }
}

.badge-pos {
    color: #4ade80;
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.2);
    box-shadow: 0 0 40px rgba(74, 222, 128, 0.1), inset 0 0 40px rgba(74, 222, 128, 0.05);
}

.badge-neg {
    color: #fb7185;
    background: rgba(251, 113, 133, 0.08);
    border: 1px solid rgba(251, 113, 133, 0.2);
    box-shadow: 0 0 40px rgba(251, 113, 133, 0.1), inset 0 0 40px rgba(251, 113, 133, 0.05);
}

.badge-neu {
    color: #60a5fa;
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.2);
    box-shadow: 0 0 40px rgba(96, 165, 250, 0.1), inset 0 0 40px rgba(96, 165, 250, 0.05);
}

/* ── Polarity Gauge ─────────────────────────────────────────────────────── */
.gauge-container {
    margin: 24px auto 8px auto;
    max-width: 400px;
}

.gauge-track {
    height: 8px;
    border-radius: 100px;
    background: linear-gradient(90deg, #fb7185 0%, #64748b 50%, #4ade80 100%);
    position: relative;
    opacity: 0.4;
}

.gauge-marker {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #fff;
    border: 3px solid #818cf8;
    box-shadow: 0 0 12px rgba(129, 140, 248, 0.5);
    transition: left 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #475569;
    letter-spacing: 1px;
    font-weight: 600;
    margin-top: 6px;
    text-transform: uppercase;
}

/* ── Metric Cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(129, 140, 248, 0.2);
    background: rgba(255,255,255,0.04);
    transform: translateY(-2px);
}

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    color: #475569;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #c4b5fd;
}

/* ── Confidence Bar ─────────────────────────────────────────────────────── */
.confidence-wrap {
    margin-top: 20px;
}

.conf-label {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #475569;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 8px;
}

.conf-bar-track {
    height: 6px;
    border-radius: 100px;
    background: rgba(255,255,255,0.05);
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Placeholder / Empty State ──────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    animation: floatIn 0.6s ease both;
}

@keyframes floatIn {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 16px;
    opacity: 0.4;
}

.empty-text {
    color: #475569;
    font-size: 1rem;
    max-width: 320px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── History Pills ──────────────────────────────────────────────────────── */
.history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 6px;
    font-size: 0.78rem;
    transition: background 0.2s;
}

.history-item:hover {
    background: rgba(255,255,255,0.05);
}

.history-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-pos { background: #4ade80; box-shadow: 0 0 6px rgba(74,222,128,0.4); }
.dot-neg { background: #fb7185; box-shadow: 0 0 6px rgba(251,113,133,0.4); }
.dot-neu { background: #60a5fa; box-shadow: 0 0 6px rgba(96,165,250,0.4); }

.history-text {
    color: #94a3b8;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-badge {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 100px;
}

.hb-pos { color: #4ade80; background: rgba(74,222,128,0.1); }
.hb-neg { color: #fb7185; background: rgba(251,113,133,0.1); }
.hb-neu { color: #60a5fa; background: rgba(96,165,250,0.1); }

/* ── Misc overrides ─────────────────────────────────────────────────────── */
.stDivider { opacity: 0.06 !important; }

/* Hide default Streamlit metric styling */
[data-testid="metric-container"] { display: none; }

/* Button style */
.stButton > button {
    background: rgba(129, 140, 248, 0.1) !important;
    border: 1px solid rgba(129, 140, 248, 0.2) !important;
    border-radius: 12px !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: rgba(129, 140, 248, 0.2) !important;
    border-color: rgba(129, 140, 248, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session state for history ────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_models():
    try:
        vec   = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
        model = joblib.load(os.path.join(MODELS_DIR, 'best_sentiment_model.pkl'))
        return vec, model
    except Exception:
        return None, None


def hybrid_predict(text, vec, model):
    """
    Hybrid prediction: ML model + TextBlob polarity as tiebreaker.
    The ML model is trained on the dataset, but for out-of-vocabulary
    casual text, TextBlob polarity provides a reliable fallback.
    """
    # ML prediction
    ml_pred = model.predict(vec.transform([text]))[0]

    if not HAS_TEXTBLOB:
        return ml_pred, 0, 0, "N/A", 0.75

    blob = TextBlob(text)
    pol  = round(blob.sentiment.polarity, 2)
    subj = round(blob.sentiment.subjectivity, 2)

    # Only override ML when TextBlob has a STRONG opinion.
    if pol >= 0.3 and ml_pred != "Positive":
        final = "Positive"
    elif pol <= -0.3 and ml_pred != "Negative":
        final = "Negative"
    else:
        final = ml_pred

    if pol > 0.1:   tone = "Optimistic"
    elif pol < -0.1: tone = "Pessimistic"
    else:            tone = "Neutral"

    # Confidence heuristic: how strongly both signals agree
    agreement = 1.0 if (
        (final == "Positive" and pol > 0) or
        (final == "Negative" and pol < 0) or
        (final == "Neutral" and abs(pol) < 0.15)
    ) else 0.6
    confidence = round(min(1.0, (abs(pol) + 0.5) * agreement), 2)

    return final, pol, subj, tone, confidence


def render_polarity_gauge(polarity):
    """Render a horizontal polarity gauge from -1 to +1."""
    pct = (polarity + 1) / 2 * 100  # map [-1, 1] → [0%, 100%]
    return f"""
<div class="gauge-container">
<div class="gauge-track">
<div class="gauge-marker" style="left: {pct}%;"></div>
</div>
<div class="gauge-labels">
<span>Negative</span>
<span>Neutral</span>
<span>Positive</span>
</div>
</div>
"""


def render_metric_card(icon, label, value):
    """Render a single glassmorphic metric card."""
    return f"""
<div class="metric-card">
<div class="metric-icon">{icon}</div>
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
</div>
"""


def render_confidence_bar(confidence):
    """Render a gradient confidence bar."""
    pct = int(confidence * 100)
    return f"""
<div class="confidence-wrap">
<div class="conf-label">Model Confidence — {pct}%</div>
<div class="conf-bar-track">
<div class="conf-bar-fill" style="width: {pct}%;"></div>
</div>
</div>
"""


def render_history_item(text, prediction):
    """Render a single history row."""
    dot_cls = {"Positive": "dot-pos", "Negative": "dot-neg"}.get(prediction, "dot-neu")
    hb_cls  = {"Positive": "hb-pos",  "Negative": "hb-neg"}.get(prediction, "hb-neu")
    short   = text[:50] + ("…" if len(text) > 50 else "")
    return f"""
<div class="history-item">
<div class="history-dot {dot_cls}"></div>
<div class="history-text">{short}</div>
<div class="history-badge {hb_cls}">{prediction.upper()}</div>
</div>
"""


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("VibeCheck")
        st.markdown("""
        <div style="margin: 8px 0 20px 0;">
            <span class="sidebar-tag tag-ml">ML Ensemble</span>
            <span class="sidebar-tag tag-nlp">NLP</span>
            <span class="sidebar-tag tag-ens">Hybrid</span>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown("""
<div style="padding: 12px 0;">
<div style="font-size: 0.65rem; letter-spacing: 2px; color: #475569; text-transform: uppercase; font-weight: 600; margin-bottom: 10px;">How it works</div>
<div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.8;">
<b style="color:#818cf8;">1.</b> Text vectorized via TF-IDF<br>
<b style="color:#a78bfa;">2.</b> Voted by SVC + LR + RF<br>
<b style="color:#c084fc;">3.</b> TextBlob polarity tiebreaker<br>
<b style="color:#e879f9;">4.</b> Confidence score computed
</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # History section
        if st.session_state.history:
            st.markdown("""
            <div style="font-size: 0.65rem; letter-spacing: 2px; color: #475569; text-transform: uppercase; font-weight: 600; margin-bottom: 10px;">Recent analyses</div>
            """, unsafe_allow_html=True)
            for item in reversed(st.session_state.history[-8:]):
                st.markdown(render_history_item(item['text'], item['pred']), unsafe_allow_html=True)
            st.divider()

        if st.button("🔄  Clear & Reset"):
            st.session_state.history = []
            st.cache_resource.clear()
            st.rerun()

    # ── Main Content ─────────────────────────────────────────────────────────
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">VibeCheck</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time emotional intelligence · Powered by hybrid ML + NLP</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_input(
        label="Sentence",
        placeholder="Type anything — a thought, a review, a feeling...",
        label_visibility="collapsed"
    )

    vec, model = load_models()

    if not vec or not model:
        st.error("⚠️  Models not found. Run `python3 src/main.py` first.")
        return

    if user_input.strip():
        # Animate a quick "thinking" state
        with st.spinner(""):
            time.sleep(0.3)

        prediction, pol, subj, tone, confidence = hybrid_predict(user_input, vec, model)

        # Save to history
        if not st.session_state.history or st.session_state.history[-1]['text'] != user_input:
            st.session_state.history.append({'text': user_input, 'pred': prediction})

        badge_cls = {"Positive": "badge-pos", "Negative": "badge-neg"}.get(prediction, "badge-neu")
        emoji_map = {"Positive": "✨", "Negative": "💔", "Neutral": "💭"}
        emoji = emoji_map.get(prediction, "💭")

        # ── Result Card ──────────────────────────────────────────────────────
        st.markdown(f"""
<div class="glass-card">
<div class="result-section">
<div class="result-eyebrow">AI Detection Result</div>
<div class="result-badge {badge_cls}">{emoji} {prediction.upper()}</div>
</div>

{render_polarity_gauge(pol)}
{render_confidence_bar(confidence)}
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # ── Metric Cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tone_emoji = {"Optimistic": "🙂", "Pessimistic": "😔", "Neutral": "😐"}.get(tone, "😐")
            st.markdown(render_metric_card(tone_emoji, "Emotion Tone", tone), unsafe_allow_html=True)
        with c2:
            pol_emoji = "🟢" if pol > 0 else ("🔴" if pol < 0 else "⚪")
            st.markdown(render_metric_card(pol_emoji, "Polarity", f"{pol:+.2f}"), unsafe_allow_html=True)
        with c3:
            subj_emoji = "🎯" if subj > 0.5 else "📐"
            st.markdown(render_metric_card(subj_emoji, "Subjectivity", f"{subj:.2f}"), unsafe_allow_html=True)
        with c4:
            conf_emoji = "🔥" if confidence > 0.8 else ("⚡" if confidence > 0.5 else "❓")
            st.markdown(render_metric_card(conf_emoji, "Confidence", f"{int(confidence*100)}%"), unsafe_allow_html=True)

    else:
        # ── Empty State ──────────────────────────────────────────────────────
        st.markdown("""
<div class="empty-state">
<div class="empty-icon">🔮</div>
<div class="empty-text">
Type a sentence above and press <strong>Enter</strong><br>
to unlock real-time sentiment analysis
</div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
