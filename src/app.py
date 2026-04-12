import streamlit as st
import joblib
import os

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Root of the project (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

st.set_page_config(
    page_title="Sentira | Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 25%, #0d0d1a 0%, #111122 100%);
    color: #f0f0f0;
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 8px;
}

.hero-sub {
    font-size: 1.2rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.result-label {
    font-size: 0.85rem;
    letter-spacing: 3px;
    color: #64748b;
    margin-bottom: 4px;
    margin-top: 20px;
}

.result-pos {
    font-size: 4rem; font-weight: 800; color: #4ade80;
    letter-spacing: 4px;
    text-shadow: 0 0 30px rgba(74, 222, 128, 0.4);
}

.result-neg {
    font-size: 4rem; font-weight: 800; color: #f87171;
    letter-spacing: 4px;
    text-shadow: 0 0 30px rgba(248, 113, 113, 0.4);
}

.result-neu {
    font-size: 4rem; font-weight: 800; color: #60a5fa;
    letter-spacing: 4px;
    text-shadow: 0 0 30px rgba(96, 165, 250, 0.4);
}

/* Override streamlit metric widget colors */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
}
</style>
""", unsafe_allow_html=True)


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
        return ml_pred, 0, 0, "N/A"

    blob = TextBlob(text)
    pol  = round(blob.sentiment.polarity, 2)
    subj = round(blob.sentiment.subjectivity, 2)

    # Only override ML when TextBlob has a STRONG opinion.
    # When TextBlob returns ~0 (doesn't understand the word), trust ML fully.
    if pol >= 0.3 and ml_pred != "Positive":
        final = "Positive"
    elif pol <= -0.3 and ml_pred != "Negative":
        final = "Negative"
    else:
        # Trust the ML model — it was trained on the dataset
        final = ml_pred

    if pol > 0.1:   tone = "Optimistic 🙂"
    elif pol < -0.1: tone = "Pessimistic 😔"
    else:            tone = "Neutral 😐"

    return final, pol, subj, tone


def main():
    with st.sidebar:
        st.title("Sentira Pro")
        st.caption("Hybrid: ML Ensemble + NLP Polarity")
        st.divider()
        st.info("Type any sentence and press **Enter** for instant analysis.")
        if st.button("🔄  Reset cache"):
            st.cache_resource.clear()
            st.rerun()

    st.markdown('<p class="hero-title">Sentira Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">In-Depth Emotional Intelligence Dashboard</p>', unsafe_allow_html=True)

    user_input = st.text_input(
        label="Sentence",
        placeholder="Describe how you feel...",
        label_visibility="collapsed"
    )

    vec, model = load_models()

    if not vec or not model:
        st.error("⚠️  Models not found. Run `python3 src/main.py` first.")
        return

    if user_input.strip():
        prediction, pol, subj, tone = hybrid_predict(user_input, vec, model)

        css_class = {"Positive": "result-pos", "Negative": "result-neg"}.get(prediction, "result-neu")

        # Result — NO wrapping div, just direct markdown
        st.markdown('<p class="result-label">AI DETECTION RESULT</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="{css_class}">{prediction.upper()}</p>', unsafe_allow_html=True)

        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("EMOTION TONE", tone)
        c2.metric("POLARITY SCORE", pol)
        c3.metric("SUBJECTIVITY", subj)

        st.caption(f'Analysed: **"{user_input}"**  · Model: Hybrid (ML Ensemble + NLP)')

    else:
        st.info("💡 Type a sentence above and press **Enter** to start analysis.")


if __name__ == "__main__":
    main()
