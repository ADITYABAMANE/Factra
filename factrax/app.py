"""
app.py
------
Streamlit UI for the Fake News Detector.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.pipeline import FakeNewsPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .trust-high   { color: #1a7a4a; font-weight: 600; }
    .trust-mid    { color: #b45309; font-weight: 600; }
    .trust-low    { color: #b91c1c; font-weight: 600; }
    .label-fake   { background: #fee2e2; color: #991b1b; padding: 4px 14px;
                    border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    .label-real   { background: #dcfce7; color: #166534; padding: 4px 14px;
                    border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    .label-uncertain { background: #fef9c3; color: #854d0e; padding: 4px 14px;
                    border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    .section-box  { background: #f8fafc; border: 1px solid #e2e8f0;
                    border-radius: 10px; padding: 16px; margin-top: 12px; }
    .flag-item    { color: #991b1b; margin: 4px 0; }
    .signal-item  { color: #166534; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown(
    "Paste any news headline or statement below. "
    "The AI will analyse it and give you a **trust score** with reasoning."
)
st.divider()

# ── Load pipeline (cached so model loads only once) ───────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipeline():
    return FakeNewsPipeline()

try:
    pipeline = load_pipeline()
except ValueError as e:
    st.error(f"Setup error: {e}")
    st.stop()

# ── Example news buttons ──────────────────────────────────────────────────────
st.markdown("**Try an example:**")
examples = [
    "Scientists discover that drinking lemon juice cures all types of cancer overnight.",
    "NASA confirms water ice found near the lunar south pole.",
    "Government secretly putting mind-control chips in COVID vaccines.",
    "India launches its first solar observatory mission Aditya-L1.",
]

cols = st.columns(2)
selected_example = None
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex[:55] + "...", key=f"ex_{i}"):
        selected_example = ex

# ── Input ─────────────────────────────────────────────────────────────────────
default_text = selected_example or ""
news_text = st.text_area(
    "Enter news headline or statement:",
    value=default_text,
    height=120,
    placeholder="e.g. Scientists discover that coffee consumption eliminates all diseases...",
)

analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyse_btn:
    if not news_text.strip():
        st.warning("Please enter a news statement first.")
    else:
        with st.spinner("Analysing... this takes 5–10 seconds"):
            try:
                result = pipeline.analyse(news_text)
            except ValueError as e:
                st.warning(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        # ── Trust Score ───────────────────────────────────────────────────────
        score = result["trust_score"]
        label = result["final_label"]

        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Trust Score")
            if score >= 65:
                css_class = "trust-high"
                emoji = "✅"
            elif score >= 40:
                css_class = "trust-mid"
                emoji = "⚠️"
            else:
                css_class = "trust-low"
                emoji = "❌"

            st.markdown(
                f"<div style='font-size:3rem; font-weight:700;' "
                f"class='{css_class}'>{emoji} {score}/100</div>",
                unsafe_allow_html=True,
            )

            label_css = {
                "FAKE": "label-fake",
                "REAL": "label-real",
                "UNCERTAIN": "label-uncertain",
            }.get(label, "label-uncertain")

            st.markdown(
                f"<br><span class='{label_css}'>{label}</span>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("### Verdict")
            st.info(result["verdict"])

        # ── BERT Model result ─────────────────────────────────────────────────
        with st.expander("🤖 AI Classifier (BERT) result"):
            b_label = result["bert_label"]
            b_conf = round(result["bert_confidence"] * 100)
            b_fake = round(result["bert_fake_prob"] * 100)
            b_real = round(result["bert_real_prob"] * 100)

            st.markdown(f"**BERT verdict:** `{b_label}` ({b_conf}% confidence)")
            st.progress(result["bert_real_prob"], text=f"Real probability: {b_real}%")
            st.progress(result["bert_fake_prob"], text=f"Fake probability: {b_fake}%")
            st.caption(
                "BERT is a fine-tuned language model trained on thousands of "
                "labelled news statements. It gives a quick first-pass verdict."
            )

        # ── GPT Reasoning ─────────────────────────────────────────────────────
        st.markdown("### 🧠 Detailed Reasoning")
        st.markdown(
            f"<div class='section-box'>{result['reasoning']}</div>",
            unsafe_allow_html=True,
        )

        # ── Red Flags & Positive Signals ──────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 🚩 Red Flags")
            if result["red_flags"]:
                for flag in result["red_flags"]:
                    st.markdown(
                        f"<div class='flag-item'>• {flag}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_No red flags found._")

        with col4:
            st.markdown("### ✅ Positive Signals")
            if result["positive_signals"]:
                for sig in result["positive_signals"]:
                    st.markdown(
                        f"<div class='signal-item'>• {sig}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_No positive signals found._")

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "⚠️ This tool is for educational purposes. "
            "Always verify news with trusted sources like Reuters, AP, or fact-checking sites."
        )

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## How it works")
    st.markdown("""
1. **BERT Classifier** — A fine-tuned transformer model gives an instant fake/real prediction based on patterns learned from thousands of news statements.

2. **GPT-4o Analysis** — OpenAI's GPT cross-checks the claim, identifies red flags, and generates human-readable reasoning.

3. **Trust Score** — Combined verdict from both models, shown as a 0–100 score.
    """)
    st.divider()
    st.markdown("## Tech Stack")
    st.markdown("""
- `Python`
- `HuggingFace Transformers` (BERT)
- `PyTorch`
- `OpenAI API` (GPT-4o-mini)
- `LangChain` (prompt management)
- `Streamlit` (UI)
    """)
    st.divider()
    st.caption("Built as a resume project demonstrating NLP, fine-tuning, and LLM integration.")
