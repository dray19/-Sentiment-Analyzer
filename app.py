import streamlit as st
import pandas as pd
import altair as alt
import logging
from sentiment import analyze_sentiment, analyze_vader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Sentiment Analyzer")
st.write("Analyze text sentiment using a transformer-based NLP model.")

col_input, col_output = st.columns([2, 1])

def sentiment_card(label, score):
    """
    Renders a sentiment card with gradient background and emoji based on sentiment label.
    """
    gradients = {
        "Positive": "linear-gradient(135deg, #2ecc71, #27ae60)",
        "Negative": "linear-gradient(135deg, #e74c3c, #c0392b)",
        "Neutral":  "linear-gradient(135deg, #f1c40f, #f39c12)"
    }

    emojis = {
        "Positive": "😊",
        "Negative": "😡",
        "Neutral":  "😐"
    }

    st.markdown(
        f"""
        <div style="
            background:{gradients[label]};
            padding:28px;
            border-radius:16px;
            text-align:center;
            color:white;
            box-shadow:0 8px 20px rgba(0,0,0,0.15);
        ">
            <div style="font-size:48px;">{emojis[label]}</div>
            <div style="font-size:26px; font-weight:600;">{label}</div>
            <div style="font-size:16px; opacity:0.9;">
                Confidence: {score:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Input Area
with col_input:
    user_input = st.text_area(
        "Enter text to analyze",
        height=180,
        placeholder="e.g. 'The service was amazing but the wait time was terrible.'"
    )
    st.caption(f"Characters: {len(user_input)}")

# Output Area
with col_output:
    st.markdown("### Prediction")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                label, score, scores = analyze_sentiment(user_input)
                logger.info(f"Analyzed sentiment: {label}, Score: {score}")
                sentiment_card(label, score)
                # Display confidence progress
                st.markdown("### 🔍 Confidence Meter")
                st.progress(float(score))

                # Display model confidence chart
                st.markdown("### Model Confidence")
                df_scores = (
                    pd.DataFrame.from_dict(scores, orient="index", columns=["Probability"])
                    .reset_index()
                    .rename(columns={"index": "Sentiment"})
                )

                chart = (
                    alt.Chart(df_scores)
                    .mark_bar()
                    .encode(
                        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Confidence"),
                        y=alt.Y("Sentiment:N", sort="-x", title=None),
                        color=alt.Color(
                            "Sentiment:N",
                            scale=alt.Scale(
                                domain=["Positive", "Neutral", "Negative"],
                                range=["#2ecc71", "#f1c40f", "#e74c3c"]
                            ),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip("Sentiment:N", title="Sentiment"),
                            alt.Tooltip("Probability:Q", title="Confidence", format=".0%")
                        ]
                    )
                ).properties(height=200)

                st.altair_chart(chart, width='stretch')
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {str(e)}")
                st.error("An error occurred while analyzing sentiment.")
        else:
            st.warning("Please enter some text.")

# Model Comparison
show_comparison = st.checkbox("🔍 Show model comparison")

if show_comparison and user_input.strip():
    st.markdown("## 🔍 Model Comparison")
    try:
        # Analyze text with both models
        label_t, score_t, scores_t = analyze_sentiment(user_input)
        label_v, score_v, scores_v = analyze_vader(user_input)

        # Display Transformer results
        col_t, col_v = st.columns(2)
        with col_t:
            st.markdown("### 🤖 Transformer (RoBERTa)")
            sentiment_card(label_t, score_t)
            st.caption("Context-aware, deep learning model")

        # Display VADER results
        with col_v:
            st.markdown("### 📘 VADER")
            sentiment_card(label_v, score_v)
            st.caption("Rule-based, lexicon-driven model")

        # Display comparison table
        comparison_df = pd.DataFrame({
            "Model": ["RoBERTa", "VADER"],
            "Sentiment": [label_t, label_v],
            "Score": [score_t, score_v]
        })

        col1, _ = st.columns([1, 2])
        with col1:
            st.dataframe(comparison_df, hide_index=True)
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        st.error("An error occurred while comparing models.")

