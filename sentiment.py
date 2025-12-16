from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load sentiment analysis pipeline (using Roberta model)
logging.info("Initializing sentiment analysis pipeline.")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    return_all_scores=True
)

# Map model labels to human-readable labels
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Ensure VADER sentiment analyzer is ready
try:
    logging.info("Loading VADER SentimentIntensityAnalyzer.")
    vader = SentimentIntensityAnalyzer()
except LookupError:
    logging.warning("VADER lexicon not found, downloading...")
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()
logging.info("VADER SentimentIntensityAnalyzer is ready.")


def analyze_sentiment(text: str):
    """Analyze sentiment using the Roberta-based model."""
    logging.info(f"Analyzing text with Roberta model: '{text}'")
    results = sentiment_pipeline(text)[0]

    # Map prediction results to human-readable scores
    scores = {
        LABEL_MAP[item["label"]]: round(item["score"], 3)
        for item in results
    }
    logging.debug(f"Model scores: {scores}")
    
    # Determine the top sentiment based on scores
    label = max(scores, key=scores.get)
    score = scores[label]
    logging.info(f"Predicted sentiment: {label}, Score: {score}")

    return label, score, scores


def analyze_vader(text: str):
    """Analyze sentiment using VADER."""
    logging.info(f"Analyzing text with VADER: '{text}'")
    scores = vader.polarity_scores(text)

    # Extract compound score and interpret sentiment
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    logging.info(
        f"VADER sentiment analysis complete. Sentiment: {label}, Compound Score: {compound}"
    )
    logging.debug(f"VADER scores: {scores}")

    return label, round(compound, 3), scores