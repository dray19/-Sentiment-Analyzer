"""
Sentiment Analyzer - A Streamlit Application
Combines Hugging Face Transformers and NLTK VADER for sentiment analysis
"""

import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide"
)

# Download NLTK VADER lexicon
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

# Initialize VADER
@st.cache_resource
def load_vader():
    """Load NLTK VADER sentiment analyzer"""
    download_nltk_data()
    return SentimentIntensityAnalyzer()

# Initialize Hugging Face transformer model
@st.cache_resource
def load_transformer():
    """Load Hugging Face sentiment analysis pipeline"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

def get_sentiment_emoji(sentiment):
    """Return emoji based on sentiment"""
    if sentiment == "POSITIVE" or sentiment == "positive":
        return "😊"
    elif sentiment == "NEGATIVE" or sentiment == "negative":
        return "😞"
    else:
        return "😐"

def get_vader_sentiment_label(compound_score):
    """
    Determine sentiment label from VADER compound score
    
    Args:
        compound_score: VADER compound score (-1 to 1)
    
    Returns:
        Sentiment label: "POSITIVE", "NEGATIVE", or "NEUTRAL"
    """
    if compound_score >= 0.05:
        return "POSITIVE"
    elif compound_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def analyze_with_vader(text, vader_analyzer):
    """Analyze sentiment using NLTK VADER"""
    scores = vader_analyzer.polarity_scores(text)
    sentiment = get_vader_sentiment_label(scores['compound'])
    
    return {
        'sentiment': sentiment,
        'scores': scores,
        'compound': scores['compound']
    }

def analyze_with_transformer(text, transformer_pipeline):
    """Analyze sentiment using Hugging Face Transformer"""
    result = transformer_pipeline(text[:512])[0]  # Limit to 512 tokens
    return {
        'sentiment': result['label'],
        'confidence': result['score']
    }

def main():
    # Title and description
    st.title("🎭 Sentiment Analyzer")
    st.markdown("""
    A lightweight, interactive sentiment analysis application combining:
    - 🤗 **Hugging Face Transformers** (DistilBERT)
    - 📊 **NLTK VADER** (Valence Aware Dictionary and sEntiment Reasoner)
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        vader_analyzer = load_vader()
        transformer_pipeline = load_transformer()
    
    st.success("✅ Models loaded successfully!")
    
    # Input section
    st.header("📝 Enter Text for Analysis")
    
    # Provide example texts
    examples = [
        "I absolutely love this product! It's amazing and exceeded all my expectations!",
        "This is the worst experience I've ever had. Completely disappointing.",
        "The weather is okay today. Nothing special.",
        "I'm so excited about this new opportunity! Can't wait to get started!",
        "I feel terrible about the situation. Everything went wrong."
    ]
    
    example_choice = st.selectbox(
        "Choose an example or enter your own text:",
        ["Custom text"] + examples
    )
    
    if example_choice == "Custom text":
        user_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        user_input = st.text_area(
            "Enter your text here:",
            value=example_choice,
            height=150
        )
    
    # Analysis button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            # VADER Analysis
            with col1:
                st.subheader("📊 NLTK VADER Analysis")
                with st.spinner("Analyzing with VADER..."):
                    vader_result = analyze_with_vader(user_input, vader_analyzer)
                
                sentiment_emoji = get_sentiment_emoji(vader_result['sentiment'])
                st.markdown(f"### {sentiment_emoji} Sentiment: **{vader_result['sentiment']}**")
                
                st.markdown("**Detailed Scores:**")
                st.progress(vader_result['scores']['pos'], text=f"Positive: {vader_result['scores']['pos']:.3f}")
                st.progress(vader_result['scores']['neu'], text=f"Neutral: {vader_result['scores']['neu']:.3f}")
                st.progress(vader_result['scores']['neg'], text=f"Negative: {vader_result['scores']['neg']:.3f}")
                st.metric("Compound Score", f"{vader_result['compound']:.3f}")
                
                st.info("""
                **VADER** uses a lexicon-based approach, analyzing the intensity of 
                emotions and handling negations, intensifiers, and emoticons.
                """)
            
            # Transformer Analysis
            with col2:
                st.subheader("🤗 Hugging Face Transformer Analysis")
                with st.spinner("Analyzing with Transformer..."):
                    transformer_result = analyze_with_transformer(user_input, transformer_pipeline)
                
                sentiment_emoji = get_sentiment_emoji(transformer_result['sentiment'])
                st.markdown(f"### {sentiment_emoji} Sentiment: **{transformer_result['sentiment']}**")
                
                st.markdown("**Confidence:**")
                st.progress(transformer_result['confidence'], 
                           text=f"Confidence: {transformer_result['confidence']:.3f}")
                st.metric("Confidence Score", f"{transformer_result['confidence']:.1%}")
                
                st.info("""
                **DistilBERT** is a transformer-based model fine-tuned on the 
                Stanford Sentiment Treebank (SST-2), providing context-aware analysis.
                """)
            
            # Comparison section
            st.header("📊 Comparison")
            st.markdown(f"""
            - **VADER Sentiment:** {vader_result['sentiment']} (Compound: {vader_result['compound']:.3f})
            - **Transformer Sentiment:** {transformer_result['sentiment']} (Confidence: {transformer_result['confidence']:.1%})
            """)
            
            # Agreement check
            # Transformer only returns POSITIVE or NEGATIVE, not NEUTRAL
            # For VADER NEUTRAL, we consider them in agreement if confidence is low
            if vader_result['sentiment'] == transformer_result['sentiment']:
                st.success("✅ Both models agree on the sentiment!")
            elif vader_result['sentiment'] == "NEUTRAL" and transformer_result['confidence'] < 0.75:
                st.info(f"ℹ️ VADER detected neutral sentiment while Transformer shows weak {transformer_result['sentiment'].lower()} sentiment.")
            else:
                st.warning("⚠️ The models have different interpretations. Consider the context and nuances of the text.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About the Models
    
    **NLTK VADER** is excellent for:
    - Social media text
    - Short texts with emoticons and slang
    - Quick, rule-based analysis
    
    **Hugging Face Transformers** excel at:
    - Understanding context and nuance
    - Longer, complex sentences
    - Deep learning-based predictions
    """)

if __name__ == "__main__":
    main()
