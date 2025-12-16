# 🎭 Sentiment Analyzer

A lightweight, interactive sentiment analysis application built with Streamlit, combining state-of-the-art transformer models from Hugging Face with the classic NLTK VADER sentiment analyzer.

## ✨ Features

- **Dual Analysis Approach**: Compare results from both transformer-based and lexicon-based sentiment analysis
- **Interactive UI**: Clean, user-friendly interface powered by Streamlit
- **Real-time Analysis**: Instant sentiment analysis on any text input
- **Example Texts**: Pre-loaded examples to quickly test the application
- **Detailed Metrics**: View comprehensive sentiment scores and confidence levels
- **Visual Comparison**: Side-by-side comparison of both analysis methods

## 🛠️ Technologies

- **Streamlit**: Interactive web application framework
- **Hugging Face Transformers**: DistilBERT model fine-tuned for sentiment analysis
- **NLTK VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **PyTorch**: Deep learning framework for transformer models

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/dray19/-Sentiment-Analyzer.git
cd -Sentiment-Analyzer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

## 💡 How It Works

### NLTK VADER
- **Rule-based** lexicon approach
- Analyzes sentiment intensity using a pre-built dictionary
- Handles negations, intensifiers, and emoticons
- Returns compound score (-1 to +1) and individual pos/neu/neg scores
- Best for: Social media text, short messages, casual language

### Hugging Face Transformer (DistilBERT)
- **Deep learning** model trained on Stanford Sentiment Treebank
- Context-aware understanding of language
- Returns sentiment label (POSITIVE/NEGATIVE) with confidence score
- Best for: Complex sentences, nuanced text, formal language

## 📊 Example Usage

1. Select a pre-loaded example or enter your own text
2. Click "🔍 Analyze Sentiment"
3. View results from both models side-by-side
4. Compare the agreement between different approaches

## 🎯 Use Cases

- Analyze customer reviews and feedback
- Monitor social media sentiment
- Evaluate product descriptions
- Assess survey responses
- Study sentiment patterns in text data

## 📝 Requirements

See `requirements.txt` for the complete list of dependencies:
- streamlit==1.29.0
- transformers==4.35.2
- torch==2.1.1
- nltk==3.8.1

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ for sentiment analysis enthusiasts