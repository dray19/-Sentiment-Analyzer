# Sentiment Analysis Web App (Transformers + VADER)

A lightweight, interactive **sentiment analysis application** built with **Streamlit**, combining **state-of-the-art transformer models** from Hugging Face with the classic **NLTK VADER** sentiment analyzer.

This project is designed as a **portfolio-ready example** demonstrating applied NLP, model comparison, and rapid ML app deployment.

---

## Why This Project?

This repository showcases:

- Practical **Natural Language Processing (NLP)** using pretrained transformer models
- Comparison between **modern deep learning** and **rule-based** sentiment approaches
- Clean separation between **model logic** and **application layer**
- Rapid prototyping and deployment using **Streamlit**

It is ideal for demonstrating skills in:
- Python
- Machine Learning / NLP
- Hugging Face Transformers
- Streamlit app development
- Reproducible environments

---
## Live Demo (Local)

```bash
streamlit run app.py
http://localhost:8501
```
---

## Application Architecture
```
.
├── app.py              # Streamlit UI and user interaction
├── sentiment.py        # Model loading and sentiment inference logic
├── requirements.txt    # Reproducible dependency list
└── README.md
```
---

## Installation
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate  
```
-Install dependencies
```
pip install -r requirements.txt
```
-Running the App
```
streamlit run app.py
```
---

## Streamlit Profile
https://share.streamlit.io/user/dray19
