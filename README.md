# Hindi News Sentiment Analysis System

## Quick Setup

### 1. Install
```bash
git clone 
cd hindi-sentiment-analysis
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud (Free!)
- Push to GitHub
- Go to https://streamlit.io/cloud
- Connect your repo and deploy (2 clicks, done!)

## Features
Real-time sentiment analysis (Positive/Negative/Neutral)
Automatic news categorization (Politics, Sports, Entertainment, Business, Technology, Social Issues)
Single article analysis
Batch processing (upload CSV)
Analytics dashboard
Download results as CSV

## How It Works
- **Sentiment**: Uses a multilingual BERT model (pre‑trained, works with Hindi text)
- **Classification**: Uses keyword‑based category matching (no training required)
- **No training required** - everything is pre-trained
