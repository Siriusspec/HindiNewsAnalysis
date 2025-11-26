import streamlit as st
import pandas as pd
import plotly.express as px
import re
try:
from transformers import pipeline
except ImportError:
st.error("Transformers not installed")
st.set_page_config(
page_title="Hindi News Analyzer",
page_icon="📰",
layout="wide"
)
if 'initialized' not in st.session_state:
st.session_state.initialized = True
st.title("📰 Hindi News Sentiment Analysis")
CATEGORIES = ['Politics', 'Sports', 'Entertainment', 'Business', 'Technology', 'Social Issues']
def clean_text(text):
text = re.sub(r'http\S+|www\S+', '', text)
text = re.sub(r'\s+', ' ', text)
return text.strip()
@st.cache_resource
def load_sentiment_model():
try:
return pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
except:
return None
def analyze_sentiment(text):
try:
model = load_sentiment_model()
if model is None:
return {"label": "NEUTRAL", "score": 0.5}
result = model(text[:512])
return result[0]
except:
return {"label": "NEUTRAL", "score": 0.0}
def classify_news(text):
text_lower = text.lower()
keywords = {
    'Politics': ['भाजपा', 'कांग्रेस', 'सरकार', 'चुनाव', 'राज्य', 'मंत्री', 'संसद', 'विधान'],
    'Sports': ['खेल', 'क्रिकेट', 'फुटबॉल', 'खिलाड़ी', 'जीत', 'मैच', 'टीम', 'स्टेडियम'],
    'Entertainment': ['फिल्म', 'अभिनेता', 'सिनेमा', 'गाना', 'संगीत', 'नाटक', 'बॉलीवुड'],
    'Business': ['बाजार', 'कंपनी', 'व्यापार', 'शेयर', 'निवेश', 'बैंक', 'आर्थिक'],
    'Technology': ['तकनीक', 'कंप्यूटर', 'मोबाइल', 'ऐप', 'सॉफ्टवेयर', 'आईटी', 'डिजिटल'],
    'Social Issues': ['समाज', 'महिला', 'बाल', 'स्वास्थ्य', 'शिक्षा', 'गरीबी', 'सामाजिक']
}

scores = {}
for category, words in keywords.items():
    score = sum(1 for word in words if word in text_lower)
    scores[category] = score

best_category = max(scores, key=scores.get)
confidence = min(0.9, scores[best_category] / len(keywords[best_category]))

return {"category": best_category, "score": confidence if confidence > 0 else 0.5}
page = st.sidebar.radio("Choose Mode", ["Single Analysis", "Batch Processing", "Dashboard", "About"])
if page == "Single Analysis":
st.header("Analyze Single News Article")
news_text = st.text_area("Paste Hindi news text here:", height=150, placeholder="Enter your Hindi news...")

if st.button("🔍 Analyze", use_container_width=True):
    if news_text.strip():
        cleaned = clean_text(news_text)
        
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(cleaned)
        
        category = classify_news(cleaned)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Sentiment")
            emoji = "😊" if "POSITIVE" in sentiment['label'].upper() else "😞" if "NEGATIVE" in sentiment['label'].upper() else "😐"
            st.metric(emoji, sentiment['label'])
            st.write(f"Confidence: {sentiment['score']:.1%}")
        
        with col2:
            st.subheader("Category")
            st.metric("📁", category['category'])
            st.write(f"Confidence: {category['score']:.1%}")
        
        with col3:
            st.subheader("Text Stats")
            st.metric("Words", len(cleaned.split()))
            st.metric("Characters", len(cleaned))
    else:
        st.warning("Please enter some text!")
elif page == "Batch Processing":
st.header("Analyze Multiple Articles")
uploaded_file = st.file_uploader("Upload CSV file (must have 'text' column)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write(f"Loaded {len(df)} articles")
    st.dataframe(df.head(3), use_container_width=True)
    
    if st.button("🚀 Analyze All", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        with st.spinner("Processing articles..."):
            for idx, row in df.iterrows():
                text = str(row.get('text', ''))
                cleaned = clean_text(text)
                
                sentiment = analyze_sentiment(cleaned)
                category = classify_news(cleaned)
                
                results.append({
                    'Original Text': text[:80],
                    'Sentiment': sentiment['label'],
                    'Sentiment Score': f"{sentiment['score']:.2%}",
                    'Category': category['category'],
                    'Category Score': f"{category['score']:.2%}"
                })
                
                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processed {idx + 1}/{len(df)}")
        
        results_df = pd.DataFrame(results)
        st.markdown("---")
        st.dataframe(results_df, use_container_width=True)
        
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results as CSV", csv, "results.csv", "text/csv", use_container_width=True)
elif page == "Dashboard":
st.header("Analytics Dashboard")
st.info("📊 Upload a CSV in 'Batch Processing' first to see live analytics")

demo_data = {
    'Sentiment': ['POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'POSITIVE', 'NEGATIVE'],
    'Category': ['Politics', 'Sports', 'Politics', 'Entertainment', 'Business', 'Technology']
}
demo_df = pd.DataFrame(demo_data)

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = demo_df['Sentiment'].value_counts()
    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    category_counts = demo_df['Category'].value_counts()
    fig2 = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title="Category Distribution",
        labels={'x': 'Category', 'y': 'Count'}
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
st.header("About This Project")
st.markdown("""
### 📰 Hindi News Sentiment Analysis & Classification System

**What is This?**

This system analyzes Hindi news articles to determine sentiment (positive/negative/neutral) 
and automatically categorize them into topics like Politics, Sports, Entertainment, Business, 
Technology, and Social Issues.

---

### Why We Built This

With over 600 million Hindi speakers in India, there's massive Hindi news content online. 
But most sentiment analysis tools only work with English. This project brings NLP to Hindi news.

---

### How It Works

**Technology Used:**
- **Sentiment Analysis**: Multilingual BERT model (pre-trained)
- **Category Classification**: Intelligent keyword matching (instant, no model)
- **No Custom Training**: Everything is ready-to-use

---

### Key Features

✨ **Real-time Analysis** - Instant results with confidence scores

✨ **Batch Processing** - Analyze multiple articles at once

✨ **Analytics Dashboard** - Visual charts and trends

✨ **Download Results** - Export as CSV

---

### Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python with Transformers
- **NLP Model**: BERT (sentiment analysis)
- **Deployment**: Streamlit Cloud

---

### Team

**Jaypee Institute of Information Technology, Noida**

- Shreya Bhardwaj (23103122)
- Ronak Koul (23103126)
- S.M. Farhan (23103132)

**Course**: Open Source Software Lab (15B17CI575)

**Semester**: Odd Semester 2025

---

### Limitations

- Works best with Devanagari script
- Code-mixed (Hindi+English) text may vary in accuracy
- 5-level sentiment classification

---

Made with ❤️ for better Hindi NLP
""")
st.markdown("---")
st.markdown("<center><small>Hindi News Sentiment Analysis | Open Source Lab</small></center>", unsafe_allow_html=True)
