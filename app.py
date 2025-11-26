import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="Hindi News Analyzer", page_icon="📰", layout="wide")

# Try importing transformers
HAS_TRANSFORMERS = True
try:
    from transformers import pipeline
except Exception:
    HAS_TRANSFORMERS = False
    st.warning("Transformers library not installed.")

# Initialize session state to persist data
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.batch_results = None  # Store batch results

st.title("📰 Hindi News Sentiment Analysis")

CATEGORIES = ['Politics', 'Sports', 'Entertainment', 'Business', 'Technology', 'Social Issues']

def clean_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_sentiment_model():
    if not HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception:
        return None

def analyze_sentiment(text: str):
    model = load_sentiment_model()
    if model is None:
        return {"label": "NEUTRAL", "score": 0.5}
    try:
        out = model(text[:512])
        if isinstance(out, list) and len(out) > 0:
            raw_label = out[0]['label']
            score = out[0]['score']

            # Map stars → sentiment
            if raw_label in ["1 star", "2 stars"]:
                label = "NEGATIVE"
            elif raw_label == "3 stars":
                label = "NEUTRAL"
            else:  # "4 stars" or "5 stars"
                label = "POSITIVE"

            return {"label": label, "score": score}
        return {"label": "NEUTRAL", "score": 0.0}
    except Exception:
        return {"label": "NEUTRAL", "score": 0.0}

def classify_news(text: str):
    text_lower = text.lower()
    keywords = {
        'Politics': ['भाजपा', 'कांग्रेस', 'सरकार', 'चुनाव', 'राज्य', 'मंत्री', 'संसद', 'विधान'],
        'Sports': ['खेल', 'क्रिकेट', 'फुटबॉल', 'खिलाड़ी', 'जीत', 'मैच', 'टीम', 'स्टेडियम'],
        'Entertainment': ['फिल्म', 'अभिनेता', 'सिनेमा', 'गाना', 'संगीत', 'नाटक', 'बॉलीवुड'],
        'Business': ['बाजार', 'कंपनी', 'व्यापार', 'शेयर', 'निवेश', 'बैंक', 'आर्थिक'],
        'Technology': ['तकनीक', 'कंप्यूटर', 'मोबाइल', 'ऐप', 'सॉफ्टवेयर', 'आईटी', 'डिजिटल'],
        'Social Issues': ['समाज', 'महिला', 'बाल', 'स्वास्थ्य', 'शिक्षा', 'गरीबी', 'सामाजिक']
    }

    scores = {cat: sum(1 for w in words if w in text_lower) for cat, words in keywords.items()}
    best = max(scores, key=scores.get)
    confidence = 0.5 if scores[best] == 0 else min(0.9, scores[best] / len(keywords[best]))
    return {"category": best, "score": confidence}

page = st.sidebar.radio("Choose Mode", ["Single Analysis", "Batch Processing", "Dashboard", "About"])

# ---------------- SINGLE ANALYSIS ----------------
if page == "Single Analysis":
    st.header("Analyze Single News Article")
    news_text = st.text_area("Paste Hindi news text here:", height=150, placeholder="Enter your Hindi news...")
    if st.button(" Analyze"):
        if news_text.strip():
            cleaned = clean_text(news_text)
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(cleaned)
            category = classify_news(cleaned)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Sentiment")
                label = sentiment.get('label', 'NEUTRAL')
                emoji = "😊" if "POS" in label.upper() else "😞" if "NEG" in label.upper() else "😐"
                st.metric(emoji, label)
                st.write(f"Confidence: {sentiment.get('score', 0.0):.1%}")

            with col2:
                st.subheader("Category")
                st.metric("", category['category'])
                st.write(f"Confidence: {category['score']:.1%}")

            with col3:
                st.subheader("Text Stats")
                st.metric("Words", len(cleaned.split()))
                st.metric("Characters", len(cleaned))
        else:
            st.warning("Please enter some text!")

# ---------------- BATCH PROCESSING ----------------
elif page == "Batch Processing":
    st.header("Analyze Multiple Articles")
    uploaded_file = st.file_uploader("Upload CSV file (must have 'text' column)", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None and 'text' in df.columns:
            st.write(f"Loaded {len(df)} articles")
            st.dataframe(df.head(3), use_container_width=True)

            if st.button(" Analyze All"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for idx, row in df.iterrows():
                    text = str(row.get('text', ''))
                    cleaned = clean_text(text)
                    sentiment = analyze_sentiment(cleaned)
                    category = classify_news(cleaned)

                    results.append({
                        'Original Text': text[:120],
                        'Sentiment': sentiment.get('label', 'NEUTRAL'),
                        'Sentiment Score': f"{sentiment.get('score', 0.0):.2%}",
                        'Category': category['category'],
                        'Category Score': f"{category['score']:.2%}"
                    })

                    progress_bar.progress((idx + 1) / len(df))
                    status_text.text(f"Processed {idx + 1}/{len(df)}")

                # Save results to session state
                st.session_state.batch_results = pd.DataFrame(results)
                st.success("Analysis complete!")

            # Display saved results
            if st.session_state.batch_results is not None:
                st.markdown("---")
                st.subheader("Results")
                st.dataframe(st.session_state.batch_results, use_container_width=True)

                csv = st.session_state.batch_results.to_csv(index=False)
                st.download_button(" Download Results as CSV", csv, "results.csv", "text/csv", use_container_width=True)

# ---------------- DASHBOARD ----------------
elif page == "Dashboard":
    st.header("Analytics Dashboard")
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results

        sentiments = results_df['Sentiment'].value_counts()
        categories = results_df['Category'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(
                values=sentiments.values,
                names=sentiments.index,
                title="Sentiment Distribution",
                color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                x=categories.index,
                y=categories.values,
                title="Category Distribution",
                labels={'x': 'Category', 'y': 'Count'}
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(results_df))
        with col2:
            st.metric("Positive", len(results_df[results_df['Sentiment'] == 'POSITIVE']))
        with col3:
            st.metric("Negative", len(results_df[results_df['Sentiment'] == 'NEGATIVE']))
    else:
        st.info(" Upload CSV and analyze in 'Batch Processing' to see results here")


# ---------------- ABOUT ----------------
else:  # About
    st.header("About This Project")
    st.markdown("""

 Hindi News Sentiment Analysis & Classification System

**What is This?**  
Analyzes Hindi news articles for sentiment (positive/negative/neutral)  
and automatically categorizes them into topics like Politics, Sports, Entertainment, Business, Technology, and Social Issues.

---

### Key Features
- Real-time analysis with confidence scores  
- Batch processing of multiple articles  
- Visual analytics dashboard  
- Download results as CSV  

---

### Technical Stack
- **Frontend**: Streamlit  
- **Backend**: Python with Transformers  
- **NLP Models**: BERT (sentiment), XLM-RoBERTa (zero-shot)  
- **Deployment**: Streamlit Cloud  

---

Made with ❤️ for better Hindi NLP
""")
    st.markdown("Hindi News Sentiment Analysis | Open Source Lab", unsafe_allow_html=True)
