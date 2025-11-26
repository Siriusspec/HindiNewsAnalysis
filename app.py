import streamlit as st
import pandas as pd
import transformers
import plotly.express as px
import re

# Page config
st.set_page_config(
    page_title="Hindi News Analyzer",
    page_icon="📰",
    layout="wide"
)

st.title("Hindi News Sentiment Analysis")

# Load models once (cached)
@st.cache_resource
def load_sentiment_model():
    return transformers.pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

@st.cache_resource
def load_zero_shot_classifier():
    return transformers.pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

# Load models
sentiment_model = load_sentiment_model()
classifier = load_zero_shot_classifier()

# Categories
CATEGORIES = ['Politics', 'Sports', 'Entertainment', 'Business', 'Technology', 'Social Issues']

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    try:
        result = sentiment_model(text[:512])
        return result[0]
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.0}

def classify_news(text):
    try:
        result = classifier(text[:512], CATEGORIES, multi_class=False)
        return {
            "category": result['labels'][0],
            "score": result['scores'][0]
        }
    except Exception as e:
        return {"category": "Unknown", "score": 0.0}

# Sidebar navigation
page = st.sidebar.radio(
    "Choose Mode",
    ["Single Analysis", "Batch Processing", "Dashboard", "About"]
)

# ---------------- SINGLE ANALYSIS ----------------
if page == "Single Analysis":
    st.header("Analyze Single News Article")

    news_text = st.text_area(
        "Paste Hindi news text here:",
        height=150,
        placeholder="Enter your Hindi news..."
    )

    if st.button("Analyze", use_container_width=True):
        if news_text.strip():
            cleaned = clean_text(news_text)

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
        df = pd.read_csv(uploaded_file)

        st.write(f"Loaded {len(df)} articles")
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("Analyze All", use_container_width=True):
            progress_bar = st.progress(0)
            results = []

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

            results_df = pd.DataFrame(results)
            st.markdown("---")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results as CSV",
                csv,
                "results.csv",
                "text/csv",
                use_container_width=True
            )

# ---------------- DASHBOARD ----------------
elif page == "Dashboard":
    st.header("Analytics Dashboard")

    st.info("Upload a CSV in 'Batch Processing' to see live analytics")

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
            title="Sentiment Distribution"
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

# ---------------- ABOUT PAGE ----------------
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### 📰 Hindi News Sentiment Analysis & Classification System
    (same content you wrote…)
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center><small>Hindi News Sentiment Analysis | Open Source Lab</small></center>",
            unsafe_allow_html=True)
