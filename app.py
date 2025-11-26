import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import re

# Page config
st.set_page_config(
    page_title="Hindi News Analyzer",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Hindi News Sentiment Analysis")

# Load models once (cached)
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

@st.cache_resource
def load_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

# Load models
sentiment_model = load_sentiment_model()
classifier = load_zero_shot_classifier()

# Categories we want to classify into
CATEGORIES = ['Politics', 'Sports', 'Entertainment', 'Business', 'Technology', 'Social Issues']

def clean_text(text):
    """Simple text cleaning"""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def analyze_sentiment(text):
    """Get sentiment"""
    try:
        result = sentiment_model(text[:512])  # Limit tokens
        return result[0]
    except:
        return {"label": "NEUTRAL", "score": 0.0}

def classify_news(text):
    """Classify into category"""
    try:
        result = classifier(text[:512], CATEGORIES, multi_class=False)
        return {
            "category": result['labels'][0],
            "score": result['scores'][0]
        }
    except:
        return {"category": "Unknown", "score": 0.0}

# Sidebar navigation
page = st.sidebar.radio(
    "Choose Mode",
    ["Single Analysis", "Batch Processing", "Dashboard", "About"]
)

# ============ SINGLE ANALYSIS PAGE ============
if page == "Single Analysis":
    st.header("Analyze Single News Article")
    
    # Input
    news_text = st.text_area(
        "Paste Hindi news text here:",
        height=150,
        placeholder="Enter your Hindi news..."
    )
    
    if st.button("🔍 Analyze", use_container_width=True):
        if news_text.strip():
            cleaned = clean_text(news_text)
            
            # Get results
            sentiment = analyze_sentiment(cleaned)
            category = classify_news(cleaned)
            
            # Display
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

# ============ BATCH PROCESSING PAGE ============
elif page == "Batch Processing":
    st.header("Analyze Multiple Articles")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file (must have 'text' column)",
        type=['csv']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.write(f"Loaded {len(df)} articles")
        st.dataframe(df.head(3), use_container_width=True)
        
        if st.button("🚀 Analyze All", use_container_width=True):
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
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                "results.csv",
                "text/csv",
                use_container_width=True
            )

# ============ DASHBOARD PAGE ============
elif page == "Dashboard":
    st.header("Analytics Dashboard")
    
    # Create sample data for demo (or load from uploaded file)
    st.info("Upload a CSV file in 'Batch Processing' first, then come back here to see live analytics")
    
    # Demo data
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

# ============ ABOUT PAGE ============
else:  # page == "About"
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
    - **Category Classification**: Zero-shot classification (no training needed!)
    - **No Custom Training**: Everything is ready-to-use
    
    ---
    
    ### Key Features
    
    ✨ **Real-time Analysis** - Instant results with confidence scores
    
    ✨ **Batch Processing** - Analyze multiple articles at once
    
    ✨ **Analytics Dashboard** - Visual charts and trends
    
    ✨ **Download Results** - Export as CSV
    
    ---
    
    ### Use Cases
    
    - **Researchers**: Study public sentiment on topics
    - **News Orgs**: Auto-tag and monitor articles
    - **Content Creators**: Track article performance
    - **Businesses**: Monitor brand sentiment in news
    
    ---
    
    ### Technical Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python with Transformers
    - **NLP Models**: BERT (sentiment), BART (classification)
    - **Deployment**: Streamlit Cloud
    
    ---
    
    ### Team
    
    **Jaypee Institute of Information Technology, Noida**
    
    - Shreya Bhardwaj (23103122)
    - Ronak Koul (23103126)
    - S.M. Farhan (23103132)
    
    **Course**: Open Source Software Lab (15B17CI575)
    
    **Instructor**: Rohit Kumar Sony
    
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
