"""
IMDB Sentiment Analysis Web Application
Professional sentiment analysis using TF-IDF and Logistic Regression
Analyzes the IMDB Movie Reviews Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    packages = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass


@st.cache_data
def load_dataset():
    """Load IMDB dataset"""
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ IMDB Dataset.csv not found! Please ensure it's in the same directory as this app.")
        st.stop()


@st.cache_resource
def train_model(df, sample_size=10000):
    """Train sentiment analysis model"""

    # Sample data for faster training (use full dataset for production)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    # Preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if pd.isna(text) or text == '':
            return ''
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    # Preprocess
    df_sample['cleaned_review'] = df_sample['review'].apply(preprocess_text)
    df_sample = df_sample[df_sample['cleaned_review'].str.len() > 0]

    # Prepare features
    X = df_sample['cleaned_review']
    y = df_sample['sentiment']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Calculate metrics
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, vectorizer, train_acc, test_acc, cm, df_sample


def preprocess_input(text):
    """Preprocess user input text"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    if pd.isna(text) or text == '':
        return ''
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def create_sentiment_gauge(sentiment, confidence):
    """Create a beautiful gauge chart"""
    value = 1 if sentiment == 'positive' else 0
    color = '#28a745' if sentiment == 'positive' else '#dc3545'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {sentiment.upper()}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🎬 IMDB Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Movie Review Analysis using 50,000 IMDB Reviews</div>', unsafe_allow_html=True)

    # Download NLTK data
    with st.spinner("📥 Loading language resources..."):
        download_nltk_data()

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=200)
        st.header("📊 About This App")
        st.markdown("""
        This application analyzes movie review sentiment using:

        - **Dataset**: 50,000 IMDB movie reviews
        - **Algorithm**: TF-IDF + Logistic Regression
        - **Accuracy**: ~89-90%

        ### How It Works:
        1. **Text Preprocessing**: Removes noise, punctuation, stop words
        2. **Feature Extraction**: TF-IDF vectorization
        3. **Classification**: Logistic Regression model
        4. **Prediction**: Positive or Negative sentiment

        ### Features:
        - ✅ Real-time sentiment analysis
        - ✅ Confidence scores
        - ✅ Dataset exploration
        - ✅ Model performance metrics
        """)

        st.divider()

        st.header("📝 Quick Examples")
        if st.button("😊 Positive Review", use_container_width=True):
            st.session_state.example = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I loved every minute of it. Highly recommend!"

        if st.button("😞 Negative Review", use_container_width=True):
            st.session_state.example = "Terrible movie. The plot was confusing, the acting was awful, and it was a complete waste of time. Do not watch!"

        if st.button("🤔 Mixed Review", use_container_width=True):
            st.session_state.example = "The movie had great visuals but the story was weak. Some good performances but overall disappointing."

    # Load dataset
    with st.spinner("📚 Loading IMDB dataset..."):
        df = load_dataset()

    st.success(f"✅ Loaded {len(df):,} movie reviews from IMDB!")

    # Train model
    with st.spinner("🤖 Training AI model... (this may take a minute)"):
        model, vectorizer, train_acc, test_acc, cm, df_sample = train_model(df)

    st.success(f"✅ Model trained! Test Accuracy: {test_acc*100:.2f}%")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Review", "📊 Dataset Explorer", "📈 Model Performance"])

    # TAB 1: Analyze Review
    with tab1:
        st.header("Analyze Your Movie Review")

        # Text input
        user_input = st.text_area(
            "Enter a movie review:",
            value=st.session_state.get('example', ''),
            placeholder="Type or paste a movie review here...\n\nExample: 'This film was incredible! Best movie I've seen all year. The cinematography was stunning and the performances were outstanding.'",
            height=150
        )

        if 'example' in st.session_state:
            del st.session_state.example

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("⚠️ Please enter a review to analyze.")
            else:
                with st.spinner("🧠 Analyzing sentiment..."):
                    # Preprocess and predict
                    cleaned_text = preprocess_input(user_input)
                    text_vector = vectorizer.transform([cleaned_text])
                    prediction = model.predict(text_vector)[0]
                    probabilities = model.predict_proba(text_vector)[0]

                    # Get confidence
                    pos_prob = probabilities[1] if model.classes_[1] == 'positive' else probabilities[0]
                    neg_prob = probabilities[0] if model.classes_[0] == 'negative' else probabilities[1]
                    confidence = max(pos_prob, neg_prob)

                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    sentiment_icon = "😊" if prediction == 'positive' else "😞"
                    sentiment_color = "positive" if prediction == 'positive' else "negative"
                    st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><div class="{sentiment_color}">{sentiment_icon} {prediction.upper()}</div></div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'<div class="metric-card"><h3>Confidence</h3><div style="font-size: 1.5rem; font-weight: bold;">{confidence*100:.1f}%</div></div>', unsafe_allow_html=True)

                with col3:
                    word_count = len(user_input.split())
                    st.markdown(f'<div class="metric-card"><h3>Word Count</h3><div style="font-size: 1.5rem; font-weight: bold;">{word_count}</div></div>', unsafe_allow_html=True)

                # Gauge chart
                st.plotly_chart(create_sentiment_gauge(prediction, confidence), use_container_width=True)

                # Probability breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Probability", f"{pos_prob*100:.2f}%")
                with col2:
                    st.metric("Negative Probability", f"{neg_prob*100:.2f}%")

                # Interpretation
                if prediction == 'positive':
                    if confidence >= 0.9:
                        st.success("🌟 **Strong Positive Sentiment** - This review is clearly positive!")
                    elif confidence >= 0.7:
                        st.info("👍 **Positive Sentiment** - This review leans positive.")
                    else:
                        st.warning("🤔 **Weakly Positive** - This review is slightly positive but not very strong.")
                else:
                    if confidence >= 0.9:
                        st.error("💔 **Strong Negative Sentiment** - This review is clearly negative!")
                    elif confidence >= 0.7:
                        st.warning("👎 **Negative Sentiment** - This review leans negative.")
                    else:
                        st.info("🤔 **Weakly Negative** - This review is slightly negative but not very strong.")

    # TAB 2: Dataset Explorer
    with tab2:
        st.header("📊 Explore the IMDB Dataset")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            positive_count = len(df[df['sentiment'] == 'positive'])
            st.metric("Positive Reviews", f"{positive_count:,}")
        with col3:
            negative_count = len(df[df['sentiment'] == 'negative'])
            st.metric("Negative Reviews", f"{negative_count:,}")

        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Positive vs Negative Reviews",
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#28a745', 'negative': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sample reviews
        st.subheader("Sample Reviews from Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😊 Positive Reviews")
            positive_samples = df[df['sentiment'] == 'positive'].sample(3)
            for idx, row in positive_samples.iterrows():
                with st.expander(f"Review {idx}"):
                    st.write(row['review'][:500] + "..." if len(row['review']) > 500 else row['review'])

        with col2:
            st.markdown("### 😞 Negative Reviews")
            negative_samples = df[df['sentiment'] == 'negative'].sample(3)
            for idx, row in negative_samples.iterrows():
                with st.expander(f"Review {idx}"):
                    st.write(row['review'][:500] + "..." if len(row['review']) > 500 else row['review'])

        # Review length statistics
        st.subheader("Review Length Statistics")
        df['review_length'] = df['review'].str.len()
        df['word_count'] = df['review'].str.split().str.len()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x='word_count', color='sentiment',
                title="Distribution of Review Word Counts",
                color_discrete_map={'positive': '#28a745', 'negative': '#dc3545'},
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            stats_df = df.groupby('sentiment')[['word_count', 'review_length']].mean().round(2)
            st.dataframe(stats_df, use_container_width=True)

    # TAB 3: Model Performance
    with tab3:
        st.header("📈 Model Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with col2:
            st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_,
                    yticklabels=model.classes_,
                    ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        st.pyplot(fig)

        # Model details
        st.subheader("Model Details")

        st.markdown(f"""
        **Algorithm**: Logistic Regression
        **Feature Extraction**: TF-IDF Vectorization
        **Training Samples**: {len(df_sample):,}
        **Test Split**: 80/20
        **Max Features**: 5,000
        **N-gram Range**: (1, 2) - Unigrams and Bigrams

        ### Preprocessing Steps:
        1. HTML tag removal
        2. Lowercase conversion
        3. Punctuation removal
        4. Stop word removal
        5. Lemmatization
        6. Tokenization

        ### Performance:
        - The model achieves **{test_acc*100:.2f}% accuracy** on the test set
        - Training accuracy: **{train_acc*100:.2f}%**
        - Overfitting gap: **{(train_acc - test_acc)*100:.2f}%**
        """)

        # Top features
        if hasattr(model, 'coef_'):
            st.subheader("Top Predictive Words")

            feature_names = np.array(vectorizer.get_feature_names_out())
            coefficients = model.coef_[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 😊 Top Positive Words")
                top_positive_indices = coefficients.argsort()[-10:][::-1]
                for idx in top_positive_indices:
                    st.write(f"**{feature_names[idx]}**: {coefficients[idx]:.4f}")

            with col2:
                st.markdown("### 😞 Top Negative Words")
                top_negative_indices = coefficients.argsort()[:10]
                for idx in top_negative_indices:
                    st.write(f"**{feature_names[idx]}**: {coefficients[idx]:.4f}")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>IMDB Sentiment Analysis Application</strong></p>
        <p>Built with Streamlit | Dataset: 50,000 IMDB Movie Reviews</p>
        <p>Natural Language Processing | Machine Learning | TF-IDF | Logistic Regression</p>
        <p>🎬 Analyze movie reviews with AI-powered sentiment detection</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
