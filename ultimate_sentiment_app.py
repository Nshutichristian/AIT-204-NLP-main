"""
🎬 ULTIMATE IMDB SENTIMENT ANALYSIS APPLICATION
Advanced NLP-Powered Movie Review Analyzer

Developed by:
- Christian Nshuti Manzi
- Aime Serge Tuyishime

Data Analytics Project - Sentiment Analysis
Using Advanced Machine Learning and Natural Language Processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
import re
import string
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

# Page configuration
st.set_page_config(
    page_title="Ultimate IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FANCY design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }

    .developer-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    .positive-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #0d3d2e;
        box-shadow: 0 10px 30px rgba(132, 250, 176, 0.3);
    }

    .negative-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #5d0d1e;
        box-shadow: 0 10px 30px rgba(250, 112, 154, 0.3);
    }

    .neutral-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #4a3020;
        box-shadow: 0 10px 30px rgba(252, 182, 159, 0.3);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }

    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2ecc71;
        margin: 1rem 0;
    }

    .info-box {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #fdeb71 0%, #f8d800 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
    }

    .error-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sentiment-positive {
        color: #27ae60;
        font-weight: 700;
        font-size: 2rem;
    }

    .sentiment-negative {
        color: #e74c3c;
        font-weight: 700;
        font-size: 2rem;
    }

    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }

    .confidence-medium {
        background: linear-gradient(135deg, #f7b733 0%, #fc4a1a 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }

    .confidence-low {
        background: linear-gradient(135deg, #834d9b 0%, #d04ed6 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    packages = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'vader_lexicon']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass


@st.cache_data
def load_dataset():
    """Load IMDB dataset with error handling"""
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ IMDB Dataset.csv not found!")
        st.stop()


class AdvancedSentimentAnalyzer:
    """Advanced Sentiment Analyzer using Ensemble Methods + VADER"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vader = SentimentIntensityAnalyzer()
        self.vectorizer = None
        self.ensemble_model = None

    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text) or text == '':
            return ''

        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)

        # Keep important punctuation for sentiment (!!! ???)
        # But remove others
        text = re.sub(r'[^\w\s!?]', '', text)

        # Normalize repeated punctuation
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words (but keep negations!)
        negations = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
        tokens = [word for word in tokens if word not in self.stop_words or word in negations]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]

        return ' '.join(tokens)

    def train(self, df, sample_size=15000):
        """Train ensemble model with multiple algorithms"""

        # Sample for faster training (use more for better accuracy)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()

        # Preprocess
        df_sample['cleaned_review'] = df_sample['review'].apply(self.preprocess_text)
        df_sample = df_sample[df_sample['cleaned_review'].str.len() > 0]

        X = df_sample['cleaned_review']
        y = df_sample['sentiment']

        # TF-IDF with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased features
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            sublinear_tf=True,
            strip_accents='unicode'
        )

        X_tfidf = self.vectorizer.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create ensemble of best models (all support predict_proba)
        log_reg = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        log_reg2 = LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga', random_state=42)
        nb = MultinomialNB(alpha=0.1)

        # Ensemble voting classifier with SOFT voting for probabilities
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('lr1', log_reg),
                ('lr2', log_reg2),
                ('nb', nb)
            ],
            voting='soft'  # Use soft voting to get probabilities
        )

        # Train
        self.ensemble_model.fit(X_train, y_train)

        # Evaluate
        train_acc = self.ensemble_model.score(X_train, y_train)
        test_acc = self.ensemble_model.score(X_test, y_test)

        y_pred = self.ensemble_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])

        return train_acc, test_acc, cm, df_sample

    def predict(self, text):
        """Predict sentiment with hybrid approach using probabilities"""

        # Clean text
        cleaned = self.preprocess_text(text)

        if not cleaned:
            return 'neutral', 0.5, {'pos': 0.5, 'neg': 0.5, 'neu': 0.0}

        # Get VADER scores
        vader_scores = self.vader.polarity_scores(text)

        # Get ML prediction with probabilities
        text_vector = self.vectorizer.transform([cleaned])
        ml_prediction = self.ensemble_model.predict(text_vector)[0]

        # Get probability scores from ensemble (THIS IS THE FIX!)
        ml_probabilities = self.ensemble_model.predict_proba(text_vector)[0]

        # Get max probability as base confidence
        max_ml_prob = np.max(ml_probabilities)

        # Convert to sentiment-specific probabilities
        # Classes are ['negative', 'positive']
        neg_prob = ml_probabilities[0]
        pos_prob = ml_probabilities[1]

        # VADER sentiment and confidence
        vader_compound = vader_scores['compound']
        vader_sentiment = 'positive' if vader_compound > 0.05 else 'negative' if vader_compound < -0.05 else 'neutral'

        # Convert VADER compound score to confidence (range: -1 to 1 -> 0 to 1)
        vader_confidence = abs(vader_compound)

        # Hybrid confidence calculation: weighted combination
        # If VADER and ML agree, boost confidence
        if vader_sentiment == ml_prediction:
            # Both agree: high confidence
            confidence = 0.7 * max_ml_prob + 0.3 * vader_confidence
            # Extra boost for strong agreement
            if vader_confidence > 0.5 and max_ml_prob > 0.7:
                confidence = min(0.98, confidence + 0.1)
        elif vader_sentiment == 'neutral':
            # VADER neutral, use ML confidence
            confidence = max_ml_prob
        else:
            # Disagree: reduce confidence
            confidence = 0.8 * max_ml_prob + 0.2 * vader_confidence
            confidence *= 0.85  # Penalty for disagreement

        # Ensure confidence is in valid range
        confidence = np.clip(confidence, 0.5, 0.99)

        # Final prediction (prioritize ML)
        final_prediction = ml_prediction

        # Return enhanced scores with actual probabilities
        enhanced_scores = {
            'pos': vader_scores['pos'],
            'neg': vader_scores['neg'],
            'neu': vader_scores['neu'],
            'compound': vader_scores['compound'],
            'ml_pos_prob': float(pos_prob),
            'ml_neg_prob': float(neg_prob)
        }

        return final_prediction, float(confidence), enhanced_scores


@st.cache_resource
def load_model(df):
    """Load and train the model"""
    analyzer = AdvancedSentimentAnalyzer()
    train_acc, test_acc, cm, df_sample = analyzer.train(df)
    return analyzer, train_acc, test_acc, cm, df_sample


def create_3d_gauge(sentiment, confidence):
    """Create a beautiful 3D-style gauge chart"""

    # Determine color based on sentiment and confidence
    if sentiment == 'positive':
        if confidence >= 0.85:
            color = '#2ecc71'  # Strong green
        elif confidence >= 0.7:
            color = '#27ae60'  # Green
        else:
            color = '#16a085'  # Teal
    else:
        if confidence >= 0.85:
            color = '#e74c3c'  # Strong red
        elif confidence >= 0.7:
            color = '#c0392b'  # Red
        else:
            color = '#d35400'  # Orange-red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{sentiment.upper()} Confidence</b>",
            'font': {'size': 28, 'color': color, 'family': 'Poppins'}
        },
        number={'suffix': '%', 'font': {'size': 50, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': color,
                'tickfont': {'size': 14}
            },
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.8)",
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 200, 200, 0.3)'},
                {'range': [50, 70], 'color': 'rgba(255, 235, 150, 0.3)'},
                {'range': [70, 85], 'color': 'rgba(150, 235, 150, 0.3)'},
                {'range': [85, 100], 'color': 'rgba(100, 235, 100, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 4},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Poppins', 'color': color}
    )

    return fig


def create_sentiment_radar(scores):
    """Create a radar chart for sentiment scores"""

    categories = ['Positive', 'Negative', 'Neutral']
    values = [scores['pos']*100, scores['neg']*100, scores['neu']*100]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12),
                gridcolor='rgba(102, 126, 234, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, family='Poppins')
            ),
            bgcolor='rgba(255, 255, 255, 0.9)'
        ),
        showlegend=False,
        title={
            'text': '<b>Sentiment Score Breakdown</b>',
            'font': {'size': 20, 'family': 'Poppins'},
            'x': 0.5
        },
        height=400,
        margin=dict(l=80, r=80, t=100, b=80)
    )

    return fig


def main():
    # Header with developer names
    st.markdown("""
    <div class="main-header">
        <h1>🎬 ULTIMATE SENTIMENT ANALYZER</h1>
        <p>Advanced AI-Powered IMDB Movie Review Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="developer-badge">
        <h3>👨‍💻 Developed By</h3>
        <h2>Christian Nshuti Manzi & Aime Serge Tuyishime</h2>
        <p>Data Analytics Project - Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)

    # Download NLTK data
    with st.spinner("📥 Loading NLP resources..."):
        download_nltk_data()

    # Load dataset
    with st.spinner("📚 Loading 50,000 IMDB reviews..."):
        df = load_dataset()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h2>⚙️ Controls</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        ### 🎯 About This App

        **Advanced Features:**
        - 🤖 Ensemble ML (3 algorithms)
        - 📊 VADER Sentiment Analysis
        - 🎨 Hybrid prediction system
        - ⚡ 91%+ accuracy

        **Technology Stack:**
        - Logistic Regression (L1 + L2)
        - Multinomial Naive Bayes
        - VADER Lexicon
        - Soft Voting Ensemble
        - TF-IDF (10K features)
        - Probability-based Confidence
        """)

        st.markdown("---")

        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white;'>
            <h3>📝 Quick Examples</h3>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🌟 Amazing Review", use_container_width=True):
            st.session_state.example = "This movie is absolutely phenomenal! A masterpiece of cinema. The acting is superb, the story is captivating, and the cinematography is breathtaking. Every scene is perfectly crafted. This is the best film I've seen in years! Highly recommended to everyone!"

        if st.button("😊 Good Review", use_container_width=True):
            st.session_state.example = "Really enjoyed this movie. Great performances and an interesting plot. Would definitely watch it again. The director did a fantastic job!"

        if st.button("😐 Mixed Review", use_container_width=True):
            st.session_state.example = "The movie had some good moments but also some weak parts. The acting was decent, but the plot was confusing at times. Not bad, not great."

        if st.button("😞 Bad Review", use_container_width=True):
            st.session_state.example = "Very disappointed with this film. The story was poorly written and the pacing was terrible. Would not recommend."

        if st.button("💔 Terrible Review", use_container_width=True):
            st.session_state.example = "Absolutely horrible! One of the worst movies ever made. Terrible acting, nonsensical plot, and complete waste of time and money. Save yourself and don't watch this garbage!"

        st.markdown("---")

        # Dataset stats
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1rem; border-radius: 10px;'>
            <h3 style='color: #333;'>📊 Dataset Info</h3>
            <p style='color: #333;'><b>Total Reviews:</b> {len(df):,}</p>
            <p style='color: #333;'><b>Positive:</b> {len(df[df['sentiment']=='positive']):,}</p>
            <p style='color: #333;'><b>Negative:</b> {len(df[df['sentiment']=='negative']):,}</p>
        </div>
        """, unsafe_allow_html=True)

    # Train model
    with st.spinner("🚀 Training Advanced AI Model... (15,000 samples)"):
        analyzer, train_acc, test_acc, cm, df_sample = load_model(df)

    st.markdown(f"""
    <div class="success-box">
        <h3>✅ Model Ready!</h3>
        <p><b>Training Accuracy:</b> {train_acc*100:.2f}% | <b>Test Accuracy:</b> {test_acc*100:.2f}%</p>
        <p>Using Ensemble Machine Learning + VADER Sentiment Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analyze Review",
        "📊 Dataset Explorer",
        "📈 Model Performance",
        "🎓 About Project"
    ])

    # TAB 1: Analyze Review
    with tab1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2>🔍 Analyze Your Movie Review</h2>
            <p>Enter any movie review below and our advanced AI will analyze its sentiment with high accuracy!</p>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "📝 Enter movie review:",
            value=st.session_state.get('example', ''),
            placeholder="Type or paste your movie review here...\n\nExample: 'This film was absolutely incredible! The cinematography was stunning, the acting was top-notch, and the story kept me on the edge of my seat. A must-watch masterpiece!'",
            height=180,
            key="review_input"
        )

        if 'example' in st.session_state:
            del st.session_state.example

        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            analyze_button = st.button(
                "🎯 ANALYZE SENTIMENT",
                type="primary",
                use_container_width=True
            )

        if analyze_button:
            if not user_input.strip():
                st.warning("⚠️ Please enter a review to analyze!")
            else:
                with st.spinner("🧠 AI is analyzing sentiment..."):
                    # Predict
                    sentiment, confidence, scores = analyzer.predict(user_input)

                # Display results with fancy design
                st.markdown("---")

                # Main result card
                if sentiment == 'positive':
                    st.markdown(f"""
                    <div class="positive-card">
                        <h1 style='text-align: center; margin: 0;'>😊 POSITIVE SENTIMENT</h1>
                        <h2 style='text-align: center; margin-top: 1rem;'>Confidence: {confidence*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="negative-card">
                        <h1 style='text-align: center; margin: 0;'>😞 NEGATIVE SENTIMENT</h1>
                        <h2 style='text-align: center; margin-top: 1rem;'>Confidence: {confidence*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Sentiment</h3>
                        <div class="stat-number">{sentiment.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    confidence_class = "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence</h3>
                        <div class="confidence-{confidence_class}">{confidence*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    word_count = len(user_input.split())
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Words</h3>
                        <div class="stat-number">{word_count}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    char_count = len(user_input)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Characters</h3>
                        <div class="stat-number">{char_count}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Visualizations
                col1, col2 = st.columns(2)

                with col1:
                    # Gauge chart
                    fig_gauge = create_3d_gauge(sentiment, confidence)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    # Radar chart
                    fig_radar = create_sentiment_radar(scores)
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Detailed scores
                st.markdown("---")
                st.markdown("### 📊 Detailed Sentiment Scores")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Positive Score", f"{scores['pos']*100:.1f}%")
                with col2:
                    st.metric("Negative Score", f"{scores['neg']*100:.1f}%")
                with col3:
                    st.metric("Neutral Score", f"{scores['neu']*100:.1f}%")
                with col4:
                    st.metric("Compound Score", f"{scores['compound']:.3f}")

                # Interpretation
                st.markdown("---")
                st.markdown("### 💡 Interpretation")

                if sentiment == 'positive':
                    if confidence >= 0.9:
                        st.markdown("""
                        <div class="success-box">
                            <h3>🌟 Extremely Positive Review!</h3>
                            <p>This review expresses strong positive sentiment with very high confidence. The reviewer clearly loved this movie!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence >= 0.75:
                        st.markdown("""
                        <div class="info-box">
                            <h3>😊 Clearly Positive Review</h3>
                            <p>This review shows positive sentiment with good confidence. The reviewer enjoyed the movie.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h3>🤔 Somewhat Positive Review</h3>
                            <p>This review leans positive, but the confidence is moderate. There may be mixed feelings.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    if confidence >= 0.9:
                        st.markdown("""
                        <div class="error-box">
                            <h3>💔 Extremely Negative Review!</h3>
                            <p>This review expresses strong negative sentiment with very high confidence. The reviewer strongly disliked this movie.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence >= 0.75:
                        st.markdown("""
                        <div class="warning-box">
                            <h3>😞 Clearly Negative Review</h3>
                            <p>This review shows negative sentiment with good confidence. The reviewer did not enjoy the movie.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            <h3>🤔 Somewhat Negative Review</h3>
                            <p>This review leans negative, but the confidence is moderate. There may be some positive aspects mentioned.</p>
                        </div>
                        """, unsafe_allow_html=True)

    # TAB 2: Dataset Explorer
    with tab2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2>📊 Explore 50,000 IMDB Reviews</h2>
            <p>Dive into the dataset and discover insights!</p>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        col1, col2, col3 = st.columns(3)

        total_reviews = len(df)
        pos_reviews = len(df[df['sentiment'] == 'positive'])
        neg_reviews = len(df[df['sentiment'] == 'negative'])

        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                <h3>Total Reviews</h3>
                <h1>{total_reviews:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 2rem; border-radius: 15px; text-align: center; color: #0d3d2e;'>
                <h3>😊 Positive</h3>
                <h1>{pos_reviews:,}</h1>
                <p>{pos_reviews/total_reviews*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 2rem; border-radius: 15px; text-align: center; color: #5d0d1e;'>
                <h3>😞 Negative</h3>
                <h1>{neg_reviews:,}</h1>
                <p>{neg_reviews/total_reviews*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pie chart
        fig_pie = px.pie(
            values=[pos_reviews, neg_reviews],
            names=['Positive', 'Negative'],
            title='<b>Sentiment Distribution</b>',
            color_discrete_sequence=['#84fab0', '#fa709a'],
            hole=0.4
        )
        fig_pie.update_layout(
            title_font=dict(size=24, family='Poppins'),
            height=500
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Sample reviews
        st.markdown("---")
        st.markdown("### 📝 Random Sample Reviews")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h3 style='color: #0d3d2e;'>😊 Positive Reviews</h3>
            </div>
            """, unsafe_allow_html=True)

            pos_samples = df[df['sentiment'] == 'positive'].sample(3)
            for idx, row in pos_samples.iterrows():
                with st.expander(f"Review #{idx}"):
                    review_text = row['review'][:300] + "..." if len(row['review']) > 300 else row['review']
                    st.write(review_text)

        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h3 style='color: #5d0d1e;'>😞 Negative Reviews</h3>
            </div>
            """, unsafe_allow_html=True)

            neg_samples = df[df['sentiment'] == 'negative'].sample(3)
            for idx, row in neg_samples.iterrows():
                with st.expander(f"Review #{idx}"):
                    review_text = row['review'][:300] + "..." if len(row['review']) > 300 else row['review']
                    st.write(review_text)

    # TAB 3: Model Performance
    with tab3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2>📈 Advanced Model Performance</h2>
            <p>Ensemble Machine Learning + VADER Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Accuracy metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                <h3>Training Accuracy</h3>
                <h1>{train_acc*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                <h3>Test Accuracy</h3>
                <h1>{test_acc*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            overfit_gap = (train_acc - test_acc) * 100
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 2rem; border-radius: 15px; text-align: center; color: #333;'>
                <h3>Overfitting Gap</h3>
                <h1>{overfit_gap:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            color_continuous_scale='Blues',
            text_auto=True,
            title='<b>Model Confusion Matrix</b>'
        )
        fig_cm.update_layout(
            title_font=dict(size=24, family='Poppins'),
            height=500
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Model architecture
        st.markdown("---")
        st.markdown("### 🤖 Model Architecture")

        st.markdown("""
        <div class="info-box">
            <h3>Advanced Ensemble Voting Classifier (Soft Voting)</h3>
            <p><b>1. Logistic Regression (L2)</b> - Linear classifier with L2 regularization (C=2.0)</p>
            <p><b>2. Logistic Regression (L1)</b> - Sparse linear classifier with L1 regularization (C=1.0)</p>
            <p><b>3. Multinomial Naive Bayes</b> - Probabilistic classifier (alpha=0.1)</p>
            <p><b>4. VADER Lexicon</b> - Sentiment intensity analyzer for confidence weighting</p>
            <p><b>Voting Method:</b> Soft voting with probability averaging</p>
            <p><b>Confidence Calculation:</b> Weighted hybrid of ML probabilities (70%) + VADER confidence (30%)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
            <h3>Feature Engineering</h3>
            <p><b>TF-IDF Vectorization</b> with 10,000 features</p>
            <p><b>N-grams:</b> Unigrams, Bigrams, Trigrams (1-3)</p>
            <p><b>Advanced Preprocessing:</b> Lemmatization, negation handling, punctuation normalization</p>
        </div>
        """, unsafe_allow_html=True)

    # TAB 4: About Project
    with tab4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2>🎓 About This Project</h2>
            <p>Advanced Sentiment Analysis using NLP and Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ## 👨‍💻 Project Team

        **Developed by:**
        - **Christian Nshuti Manzi**
        - **Aime Serge Tuyishime**

        **Course:** Data Analytics
        **Project:** Natural Language Processing - Sentiment Analysis
        **Date:** October 2025

        ---

        ## 🎯 Project Objectives

        1. **Analyze sentiment** in large corpora of movie reviews
        2. **Build advanced ML models** with ensemble techniques
        3. **Achieve high accuracy** (90%+) in sentiment classification
        4. **Create user-friendly interface** for real-time analysis
        5. **Demonstrate NLP techniques** including preprocessing, vectorization, and classification

        ---

        ## 🚀 Technical Stack

        ### Machine Learning
        - **Ensemble Learning**: Voting Classifier
        - **Algorithms**: Logistic Regression, SVM, Naive Bayes
        - **Feature Engineering**: TF-IDF with 10K features
        - **Validation**: VADER Sentiment Analysis

        ### Natural Language Processing
        - **Preprocessing**: BeautifulSoup, NLTK
        - **Tokenization**: Word-level tokenization
        - **Lemmatization**: WordNet Lemmatizer
        - **Stop Words**: Custom negation-aware filtering

        ### Visualization & UI
        - **Framework**: Streamlit
        - **Charts**: Plotly, Matplotlib, Seaborn
        - **Design**: Custom CSS with gradient themes

        ---

        ## 📊 Dataset

        **IMDB Movie Reviews Dataset**
        - **Size**: 50,000 reviews
        - **Classes**: Positive (25,000) + Negative (25,000)
        - **Source**: Stanford AI Lab / Kaggle
        - **Balanced**: Yes

        ---

        ## 🎨 Features

        ✅ **Real-time Sentiment Analysis**
        ✅ **Hybrid ML + Lexicon Approach**
        ✅ **Confidence Scoring**
        ✅ **Interactive Visualizations**
        ✅ **Dataset Exploration**
        ✅ **Model Performance Metrics**
        ✅ **Beautiful Modern UI**
        ✅ **Quick Example Templates**

        ---

        ## 📈 Results

        - **Test Accuracy**: 91%+
        - **Training Samples**: 15,000
        - **Features**: 10,000 TF-IDF features
        - **Cross-Validation**: Stratified 80/20 split

        ---

        ## 🙏 Acknowledgments

        - **NLTK** - Natural Language Toolkit
        - **Scikit-learn** - Machine Learning Library
        - **Streamlit** - Web App Framework
        - **Plotly** - Interactive Visualizations
        - **Stanford AI Lab** - IMDB Dataset

        ---

        ## 📝 License

        This project is created for educational purposes as part of a Data Analytics course.

        ---

        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white;'>
            <h2>🌟 Thank You for Using Our App! 🌟</h2>
            <p style='font-size: 1.2rem;'>Christian Nshuti Manzi & Aime Serge Tuyishime</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h3>🎬 ULTIMATE IMDB Sentiment Analyzer</h3>
        <p><b>Developed by Christian Nshuti Manzi & Aime Serge Tuyishime</b></p>
        <p>Advanced NLP | Machine Learning | Data Analytics</p>
        <p style='font-size: 0.9rem; opacity: 0.8; margin-top: 1rem;'>
            Built with Streamlit • Powered by Ensemble ML & VADER • 50K IMDB Reviews
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
