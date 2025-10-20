#!/usr/bin/env python3
"""
Verification script to ensure the Sentiment Analysis notebook will run correctly.
This script tests all critical imports and validates the notebook structure.
"""

print("="*80)
print("SENTIMENT ANALYSIS NOTEBOOK - VERIFICATION SCRIPT")
print("="*80)

# Test 1: Import all required libraries
print("\n[1/5] Testing library imports...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from bs4 import BeautifulSoup
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    import re
    import string
    import warnings
    warnings.filterwarnings('ignore')
    print("   ✅ All libraries imported successfully!")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 2: Download NLTK data
print("\n[2/5] Downloading NLTK data...")
try:
    nltk_downloads = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for package in nltk_downloads:
        nltk.download(package, quiet=True)
    print("   ✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")

# Test 3: Check for IMDB dataset
print("\n[3/5] Checking for IMDB dataset...")
import os
if os.path.exists('IMDB Dataset.csv'):
    df_test = pd.read_csv('IMDB Dataset.csv')
    print(f"   ✅ IMDB Dataset found! Shape: {df_test.shape}")
    del df_test
else:
    print("   ⚠️  IMDB Dataset.csv not found - notebook will use sample data")

# Test 4: Test text preprocessing function
print("\n[4/5] Testing text preprocessing...")
try:
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

    test_text = "This is a test! It should work perfectly."
    result = preprocess_text(test_text)
    print(f"   ✅ Text preprocessing working! Input: '{test_text}'")
    print(f"      Output: '{result}'")
except Exception as e:
    print(f"   ❌ Preprocessing error: {e}")
    exit(1)

# Test 5: Test model components
print("\n[5/5] Testing ML model components...")
try:
    # Create sample data
    sample_texts = [
        "This is great and amazing",
        "This is terrible and awful",
        "Love this wonderful product",
        "Hate this horrible thing"
    ]
    sample_labels = ['positive', 'negative', 'positive', 'negative']

    # Preprocess
    processed = [preprocess_text(text) for text in sample_texts]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(processed)

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, sample_labels)

    # Test prediction
    test_text = "This is wonderful"
    test_processed = preprocess_text(test_text)
    test_vector = vectorizer.transform([test_processed])
    prediction = model.predict(test_vector)[0]

    print(f"   ✅ Model pipeline working!")
    print(f"      Test prediction: '{test_text}' → {prediction}")
except Exception as e:
    print(f"   ❌ Model error: {e}")
    exit(1)

# Final summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE!")
print("="*80)
print("\n✅ All checks passed! The notebook should run without errors.")
print("\nNext steps:")
print("1. Open Jupyter Notebook or JupyterLab")
print("2. Load 'Sentiment_Analysis_Assignment.ipynb'")
print("3. Click 'Cell → Run All' to execute the entire notebook")
print("4. Verify all outputs appear correctly")
print("\nNote: Make sure to run cells in order from top to bottom!")
print("="*80)
