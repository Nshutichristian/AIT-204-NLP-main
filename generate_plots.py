#!/usr/bin/env python
"""
Generate all sentiment analysis plots and save as image files
Run this script to create standalone plot images
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SENTIMENT ANALYSIS - PLOT GENERATION")
print("="*80)

# Load dataset
print("\n1. Loading IMDB Dataset...")
df = pd.read_csv('IMDB Dataset.csv')
print(f"   ✅ Loaded {len(df):,} reviews")

# Add text statistics
df['review_length'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()

# ============================================================================
# PLOT 1: Sentiment Distribution (6 subplots)
# ============================================================================
print("\n2. Creating Sentiment Distribution Plots...")
fig = plt.figure(figsize=(16, 10))

# 1. Sentiment Count Bar Plot
ax1 = plt.subplot(2, 3, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71' if x == 'positive' else '#e74c3c' for x in sentiment_counts.index]
sentiment_counts.plot(kind='bar', ax=ax1, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sentiment', fontsize=12)
ax1.set_ylabel('Number of Reviews', fontsize=12)
ax1.tick_params(axis='x', rotation=0)
for i, v in enumerate(sentiment_counts):
    ax1.text(i, v + 500, str(v), ha='center', fontweight='bold')

# 2. Sentiment Pie Chart
ax2 = plt.subplot(2, 3, 2)
sentiment_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c'], startangle=90,
                      explode=(0.05, 0.05), shadow=True)
ax2.set_title('Sentiment Proportion', fontsize=14, fontweight='bold')
ax2.set_ylabel('')

# 3. Review Length Distribution by Sentiment
ax3 = plt.subplot(2, 3, 3)
df.boxplot(column='review_length', by='sentiment', ax=ax3, patch_artist=True)
ax3.set_title('Review Length by Sentiment', fontsize=14, fontweight='bold')
ax3.set_xlabel('Sentiment', fontsize=12)
ax3.set_ylabel('Review Length (characters)', fontsize=12)
plt.suptitle('')

# 4. Word Count Distribution
ax4 = plt.subplot(2, 3, 4)
df.boxplot(column='word_count', by='sentiment', ax=ax4, patch_artist=True)
ax4.set_title('Word Count by Sentiment', fontsize=14, fontweight='bold')
ax4.set_xlabel('Sentiment', fontsize=12)
ax4.set_ylabel('Word Count', fontsize=12)
plt.suptitle('')

# 5. Review Length Histogram
ax5 = plt.subplot(2, 3, 5)
for sentiment in ['positive', 'negative']:
    data = df[df['sentiment'] == sentiment]['review_length']
    ax5.hist(data, bins=30, alpha=0.6, label=sentiment, edgecolor='black')
ax5.set_title('Distribution of Review Lengths', fontsize=14, fontweight='bold')
ax5.set_xlabel('Review Length (characters)', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
ax5.legend()

# 6. Word Count Histogram
ax6 = plt.subplot(2, 3, 6)
for sentiment in ['positive', 'negative']:
    data = df[df['sentiment'] == sentiment]['word_count']
    ax6.hist(data, bins=20, alpha=0.6, label=sentiment, edgecolor='black')
ax6.set_title('Distribution of Word Counts', fontsize=14, fontweight='bold')
ax6.set_xlabel('Word Count', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.legend()

plt.tight_layout()
plt.savefig('plot1_sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plot1_sentiment_distribution.png")
plt.close()

# ============================================================================
# Train a quick model for confusion matrix and metrics
# ============================================================================
print("\n3. Training model for performance plots...")

# Simple preprocessing (basic version for speed)
df_sample = df.sample(n=5000, random_state=42)  # Use sample for speed
X = df_sample['review']
y = df_sample['sentiment']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.8, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("   ✅ Model trained")

# ============================================================================
# PLOT 2: Confusion Matrix
# ============================================================================
print("\n4. Creating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            cbar_kws={'label': 'Count'},
            square=True,
            linewidths=2,
            linecolor='black')

plt.title('Confusion Matrix - Sentiment Analysis\n', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')

# Add percentage annotations
for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                ha='center', va='center', fontsize=10, color='darkred')

plt.tight_layout()
plt.savefig('plot2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plot2_confusion_matrix.png")
plt.close()

# ============================================================================
# PLOT 3: Performance Metrics (4 subplots)
# ============================================================================
print("\n5. Creating Performance Metrics Plots...")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Metrics Bar Chart
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]
colors_list = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax1.bar(metrics, scores, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylim([0, 1.1])
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax1.axhline(y=0.9, color='red', linestyle='--', label='90% Threshold', alpha=0.5)
ax1.legend()
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Per-Class Metrics
from sklearn.metrics import classification_report
ax2 = axes[0, 1]
report_dict = classification_report(y_test, y_pred, output_dict=True)
classes = ['negative', 'positive']
metric_types = ['precision', 'recall', 'f1-score']
x = np.arange(len(classes))
width = 0.25
for i, metric in enumerate(metric_types):
    values = [report_dict[cls][metric] for cls in classes]
    ax2.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(classes)
ax2.legend()
ax2.set_ylim([0, 1.1])

# 3. Prediction Distribution
ax3 = axes[1, 0]
prediction_counts = pd.Series(y_pred).value_counts()
true_counts = pd.Series(y_test).value_counts()
x = np.arange(len(prediction_counts))
width = 0.35
ax3.bar(x - width/2, true_counts.values, width, label='True Labels', alpha=0.8, color='green')
ax3.bar(x + width/2, prediction_counts.values, width, label='Predictions', alpha=0.8, color='blue')
ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('True Labels vs Predictions', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(prediction_counts.index)
ax3.legend()

# 4. Model Confidence Distribution
ax4 = axes[1, 1]
confidence_scores = np.max(y_pred_proba, axis=1)
ax4.hist(confidence_scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
ax4.axvline(x=confidence_scores.mean(), color='red', linestyle='--',
           label=f'Mean: {confidence_scores.mean():.3f}', linewidth=2)
ax4.legend()

plt.tight_layout()
plt.savefig('plot3_performance_metrics.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plot3_performance_metrics.png")
plt.close()

# ============================================================================
# PLOT 4: Feature Importance
# ============================================================================
print("\n6. Creating Feature Importance Plots...")

# Get feature importance from model coefficients
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top positive sentiment features
top_positive_indices = coefficients.argsort()[-15:][::-1]
top_positive_features = feature_names[top_positive_indices]
top_positive_coefs = coefficients[top_positive_indices]

# Top negative sentiment features
top_negative_indices = coefficients.argsort()[:15]
top_negative_features = feature_names[top_negative_indices]
top_negative_coefs = coefficients[top_negative_indices]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Positive features
ax1.barh(range(len(top_positive_features)), top_positive_coefs, color='green', alpha=0.7)
ax1.set_yticks(range(len(top_positive_features)))
ax1.set_yticklabels(top_positive_features)
ax1.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Positive Sentiment Words', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

# Negative features
ax2.barh(range(len(top_negative_features)), top_negative_coefs, color='red', alpha=0.7)
ax2.set_yticks(range(len(top_negative_features)))
ax2.set_yticklabels(top_negative_features)
ax2.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax2.set_title('Top 15 Negative Sentiment Words', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('plot4_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plot4_feature_importance.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. plot1_sentiment_distribution.png  - 6 visualizations of data distribution")
print("  2. plot2_confusion_matrix.png        - Model performance confusion matrix")
print("  3. plot3_performance_metrics.png     - 4 performance metric visualizations")
print("  4. plot4_feature_importance.png      - Top positive/negative words")
print("\nModel Performance:")
print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  • F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("\n" + "="*80)
print("✅ All plots saved! You can now view them or include them in your report.")
print("="*80)
