# Solutions Guide

**⚠️ Important**: Try to complete all TODOs on your own first! Only check these solutions if you're truly stuck after:
1. Reading the hints in the code
2. Reviewing `STUDY_GUIDE.md`
3. Researching the concept online
4. Spending at least 15 minutes trying

Struggling is part of learning! 🧠

---

## 📝 Table of Contents

1. [Part 2: Sentiment Scale Analyzer Solutions](#part-2-sentiment-scale-analyzer)
2. [Part 3: Streamlit App Solutions](#part-3-streamlit-app)
3. [Complete Code Examples](#complete-code-examples)
4. [Extension Challenge Solutions](#extension-challenges)

---

## Part 2: Sentiment Scale Analyzer

### TODO 1: Modify Model Configuration

**Location**: `sentiment_scale_analyzer.py` (around line 63)

**Question**: How many labels do we need for -3 to +3 scale?

**SOLUTION**:
```python
num_labels = 7  # 7 classes for -3 to +3 scale
```

**Explanation**:
- Scale from -3 to +3 includes: {-3, -2, -1, 0, 1, 2, 3}
- Count them: That's 7 distinct values
- So we need 7 output classes from our model

**Why this matters**: The model's output layer must match the number of classes we're predicting.

---

### TODO 2: Create Dataset with 7 Labels

**Location**: `sentiment_scale_analyzer.py` (Function: `create_training_data()`)

**Task**: Add examples for all 7 sentiment levels

**SOLUTION**:

```python
def create_training_data():
    """Create labeled training data for 7-point scale"""

    training_data = [
        # Very Negative (-3) → Class 0
        ("This is the worst movie I have ever seen in my entire life!", 0),
        ("Absolutely terrible! Complete waste of time and money.", 0),
        ("Horrible film. I want my money back. Awful in every way.", 0),
        ("Unwatchable garbage. Save yourself the agony.", 0),
        ("This movie is an insult to cinema. Appalling!", 0),

        # Negative (-2) → Class 1
        ("This movie was quite disappointing and boring.", 1),
        ("Not good. Poor acting and weak plot.", 1),
        ("I expected better. Very underwhelming and dull.", 1),
        ("Disappointing on all fronts. Skip this one.", 1),
        ("Below par. Wouldn't recommend to anyone.", 1),

        # Slightly Negative (-1) → Class 2
        ("The movie had potential but didn't deliver.", 2),
        ("It was below average, not terrible but not good either.", 2),
        ("Not quite worth the ticket price. Mediocre film.", 2),
        ("Could have been better. Fell short of expectations.", 2),
        ("Somewhat disappointing. Has some issues.", 2),

        # Neutral (0) → Class 3
        ("It was okay, nothing special. Just average.", 3),
        ("The film was fine. Neither good nor bad.", 3),
        ("Acceptable movie. Nothing to write home about.", 3),
        ("Standard fare. Decent enough to watch once.", 3),
        ("Middle of the road. Some good, some bad.", 3),

        # Slightly Positive (+1) → Class 4
        ("Pretty decent movie. I enjoyed it overall.", 4),
        ("Good film with some nice moments.", 4),
        ("Enjoyable enough. Worth watching if you have time.", 4),
        ("Pleasant surprise. Better than I expected.", 4),
        ("Not bad at all. Had some good parts.", 4),

        # Positive (+2) → Class 5
        ("Really great movie! I thoroughly enjoyed it.", 5),
        ("Excellent film with strong performances.", 5),
        ("Highly entertaining. Well worth watching!", 5),
        ("Very good film with compelling storytelling.", 5),
        ("Impressed me. Great production and acting.", 5),

        # Very Positive (+3) → Class 6
        ("Absolutely amazing! Best movie I've seen this year!", 6),
        ("Masterpiece! Incredible storytelling and perfect execution.", 6),
        ("This movie was phenomenal! A true work of art!", 6),
        ("Outstanding in every way! Instant classic!", 6),
        ("Brilliant! One of the best films ever made!", 6),
    ]

    return training_data
```

**Key Points**:
- Each sentiment level has multiple examples (at least 3-5)
- Examples clearly represent their intensity level
- Language/words match the sentiment (e.g., "awful" for -3, "decent" for +1)
- Balanced across all 7 classes

**Tips for creating your own**:
- Think of real reviews you've read
- Use sentiment-appropriate vocabulary
- Make them realistic and varied
- Balance positive and negative examples

---

### TODO 3: Implement Class-to-Score Conversion

**Location**: `sentiment_scale_analyzer.py` (Function: `class_to_sentiment_score()`)

**Question**: Convert class_id (0-6) to sentiment score (-3 to +3)

**SOLUTION**:

```python
def class_to_sentiment_score(class_id):
    """
    Convert class ID (0-6) to sentiment score (-3 to +3)

    Args:
        class_id: Integer from 0 to 6
    Returns:
        sentiment_score: Integer from -3 to +3
    """
    sentiment_score = class_id - 3
    return sentiment_score
```

**Why this works**:
```
class_id:     0   1   2   3   4   5   6
class_id - 3: -3  -2  -1  0   1   2   3  ← sentiment score
```

**Alternative Solutions** (all correct):

```python
# Solution 1: Simple arithmetic (best)
sentiment_score = class_id - 3

# Solution 2: Using mapping dictionary
mapping = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}
sentiment_score = mapping[class_id]

# Solution 3: Using list indexing
scores = [-3, -2, -1, 0, 1, 2, 3]
sentiment_score = scores[class_id]
```

**Most Efficient**: Solution 1 (simple arithmetic - O(1) time, O(1) space)

**Concept**: This is a **linear transformation** - we're shifting the range from [0, 6] to [-3, 3] by subtracting the offset (3).

---

## Part 3: Streamlit App

### TODO 1: Add Title and Description

**Location**: `streamlit_app.py` (around line 113)

**SOLUTION**:

```python
st.title("🎬 Movie Review Sentiment Analyzer")

st.markdown("""
**Analyze movie reviews on a scale from -3 (Very Negative) to +3 (Very Positive)**

This app uses a fine-tuned DistilBERT transformer model to analyze sentiment with nuance,
going beyond simple positive/negative classification.

🎯 Perfect for understanding how audiences *really* feel about films!
""")
```

**Creative Alternatives**:

```python
# Option 1: Simple and clear
st.title("🎭 Sentiment Analyzer: The 7-Point Scale")
st.write("Analyze movie reviews with precision - from hate to love!")

# Option 2: More detailed
st.title("🌟 Advanced Sentiment Analysis")
st.markdown("""
### Beyond Thumbs Up/Down
This analyzer uses AI to measure sentiment on a 7-point scale,
capturing the full spectrum of human emotion about movies.
""")

# Option 3: Fun and engaging
st.title("🍿 Movie Review Decoder")
st.markdown("**How did you *really* feel about that movie?**")
st.caption("We'll analyze it with AI on a scale from 😢 to 🤩")
```

**Tips**:
- Use emojis to make it engaging
- Clearly explain what the app does
- Keep it concise but informative
- Make it your own!

---

### TODO 2: Create Text Input Area

**Location**: `streamlit_app.py` (around line 148)

**SOLUTION**:

```python
review_text = st.text_area(
    "📝 Enter a movie review to analyze:",
    value=st.session_state.get('example_text', ''),
    placeholder="Type or paste a movie review here...\n\nExample: 'This movie was incredible! The acting was superb and the plot kept me engaged throughout.'",
    height=150,
    key="review_input"
)
```

**Explanation**:
- **Label** (`"📝 Enter..."`): Tells user what to do
- **value**: Uses session_state to populate from examples
- **placeholder**: Shows example text when empty
- **height**: 150 pixels for comfortable typing
- **key**: Unique identifier for this widget

**Customization Ideas**:

```python
# Shorter version
review_text = st.text_area(
    "Your movie review:",
    placeholder="What did you think of the movie?",
    height=120
)

# With more guidance
review_text = st.text_area(
    "📝 Share your honest movie review:",
    placeholder="Tell us what you loved, liked, or didn't like about the film...",
    height=180,
    help="The more detailed your review, the better our analysis!"
)
```

---

### TODO 3: Display Results with Visualization

**Location**: `streamlit_app.py` (around line 194)

**SOLUTION**:

```python
# Visual sentiment indicator with colors
if sentiment_score >= 2:
    st.success(f"✅ **{sentiment_label}** {sentiment_emoji}")
elif sentiment_score >= 1:
    st.info(f"👍 **{sentiment_label}** {sentiment_emoji}")
elif sentiment_score == 0:
    st.warning(f"➖ **{sentiment_label}** {sentiment_emoji}")
elif sentiment_score >= -1:
    st.warning(f"👎 **{sentiment_label}** {sentiment_emoji}")
else:
    st.error(f"❌ **{sentiment_label}** {sentiment_emoji}")
```

**Enhanced Version with More Visual Elements**:

```python
# Add progress bar representation
progress_value = (sentiment_score + 3) / 6  # Normalize to 0-1
st.progress(progress_value)

# Add colored metric box
color = get_sentiment_color(sentiment_score)
st.markdown(f"""
<div style='padding: 20px; border-radius: 10px; background-color: {color}20;
            border: 2px solid {color}; text-align: center;'>
    <h2 style='color: {color};'>{sentiment_label} {sentiment_emoji}</h2>
    <p style='font-size: 24px; margin: 0;'>Score: {sentiment_score:+d}/3</p>
</div>
""", unsafe_allow_html=True)

# Add visual scale
scale_emoji = ['😢', '😞', '😐', '😶', '🙂', '😊', '🤩']
cols = st.columns(7)
for i, emoji in enumerate(scale_emoji):
    with cols[i]:
        if i == sentiment_score + 3:
            st.markdown(f"### {emoji}")
            st.markdown(f"**{i-3:+d}**")
        else:
            st.markdown(f"{emoji}")
            st.markdown(f"{i-3:+d}")
```

---

### TODO 4: Add Example Reviews

**Location**: `streamlit_app.py` (Sidebar section, around line 136)

**SOLUTION**:

```python
st.header("📝 Try These Examples")

if st.button("🤩 Very Positive Example"):
    st.session_state.example_text = "This movie was absolutely phenomenal! The best film I've seen this year. Every aspect from acting to cinematography was perfect!"

if st.button("😊 Positive Example"):
    st.session_state.example_text = "Really enjoyed this film. Great performances and a compelling story."

if st.button("🙂 Slightly Positive Example"):
    st.session_state.example_text = "Pretty good movie. Had some entertaining moments and decent acting."

if st.button("😶 Neutral Example"):
    st.session_state.example_text = "It was an okay movie. Nothing particularly special, but not bad either."

if st.button("😐 Slightly Negative Example"):
    st.session_state.example_text = "The movie had potential but didn't quite deliver. A bit disappointing."

if st.button("😞 Negative Example"):
    st.session_state.example_text = "Quite disappointing. The plot was weak and the pacing was off."

if st.button("😢 Very Negative Example"):
    st.session_state.example_text = "Absolutely terrible! Complete waste of time. One of the worst films I've ever seen."
```

**Key Points**:
- One button for each of the 7 sentiment levels
- Each button sets `st.session_state.example_text`
- Emojis make it visually clear
- Examples are clear representatives of their sentiment level

---

## Complete Code Examples

### Full sentiment_scale_analyzer.py (with solutions)

The complete working version is already in the file with solution comments. Key parts:

```python
# All TODOs solved:
num_labels = 7
training_data = create_training_data()  # Complete with all 7 classes
sentiment_score = class_id - 3  # Simple conversion formula
```

### Full streamlit_app.py (with solutions)

All TODOs are completed in the template. Run it to see the full working app!

---

## Extension Challenges

### Extension 1: Add Confidence Threshold ⭐

**Implementation**:
```python
# After getting prediction
if confidence < 0.6:
    st.warning(f"⚠️ Low confidence ({confidence:.1%}). The model is uncertain about this prediction.")
    st.info("💡 Tip: Try a longer, more detailed review for better results.")
```

---

### Extension 2: Batch Analysis ⭐⭐

**Implementation**:
```python
import pandas as pd

# In your Streamlit app
st.header("📊 Batch Analysis")
uploaded_file = st.file_uploader("Upload CSV with reviews", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'review' not in df.columns:
        st.error("CSV must have a 'review' column")
    else:
        # Analyze all reviews
        results = []
        progress_bar = st.progress(0)

        for i, review in enumerate(df['review']):
            score, conf, label = analyze_sentiment(review, model, tokenizer, device)
            results.append({
                'review': review,
                'sentiment_score': score,
                'sentiment_label': label,
                'confidence': conf
            })
            progress_bar.progress((i + 1) / len(df))

        # Display results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results",
            csv,
            "sentiment_results.csv",
            "text/csv"
        )
```

---

### Extension 3: Sentiment Distribution Chart ⭐⭐

**Implementation**:
```python
import plotly.express as px

# After batch analysis
sentiment_counts = results_df['sentiment_score'].value_counts().sort_index()

fig = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Distribution of Sentiments',
    color=sentiment_counts.index,
    color_continuous_scale='RdYlGn'
)

st.plotly_chart(fig, use_container_width=True)
```

---

### Extension 4: Compare Models ⭐⭐⭐

**Implementation**:
```python
# In sidebar
model_choice = st.selectbox(
    "Choose Model:",
    ["DistilBERT (fast)", "BERT (accurate)", "RoBERTa (powerful)"]
)

# Load different models based on choice
@st.cache_resource
def load_selected_model(choice):
    if choice == "DistilBERT (fast)":
        model_name = "distilbert-base-uncased"
    elif choice == "BERT (accurate)":
        model_name = "bert-base-uncased"
    else:  # RoBERTa
        model_name = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    return model, tokenizer

# Use in analysis
model, tokenizer = load_selected_model(model_choice)
```

---

## 💡 Learning from Solutions

### Don't Just Copy!

When you look at a solution:

1. **Understand WHY** it works
2. **Try variations** - what if you change something?
3. **Explain it** to yourself or someone else
4. **Apply the concept** to a different problem
5. **Close the file** and try to recreate it from memory

### Questions to Ask Yourself

- **TODO 1**: Why 7 and not 6? What if we used 5-point scale instead?
- **TODO 2**: How would I create examples for product reviews instead?
- **TODO 3**: What mathematical transformations exist besides linear?
- **Streamlit TODOs**: How would I reorganize the layout differently?

### Next Steps After Reviewing Solutions

1. **Close this file**
2. **Delete your TODO code**
3. **Try to implement again from scratch**
4. **Only check back if you get stuck**

This process builds real understanding!

---

**Remember**: The goal isn't to find the "right answer" - it's to develop problem-solving skills! 🚀
