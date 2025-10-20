# Learning Path: Building a Multi-Scale Sentiment Analyzer
**Estimated Time**: 60-90 minutes
**Level**: Intermediate
**Learning Mode**: Self-Study with Research

---

## 🎯 What You'll Learn

1. Understand transformer-based sentiment analysis
2. Build a 7-point scale classifier (-3 to +3)
3. Create an interactive web app using Streamlit
4. Deploy the app to the cloud for free

---

## 📋 Prerequisites

- Python 3.8+
- Basic Python knowledge (functions, classes)
- Internet connection
- GitHub account (free)
- Dependencies installed: `pip install -r requirements.txt`

---

## ⏱️ Estimated Time

| Part | Time | What You'll Do |
|------|------|----------------|
| **Part 1** | 15 min | Test binary classifier & understand limitations |
| **Part 2** | 20-30 min | Build 7-point scale classifier (TODOs) |
| **Part 3** | 20-30 min | Create Streamlit web app (TODOs) |
| **Part 4** | 15-20 min | Deploy to Streamlit Cloud |

**Work at your own pace!** Research concepts as needed.

---

## 🚀 Part 1: Understanding Binary Classification (15 min)

### What You'll Discover

Binary classifiers can only output **Positive** or **Negative**. But what about:
- "It was okay" → Neutral?
- "Amazing!" vs "Good" → Different intensities?

Let's see this limitation in action!

### Run the Test

```bash
python quick_test.py
```

**Observe**:
- ✅ "Amazing movie!" → Positive (correct)
- ✅ "Terrible film!" → Negative (correct)
- ❌ "It was okay" → Must choose positive OR negative (problem!)

### Reflection

**Think about**: Can you express how you really feel about a movie with just thumbs up/down?

**Answer**: No! We need a scale: -3 (hate it) to +3 (love it)

**Research if interested**: Google "limitations of binary classification"

---

## 🔧 Part 2: Build 7-Point Scale Classifier (20-30 min)

### The Goal

Transform the classifier to output **7 sentiment levels**:

```
-3 😢  Very Negative    "Worst movie ever!"
-2 😞  Negative         "Pretty bad"
-1 😐  Slightly Negative "Could be better"
 0 😶  Neutral          "It was okay"
+1 🙂  Slightly Positive "Decent film"
+2 😊  Positive         "Really good!"
+3 🤩  Very Positive    "Masterpiece!"
```

### Open the File

```bash
# Open in your editor
code sentiment_scale_analyzer.py
# or
nano sentiment_scale_analyzer.py
```

### TODO 1: Set Number of Labels (5 min)

**Find this code** (around line 63):
```python
# TODO 1: Change num_labels to support 7-point scale
num_labels = ___  # Fill in
```

**Question**: How many sentiment levels do we have from -3 to +3?

**Hint**: Count them: -3, -2, -1, 0, 1, 2, 3

**Try it yourself first!** Then check `SOLUTIONS.md` if stuck.

---

### TODO 2: Create Training Examples (10-15 min)

**Find the function** `create_training_data()` (around line 35):

**Your task**: Add at least 2 examples for each sentiment level!

**Examples to inspire you**:
- **Very Negative (-3)**: "Absolutely terrible! Waste of money!"
- **Neutral (0)**: "It was fine, nothing special"
- **Very Positive (+3)**: "Best movie I've ever seen! Masterpiece!"

**Tips**:
- Think of real reviews you've read
- Each example should clearly match its level
- Be creative - make them realistic!

**Research if stuck**: Google "sentiment analysis example sentences 5-point scale"

---

### TODO 3: Convert Class to Score (5 min)

**Find this function** (around line 12):
```python
def class_to_sentiment_score(class_id):
    # class_id ranges from 0 to 6
    # We want -3 to +3
    sentiment_score = ___  # What formula?
    return sentiment_score
```

**Think**: What math converts:
- 0 → -3
- 3 → 0
- 6 → +3

<details>
<summary>💡 Need a hint? (Click to expand)</summary>

What if you subtract 3 from class_id?
- 0 - 3 = -3 ✓
- 3 - 3 = 0 ✓
- 6 - 3 = +3 ✓

</details>

**Try it first!** Research "linear transformation" if stuck.

---

### Test Your Code

```bash
python sentiment_scale_analyzer.py
```

**Expected output**:
```
Review: "This movie was absolutely fantastic!"
Sentiment Score: +3/3
Label: Very Positive 🤩
Confidence: 87%
```

**Not working?** Check `SOLUTIONS.md` for the complete implementation.

---

## 🎨 Part 3: Build Streamlit Web App (20-30 min)

### The Goal

Create a beautiful, interactive web interface for your sentiment analyzer!

### Learn Streamlit Basics (5 min)

**Quick tutorial**:
```python
import streamlit as st

st.title("My App")  # Big title
st.text_input("Enter text:")  # Input box
st.button("Click me")  # Button
st.success("Success!")  # Green message
```

**Research**: Spend 5 min on https://docs.streamlit.io/library/get-started

---

### Open the App File

```bash
code streamlit_app.py
```

### TODO 1: Add Title & Description (3 min)

**Find** (around line 113):
```python
# TODO 1: Add an attractive title and description
st.title("___")  # Make it catchy!
st.markdown("___")  # Explain what it does
```

**Ideas**:
- "🎬 Movie Review Sentiment Analyzer"
- "🎭 Sentiment Analysis: The 7-Point Scale"
- Make it your own!

---

### TODO 2: Create Input Area (5 min)

**Find** (around line 148):
```python
# TODO 2: Create a text area for user input
review_text = st.text_area(
    "___",  # Label - what should user do?
    placeholder="___",  # Example text
    height=___  # Try 150
)
```

**Research**: https://docs.streamlit.io/library/api-reference/widgets/st.text_area

---

### TODO 3: Visualize Results (7 min)

**Find** (around line 194):
```python
# TODO 3: Display results with colors/emojis
if sentiment_score >= 2:
    st.success(f"___")  # Very positive - green!
elif sentiment_score <= -2:
    st.error(f"___")  # Very negative - red!
else:
    st.info(f"___")  # Neutral - blue!
```

**Be creative!**:
- Add emojis
- Use different colors
- Make it engaging!

---

### TODO 4: Add Example Reviews (5 min)

**Find** (in sidebar section, around line 136):
```python
# TODO 4: Add example buttons
if st.button("Very Positive Example"):
    st.session_state.example_text = "___"  # Your example
```

**Task**: Create examples for all 7 sentiment levels!

---

### Run Locally

```bash
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501` 🎉

**Test it!** Try different reviews and see the visualizations.

**Troubleshooting**: See `NETWORK_TROUBLESHOOTING.md` if issues arise.

---

## 🚀 Part 4: Deploy to the Cloud (15-20 min)

### Why Deploy?

Make your app accessible to anyone with a URL! Great for:
- Portfolio
- Resume
- Showing friends/family
- Getting feedback

### Step 1: Push to GitHub (7 min)

**If you don't have Git set up**:
```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"
```

**Create GitHub repository**:
1. Go to https://github.com
2. Click "New repository"
3. Name: `sentiment-analyzer`
4. Make it Public
5. Click "Create"

**Push your code**:
```bash
cd "/Users/isac/Desktop/GCU COURSES/AIT-204/NLP/transformers"

git init
git add streamlit_app.py sentiment_scale_analyzer.py requirements.txt
git commit -m "Add sentiment analyzer"
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analyzer.git
git push -u origin main
```

**Research if stuck**: See `DEPLOYMENT_GUIDE.md` or Google "how to push to github"

---

### Step 2: Deploy on Streamlit Cloud (8-10 min)

1. **Go to** https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Select**:
   - Repository: `sentiment-analyzer`
   - Branch: `main`
   - Main file: `streamlit_app.py`
5. **Click** "Deploy!"

**Wait** 5-10 minutes for first deployment (downloads model)

**You'll get a URL** like: `https://your-name-sentiment-analyzer.streamlit.app`

**Detailed instructions**: See `DEPLOYMENT_GUIDE.md`

---

### Step 3: Share Your Work! (2 min)

✅ Your app is live!

**Share it**:
- LinkedIn: "Just deployed my first ML app!"
- Portfolio: Add the URL
- Resume: Link to your GitHub & live app
- Friends: Get feedback!

---

## ✅ Progress Checklist

Track what you've completed:

- [ ] Part 1: Tested binary classifier
- [ ] Part 1: Understood limitations
- [ ] Part 2: Completed TODO 1 (num_labels)
- [ ] Part 2: Completed TODO 2 (training data)
- [ ] Part 2: Completed TODO 3 (conversion formula)
- [ ] Part 2: Tested locally - it works!
- [ ] Part 3: Completed TODO 1 (title)
- [ ] Part 3: Completed TODO 2 (input area)
- [ ] Part 3: Completed TODO 3 (visualization)
- [ ] Part 3: Completed TODO 4 (examples)
- [ ] Part 3: Tested locally - looks great!
- [ ] Part 4: Pushed to GitHub
- [ ] Part 4: Deployed to Streamlit Cloud
- [ ] Part 4: Shared my app URL!

---

## 🎁 Extension Challenges

Finished early? Try these:

### ⭐ Easy: Add Confidence Warning
```python
if confidence < 0.6:
    st.warning("⚠️ Model is uncertain")
```

### ⭐⭐ Medium: Batch Analysis
Allow CSV upload and analyze multiple reviews at once

### ⭐⭐⭐ Hard: Fine-tune the Model
Actually train on IMDB dataset using `imdb_sentiment_analysis.py`

**More ideas**: Check `SOLUTIONS.md`

---

## 🐛 Troubleshooting

### Model won't download
→ Check internet connection
→ See `NETWORK_TROUBLESHOOTING.md`

### Streamlit won't start
→ Try: `streamlit run streamlit_app.py --server.port 8502`
→ Check: `streamlit --version`

### Stuck on TODO
→ Re-read the hints
→ Google the concept
→ Check `SOLUTIONS.md` (after trying!)

### Deployment fails
→ Check requirements.txt has all dependencies
→ Verify file paths
→ See `DEPLOYMENT_GUIDE.md`

---

## 📚 Learning Resources

### Included:
- `STUDY_GUIDE.md` - Concepts & strategies
- `SOLUTIONS.md` - Complete solutions
- `TRANSFORMER_EXPLANATION.md` - How transformers work
- `DEPLOYMENT_GUIDE.md` - Deployment details

### External:
- [Hugging Face Course](https://huggingface.co/course) - Free NLP course
- [Streamlit Docs](https://docs.streamlit.io) - Official guide
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Research

---

## 🎉 You Did It!

### What You Built:
- ✅ 7-point sentiment classifier
- ✅ Interactive web app
- ✅ Deployed cloud application
- ✅ Portfolio piece!

### Skills Gained:
- Multi-class classification
- Streamlit development
- Cloud deployment
- Self-directed learning

### Next Steps:
1. Try extension challenges
2. Fine-tune on more data
3. Build another ML app
4. Share your learnings!

---

**Share your app URL! You've earned it! 🚀**
