# Deployment Guide: Sentiment Analyzer Web App

Complete guide to deploying your sentiment analyzer to Streamlit Cloud

---

## 🎯 Deployment Options

| Platform | Difficulty | Cost | Best For |
|----------|-----------|------|----------|
| **Streamlit Cloud** | ⭐ Easy | Free | Quick demos, class projects |
| **Heroku** | ⭐⭐ Medium | Free tier | More control, custom domains |
| **AWS/GCP** | ⭐⭐⭐ Hard | Pay-as-you-go | Production apps, scaling |
| **Hugging Face Spaces** | ⭐ Easy | Free | ML apps, community sharing |

**Recommended for this activity**: Streamlit Cloud (easiest and free!)

---

## 🚀 Option 1: Streamlit Cloud (Recommended)

### Step 1: Prepare Your Repository

1. **Ensure all required files are present**:
   ```
   your-repo/
   ├── streamlit_app.py          ← Main app file
   ├── sentiment_scale_analyzer.py ← Model logic
   ├── requirements.txt           ← Dependencies
   └── README.md                  ← (Optional) Documentation
   ```

2. **Verify requirements.txt**:
   ```txt
   torch>=2.0.0
   transformers>=4.30.0
   datasets>=2.12.0
   scikit-learn>=1.2.0
   numpy>=1.24.0
   accelerate>=0.20.0
   streamlit>=1.28.0
   plotly>=5.18.0
   pandas>=2.0.0
   ```

3. **Test locally first**:
   ```bash
   streamlit run streamlit_app.py
   ```
   - Make sure it works without errors
   - Test all features
   - Try different inputs

### Step 2: Push to GitHub

1. **Initialize Git** (if not already done):
   ```bash
   cd /path/to/your/project
   git init
   ```

2. **Add files**:
   ```bash
   git add streamlit_app.py
   git add sentiment_scale_analyzer.py
   git add requirements.txt
   git add README.md
   ```

3. **Commit**:
   ```bash
   git commit -m "Add sentiment analyzer app"
   ```

4. **Create GitHub repository**:
   - Go to https://github.com
   - Click "New repository"
   - Name: `sentiment-analyzer`
   - Choose Public
   - Don't initialize with README (you have one)
   - Click "Create repository"

5. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/sentiment-analyzer.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io
   - Click "Sign up" or "Sign in"
   - Choose "Continue with GitHub"

2. **Create new app**:
   - Click "New app" button
   - Select your repository: `sentiment-analyzer`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Wait for deployment** (2-5 minutes):
   - Streamlit will:
     - Install dependencies from requirements.txt
     - Download the DistilBERT model (~250MB)
     - Launch your app
   - You can watch the logs in real-time

4. **Access your app**:
   - You'll get a URL like: `https://your-username-sentiment-analyzer.streamlit.app`
   - This URL is public and shareable!

### Step 4: Customize Your App URL (Optional)

1. In Streamlit Cloud dashboard, click on your app
2. Click "Settings" ⚙️
3. Go to "General"
4. Customize the URL subdomain
5. Click "Save"

### Step 5: Update Your App

When you make changes:

```bash
# Make your changes to streamlit_app.py or other files
git add .
git commit -m "Update: improved UI"
git push origin main
```

Streamlit Cloud will automatically detect changes and redeploy!

---

## 🎨 Option 2: Hugging Face Spaces

Hugging Face Spaces is another great free option, especially for ML apps.

### Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Name: `sentiment-analyzer`
   - License: Apache 2.0
   - SDK: Streamlit
   - Click "Create Space"

### Step 2: Upload Files

1. **Clone the space**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analyzer
   cd sentiment-analyzer
   ```

2. **Copy your files**:
   ```bash
   cp /path/to/streamlit_app.py .
   cp /path/to/sentiment_scale_analyzer.py .
   cp /path/to/requirements.txt .
   ```

3. **Create app.py** (Hugging Face expects this name):
   ```bash
   mv streamlit_app.py app.py
   ```

4. **Push to Hugging Face**:
   ```bash
   git add .
   git commit -m "Add sentiment analyzer"
   git push
   ```

5. **Access your app**:
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analyzer`

---

## 🐳 Option 3: Docker Deployment (Advanced)

For those who want full control or need to deploy on custom infrastructure.

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t sentiment-analyzer .

# Run container
docker run -p 8501:8501 sentiment-analyzer
```

### Deploy to Cloud

- **AWS ECS/Fargate**: Upload to ECR, create task definition
- **Google Cloud Run**: `gcloud run deploy`
- **Azure Container Instances**: `az container create`

---

## 🔧 Troubleshooting

### Issue: Deployment Fails with "ModuleNotFoundError"

**Cause**: Missing dependency in requirements.txt

**Solution**:
1. Add the missing module to requirements.txt
2. Test locally: `pip install -r requirements.txt`
3. Push changes to GitHub
4. Streamlit Cloud will redeploy

### Issue: "Out of Memory" Error

**Cause**: Model is too large for free tier

**Solution**:
1. Use a smaller model:
   ```python
   model_name = "distilbert-base-uncased"  # Already small
   # OR try even smaller:
   model_name = "prajjwal1/bert-tiny"
   ```

2. Enable model caching:
   ```python
   @st.cache_resource
   def load_model():
       # Your model loading code
   ```

### Issue: App is Very Slow

**Cause**: Model loading on every request

**Solution**: Use `@st.cache_resource` decorator (already in template)

```python
@st.cache_resource
def load_model():
    # This will only run once and cache the result
    return model, tokenizer
```

### Issue: App Times Out During Deployment

**Cause**: First-time model download takes long

**Solution**:
1. Be patient (can take 5-10 minutes first time)
2. Check logs for progress
3. If it fails, try deploying again

### Issue: Git Push Fails

**Cause**: Authentication or repository issues

**Solution**:
```bash
# Configure Git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use personal access token instead of password
# GitHub Settings → Developer Settings → Personal Access Tokens
```

---

## 📊 Monitoring Your Deployed App

### Streamlit Cloud Analytics

1. Go to your app on Streamlit Cloud
2. Click "Analytics" tab
3. View:
   - Number of visitors
   - Usage over time
   - Resource consumption

### Add Google Analytics (Optional)

In `streamlit_app.py`:

```python
# Add to the end of your app
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR_GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR_GA_ID');
</script>
""", unsafe_allow_html=True)
```

---

## 🔒 Security Best Practices

### 1. Don't Commit Secrets

Create `.gitignore`:
```
.env
*.pyc
__pycache__/
.streamlit/secrets.toml
```

### 2. Use Environment Variables

For API keys or secrets:

```python
import os
api_key = os.environ.get("API_KEY")
```

In Streamlit Cloud:
- Go to App Settings
- Click "Secrets"
- Add your secrets in TOML format

### 3. Rate Limiting (Advanced)

Prevent abuse:

```python
import streamlit as st
from datetime import datetime, timedelta

if 'last_request' not in st.session_state:
    st.session_state.last_request = datetime.now()

# Allow only 1 request per second
if (datetime.now() - st.session_state.last_request).seconds < 1:
    st.warning("Please wait a moment before analyzing again.")
    st.stop()

st.session_state.last_request = datetime.now()
```

---

## 🎓 Deployment Checklist

Before deploying, ensure:

- [ ] App runs locally without errors
- [ ] All dependencies in requirements.txt
- [ ] Code is committed to GitHub
- [ ] README.md explains what the app does
- [ ] No secrets/API keys in code
- [ ] Model caching is enabled
- [ ] UI is user-friendly
- [ ] Error handling is in place
- [ ] Loading states are shown
- [ ] App is tested on mobile (responsive)

---

## 📚 Additional Resources

### Streamlit
- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Deployment
- [Streamlit Cloud Limits](https://docs.streamlit.io/streamlit-cloud/get-started#streamlit-cloud-limitations)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Docker for Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

### Git & GitHub
- [GitHub Docs](https://docs.github.com)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

---

## 🌟 Tips for a Great Deployment

1. **Add a README**: Explain what your app does
2. **Use emojis**: Make it visually appealing 🎨
3. **Add examples**: Help users understand how to use it
4. **Show loading states**: Let users know something is happening
5. **Handle errors gracefully**: Don't crash on bad input
6. **Make it mobile-friendly**: Many users will access on phones
7. **Add your name**: Take credit for your work! 🎉

---

## 🎉 You're Done!

Your sentiment analyzer is now live and accessible to anyone with the URL!

**Share your app**:
- Post the URL in class discussion
- Share on social media
- Add to your portfolio
- Show to friends and family

**Next steps**:
- Collect user feedback
- Add new features
- Improve the UI
- Try deploying other ML models

---

**Congratulations on your deployment! 🚀**
