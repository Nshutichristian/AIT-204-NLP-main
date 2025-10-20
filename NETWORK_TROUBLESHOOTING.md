# Network Connectivity Troubleshooting

## Current Issue
The training scripts are unable to connect to `huggingface.co` to download:
1. IMDB dataset
2. DistilBERT pre-trained model and tokenizer

**Error**: `ReadTimeoutError: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out`

## Possible Causes

### 1. Firewall or Network Restrictions
- Corporate/school firewall blocking Hugging Face
- VPN interference
- Proxy configuration issues

### 2. Internet Connectivity
- Slow or unstable internet connection
- DNS resolution issues
- SSL/TLS certificate problems

### 3. Hugging Face Service
- Temporary service outage (rare)
- Rate limiting

## Solutions to Try

### Option 1: Use Mobile Hotspot
```bash
# Connect to mobile hotspot and try again
python imdb_sentiment_analysis.py
```

### Option 2: Use Different Network
```bash
# Try from home network or different WiFi
python imdb_sentiment_demo.py
```

### Option 3: Manual Download (When Access Available)

#### Download Dataset Manually:
```python
# On a machine with access, run:
from datasets import load_dataset
dataset = load_dataset("imdb")
dataset.save_to_disk("./imdb_dataset")

# Then copy ./imdb_dataset folder to your machine
```

#### Download Model Manually:
```python
# On a machine with access, run:
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

tokenizer.save_pretrained("./distilbert_local")
model.save_pretrained("./distilbert_local")

# Then copy ./distilbert_local folder to your machine
```

#### Use Local Files:
```python
# Modify scripts to use local files:
# dataset = load_from_disk("./imdb_dataset")
# tokenizer = AutoTokenizer.from_pretrained("./distilbert_local")
# model = AutoModelForSequenceClassification.from_pretrained("./distilbert_local")
```

### Option 4: Configure Proxy (If Applicable)
```bash
# Set proxy environment variables
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
export HF_HUB_DOWNLOAD_TIMEOUT=300

python imdb_sentiment_analysis.py
```

### Option 5: Use Offline Mode (After Initial Download)
```python
# Modify scripts to use offline mode:
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
```

### Option 6: Try Alternative Models (Smaller Download)
```python
# Use a smaller model that might download faster:
model_name = "prajjwal1/bert-tiny"  # Only 4.4MB
# OR
model_name = "google/bert_uncased_L-2_H-128_A-2"  # 5MB
```

### Option 7: Increase Timeout
```python
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes
```

### Option 8: Contact IT/Network Admin
If on corporate/school network:
- Request access to `huggingface.co`
- Request access to `cdn.huggingface.co`
- Whitelist SSL certificate for *.huggingface.co

## Verification Steps

### Check Internet Connection:
```bash
ping huggingface.co
```

### Test HTTPS Access:
```bash
curl -I https://huggingface.co
```

### Check DNS Resolution:
```bash
nslookup huggingface.co
```

### Test Python Requests:
```python
import requests
response = requests.get("https://huggingface.co", timeout=30)
print(response.status_code)  # Should be 200
```

## Alternative: Run on Google Colab

If local network issues persist, use Google Colab:

1. Go to: https://colab.research.google.com
2. Create new notebook
3. Copy the code from `imdb_sentiment_analysis.py`
4. Run in Colab (has internet access to Hugging Face)

```python
# In Google Colab:
!pip install transformers datasets torch scikit-learn

# Then paste and run the training code
```

## What Works Currently

Even without network access, you have:
1. ✅ Complete code implementation
2. ✅ Comprehensive README with explanations
3. ✅ Detailed transformer architecture explanation
4. ✅ Code structure and methodology
5. ✅ Understanding of how transformers work

## Expected Results (When Network Works)

Once you can download the model and dataset, expect:
- **Training Time**: 1-3 hours for full dataset
- **Accuracy**: 92-94%
- **Model Size**: ~250MB
- **Dataset Size**: ~80MB

## Quick Test When Network Works

```bash
# Test with small subset first (5-10 minutes):
# Uncomment lines 74-75 in imdb_sentiment_analysis.py:
# dataset['train'] = dataset['train'].select(range(1000))
# dataset['test'] = dataset['test'].select(range(500))

python imdb_sentiment_analysis.py
```

---

**Next Steps**: Try the solutions above when you have access to a different network or contact your network administrator to whitelist Hugging Face.
