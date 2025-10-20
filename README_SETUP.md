# Sentiment Analysis Project - Setup Guide

## Project Overview
This project performs sentiment analysis on the IMDB Movie Reviews dataset using Natural Language Processing (NLP) and Machine Learning techniques. It meets all requirements for the Data Analytics assignment.

## ✅ Assignment Requirements Checklist

- ✅ Import required modules (numpy, pandas, matplotlib, seaborn, BeautifulSoup, nltk)
- ✅ Import IMDB Dataset (50,000 movie reviews)
- ✅ Descriptive statistical analysis
- ✅ Handle missing values
- ✅ Store data in DataFrame
- ✅ Count sentiment categories (positive/negative)
- ✅ Visualize findings with plots
- ✅ Remove punctuation
- ✅ Remove stop words
- ✅ Use TfidfVectorizer for feature extraction
- ✅ Binary classification with Logistic Regression
- ✅ 80:20 train-test split
- ✅ Fit and train the model
- ✅ Compute accuracy score
- ✅ Make predictions on new text
- ✅ Create confusion matrix
- ✅ Performance metrics visualization
- ✅ Problem statement documentation
- ✅ Algorithm explanation
- ✅ Analysis of findings
- ✅ References (12 academic sources)

## 📁 Project Structure

```
AIT-204-NLP-main/
├── Sentiment_Analysis_Assignment.ipynb  # Main notebook (RECOMMENDED)
├── sentiment_analysis.ipynb             # Alternative simpler version
├── IMDB Dataset.csv                     # Dataset (50,000 reviews)
├── verify_notebook.py                   # Verification script
├── README_SETUP.md                      # This file
└── venv/                                # Virtual environment
```

## 🚀 Quick Start Guide

### Option 1: Using Virtual Environment (Recommended)

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

2. **Install NLTK (if not already installed):**
   ```bash
   pip install nltk
   ```

3. **Verify setup:**
   ```bash
   python verify_notebook.py
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Sentiment_Analysis_Assignment.ipynb
   ```

5. **Run all cells:**
   - In Jupyter, click **Cell → Run All**
   - Or run each cell sequentially with **Shift+Enter**

### Option 2: Using Anaconda/Conda

1. **Create a new environment:**
   ```bash
   conda create -n sentiment-analysis python=3.10
   conda activate sentiment-analysis
   ```

2. **Install required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn beautifulsoup4 nltk scikit-learn jupyter
   ```

3. **Follow steps 3-5 from Option 1**

## 📦 Required Packages

All packages are already installed in the `venv/` directory:

- numpy >= 2.3.3
- pandas >= 2.3.3
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- beautifulsoup4 >= 4.14.2
- nltk (latest)
- scikit-learn >= 1.7.2
- jupyter

## 📊 Dataset Information

**IMDB Movie Reviews Dataset**
- **Size:** 50,000 reviews
- **File:** `IMDB Dataset.csv` (66 MB)
- **Columns:** `review`, `sentiment`
- **Classes:** positive, negative (balanced 25,000 each)
- **Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset is already in the project directory and will be loaded automatically.

## 🔧 Troubleshooting

### Issue: "IMDB Dataset.csv not found"
**Solution:** The notebook will automatically create a sample dataset. For full analysis, download from Kaggle (link above).

### Issue: "ModuleNotFoundError: No module named 'nltk'"
**Solution:**
```bash
source venv/bin/activate
pip install nltk
```

### Issue: "Resource punkt_tab not found"
**Solution:** The notebook automatically downloads this. If it fails, run:
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Cells show NameError
**Solution:** Make sure to run cells in order from top to bottom. Click **Cell → Run All** to ensure proper execution order.

### Issue: Plots not displaying
**Solution:** Add this to the first cell:
```python
%matplotlib inline
```

## 📝 Running the Notebook

### Step-by-Step Instructions:

1. **Open the notebook:**
   ```bash
   jupyter notebook Sentiment_Analysis_Assignment.ipynb
   ```

2. **Restart kernel and clear all outputs:**
   - Click **Kernel → Restart & Clear Output**

3. **Run all cells:**
   - Click **Cell → Run All**
   - Wait for all cells to execute (may take 2-5 minutes for full dataset)

4. **Verify outputs:**
   - Check that all cells show outputs
   - Verify plots are displayed
   - Ensure no error messages appear

### Expected Execution Time:
- **With sample data (400 reviews):** ~30 seconds
- **With full IMDB dataset (50,000 reviews):** ~2-5 minutes

## 🎯 What Each Section Does

1. **Cells 1-3:** Documentation and problem statement
2. **Cell 4:** Import all required libraries
3. **Cells 5-6:** Load IMDB dataset
4. **Cells 7-12:** Exploratory data analysis and visualization
5. **Cells 13-14:** Text preprocessing (remove punctuation, stop words)
6. **Cells 15-16:** TF-IDF feature extraction
7. **Cells 17-18:** Train-test split (80:20)
8. **Cells 19-22:** Model training and accuracy evaluation
9. **Cells 23-24:** Predictions on new text
10. **Cells 25-32:** Model evaluation (confusion matrix, metrics, visualizations)
11. **Cells 33-34:** Analysis of findings and references

## ✨ Key Features

- **Comprehensive preprocessing:** HTML removal, punctuation removal, stop word removal, lemmatization
- **TF-IDF vectorization:** Converts text to numerical features
- **Logistic Regression:** Binary classification (positive/negative)
- **Multiple visualizations:** Sentiment distribution, confusion matrix, performance metrics, feature importance
- **Detailed analysis:** Model strengths, limitations, recommendations
- **Academic references:** 12 properly cited sources

## 🎓 Academic Writing

The notebook includes:
- ✅ Problem statement
- ✅ Algorithm explanation with workflow diagram
- ✅ Comprehensive analysis of findings
- ✅ 12 academic references in proper format
- ✅ Professional technical writing
- ✅ Clear code comments throughout

## 📤 Submission Checklist

Before submitting, ensure:
- [ ] All cells execute without errors
- [ ] All visualizations display correctly
- [ ] Accuracy scores are printed
- [ ] Confusion matrix is displayed
- [ ] Feature importance analysis shows top words
- [ ] Analysis section is complete
- [ ] References are included

## 🆘 Support

If you encounter any issues:

1. **Run the verification script:**
   ```bash
   python verify_notebook.py
   ```

2. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Reinstall packages:**
   ```bash
   pip install --upgrade numpy pandas matplotlib seaborn nltk scikit-learn beautifulsoup4
   ```

## 📚 Additional Resources

- **NLTK Documentation:** https://www.nltk.org/
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/

## ✅ Final Verification

Run this command to verify everything works:

```bash
source venv/bin/activate && python verify_notebook.py
```

Expected output:
```
✅ All checks passed! The notebook should run without errors.
```

---

**Note:** This project fully meets all assignment requirements. The main notebook (`Sentiment_Analysis_Assignment.ipynb`) is comprehensive, well-documented, and ready for submission.

**Good luck with your assignment! 🎉**
