# PhishGuard — Phishing Detection Suite

> ML-powered detection for malicious URLs, phishing SMS, and phishing emails.  
> Built with Flask · Python 3 · Scikit-learn · NLTK · Bootstrap 5

---
<img width="1353" height="582" alt="PhishGuard" src="https://github.com/user-attachments/assets/da3a5d08-5393-4950-911b-34f188f10b35" />

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Tree](#directory-tree)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Conclusion](#conclusion)

---

## Introduction

The internet has become an indispensable part of daily life — but it has also enabled phishing attacks at scale. Phishers deceive victims through social engineering, fake websites, malicious SMS messages, and spoofed emails to steal credentials, financial details, and personal data.

**PhishGuard** is a multi-module phishing detection system that uses Machine Learning to identify threats across three attack vectors:

- **URLs** — structural and behavioral analysis of web links
- **SMS** — NLP-based classification of smishing messages
- **Email** — text classification for phishing email detection

---

## Features

| Module | Method | Description |
|--------|--------|-------------|
| 🔗 URL Scanner | Gradient Boosting Classifier | Extracts 30+ features from any URL and classifies it as safe or phishing |
| 💬 SMS Detector | Naive Bayes + TF-IDF | NLP model trained on real SMS datasets to detect smishing attempts |
| ✉️ Email Analyzer | Naive Bayes + TF-IDF | Classifies email body text as phishing or legitimate |

**Additional highlights:**
- Dark-themed responsive UI (Bootstrap 5)
- Confidence % shown for every prediction
- Chrome Extension button for browser integration
- No API keys or external services required — fully offline inference

---

## Installation

**Requirements:** Python 3.8+

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/phishguard.git
cd phishguard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NLTK data** (required for SMS and Email modules)
```bash
python -c "import nltk; nltk.download('stopwords')"
```

**4. Run the app**
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## Usage

### URL Detection — `/`
Paste any URL into the input field and click **Scan URL**.  
The model extracts 30 structural features (HTTPS usage, domain length, IP address usage, anchor tags, etc.) and returns a safe/unsafe verdict with confidence score.

### SMS Detection — `/sms`
Paste any suspicious SMS message and click **Analyze SMS**.  
The NLP model preprocesses the text (lowercasing, stopword removal, regex cleaning) and classifies it as `safe` or `unsafe`.

### Email Detection — `/email`
Paste the full email body and click **Analyze Email**.  
Same NLP pipeline as SMS — returns a verdict with phishing probability percentage.

---

## Directory Tree

```
PhishGuard/
├── pickle/
│   ├── model.pkl                  # URL - Gradient Boosting Classifier
│   ├── sms_model.pkl              # SMS - Naive Bayes Classifier
│   ├── sms_vectorizer.pkl         # SMS - TF-IDF Vectorizer
│   ├── email_model.pkl            # Email - Naive Bayes Classifier
│   └── email_vectorizer.pkl       # Email - TF-IDF Vectorizer
├── datasets/
│   ├── sms_dataset.csv            # SMS training data
│   └── email_dataset.csv          # Email training data
├── static/
│   └── styles.css                 # Global stylesheet
├── templates/
│   ├── index.html                 # URL scanner page
│   ├── sms.html                   # SMS detector page
│   └── email.html                 # Email analyzer page
├── app.py                         # Flask application
├── feature.py                     # URL feature extraction (30 features)
├── train_model.py                 # URL model training script
├── train_sms.py                   # SMS model training script
├── train_email.py                 # Email model training script
├── phishing.csv                   # URL training dataset
├── Procfile                       # Deployment config
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Category | Library / Tool |
|----------|---------------|
| Backend | Flask, Python 3 |
| ML / Data | Scikit-learn, NumPy, Pandas |
| NLP | NLTK, TF-IDF Vectorizer |
| Frontend | Bootstrap 5, Jinja2 |
| Model Persistence | Pickle |
| Deployment | Render / Railway (Procfile included) |

---

## Model Performance

### URL Detection — Gradient Boosting Classifier

| # | Model | Accuracy | F1 Score | Recall | Precision |
|---|-------|----------|----------|--------|-----------|
| 1 | **Gradient Boosting Classifier** ✅ | **0.974** | **0.977** | **0.994** | **0.986** |
| 2 | CatBoost Classifier | 0.972 | 0.975 | 0.994 | 0.989 |
| 3 | XGBoost Classifier | 0.969 | 0.973 | 0.993 | 0.984 |
| 4 | Multi-layer Perceptron | 0.969 | 0.973 | 0.995 | 0.981 |
| 5 | Random Forest | 0.967 | 0.971 | 0.993 | 0.990 |
| 6 | Support Vector Machine | 0.964 | 0.968 | 0.980 | 0.965 |
| 7 | Decision Tree | 0.960 | 0.964 | 0.991 | 0.993 |
| 8 | K-Nearest Neighbors | 0.956 | 0.961 | 0.991 | 0.989 |
| 9 | Logistic Regression | 0.934 | 0.941 | 0.943 | 0.927 |
| 10 | Naive Bayes | 0.605 | 0.454 | 0.292 | 0.997 |

Gradient Boosting Classifier was selected for URL detection with **97.4% accuracy**.

### SMS & Email Detection — Naive Bayes + TF-IDF
Both SMS and Email models use a Multinomial Naive Bayes classifier with TF-IDF vectorization, trained on labeled datasets with stopword removal and regex-based text cleaning.

---

## Conclusion

1. PhishGuard demonstrates that ML-based phishing detection works effectively across multiple attack vectors — URLs, SMS, and email.
2. For URL detection, features like `HTTPS`, `AnchorURL`, and `WebsiteTraffic` carry the most predictive weight.
3. Gradient Boosting Classifier achieves **97.4% accuracy** on URL classification, making it the strongest performer tested.
4. NLP-based classifiers (Naive Bayes + TF-IDF) provide fast, lightweight detection for SMS and email threats without requiring deep learning infrastructure.
5. The modular architecture makes it easy to swap or retrain individual models independently.

---

## Deployment

This project is ready to deploy on **Render** or **Railway** with zero code changes.

```bash
# Procfile (already included)
web: python app.py
```

Make sure `requirements.txt` includes all dependencies before deploying.

---

*PhishGuard — Stay safe online.*
