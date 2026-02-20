import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load dataset
df = pd.read_csv('datasets/sms_dataset.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text

df['message'] = df['message'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Naive Bayes works best for SMS spam detection
model = MultinomialNB()
model.fit(X, y)

# Save
with open('pickle/sms_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('pickle/sms_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("SMS model trained and saved successfully!")
