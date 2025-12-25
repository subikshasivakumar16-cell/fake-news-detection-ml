import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.sample(frac=1, random_state=42)

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text'])
y = data['label']

print("TF-IDF feature matrix shape:", X.shape)
