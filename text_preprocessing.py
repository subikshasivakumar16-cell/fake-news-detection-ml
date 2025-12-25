import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.sample(frac=1, random_state=42)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove symbols & numbers
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

# Show result
print(data[['text', 'clean_text']].head())
