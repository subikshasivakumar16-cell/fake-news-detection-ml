from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (first run only)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    cleaned_text = clean_text(news)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    # 1 = Real, 0 = Fake
    if prediction == 1:
        result = "REAL NEWS"
        color = "green"
    else:
        result = "FAKE NEWS"
        color = "red"

    return render_template(
        "result.html",
        prediction=result,
        color=color
    )

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
