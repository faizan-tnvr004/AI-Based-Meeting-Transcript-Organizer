"""
Meeting Sentence Classifier - Flask Backend API
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import nltk

# Download NLTK data (only needed once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─── Load TensorFlow lazily to avoid slow import at top ───
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ─── Load saved model, vectorizer, and label encoder ───
print("Loading model and assets...")
model = tf.keras.models.load_model('meeting_classifier_model.keras')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✅ All assets loaded!")

# ─── Text cleaning (same as notebook) ───
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub(r'^[A-Za-z][\w\s\.]*:\s*', '', text)
    text = text.lower()
    text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'\(\d{1,2}:\d{2}(?:am|pm)?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(cleaned)


# ─── Routes ───

@app.route('/')
def home():
    return jsonify({"status": "Meeting Classifier API is running!", "version": "1.0"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "We decided to use Python for the backend." }
    Returns: { "label": "Decision", "confidence": 92.3, "probabilities": {...} }
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please send JSON with a 'text' field"}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    # Clean and vectorize
    cleaned = clean(text)
    if not cleaned:
        return jsonify({"error": "Text became empty after cleaning — try a longer sentence"}), 400

    vec = vectorizer.transform([cleaned]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    prob_dict = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(label_encoder.classes_, probs)
    }

    return jsonify({
        "label": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "cleaned_text": cleaned
    })


if __name__ == '__main__':
    print("\n Starting Meeting Classifier API...")
    print("   → Single prediction:  POST http://localhost:5000/predict")
    print("   → Batch prediction:   POST http://localhost:5000/predict-batch")
    print("   → Open frontend:      index.html in your browser\n")
    app.run(debug=True, port=5000)

    @app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    POST /predict-batch
    Body: { "sentences": ["sentence 1", "sentence 2", ...] }
    Returns list of predictions.
    """
    data = request.get_json()
    if not data or 'sentences' not in data:
        return jsonify({"error": "Please send JSON with a 'sentences' list"}), 400

    sentences = data['sentences']
    if not isinstance(sentences, list) or len(sentences) == 0:
        return jsonify({"error": "'sentences' must be a non-empty list"}), 400

    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            results.append({"error": "empty sentence"})
            continue
        cleaned = clean(sentence)
        if not cleaned:
            results.append({"sentence": sentence, "error": "too short after cleaning"})
            continue
        vec = vectorizer.transform([cleaned]).toarray()
        probs = model.predict(vec, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.classes_[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        prob_dict = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(label_encoder.classes_, probs)
        }
        results.append({
            "sentence": sentence,
            "label": pred_label,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict
        })

    return jsonify({"results": results, "total": len(results)})