import numpy as np
import pickle
import re
import joblib
import os

from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Path setup (FIXED)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # src/
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# -------------------------------
# Load LSTM model
# -------------------------------
lstm_model = load_model(os.path.join(ARTIFACTS_DIR, "sentiment_lstm.h5"))

with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# -------------------------------
# Load ML model (TF-IDF)
# -------------------------------
ml_model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl"))

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI()

# -------------------------------
# Request schema
# -------------------------------
class TextRequest(BaseModel):
    text: str

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -------------------------------
# Root
# -------------------------------
@app.get("/")
def home():
    return {"message": "Sentiment API (LSTM + ML) is running"}

# -------------------------------
# LSTM Endpoint
# -------------------------------
@app.post("/predict-lstm")
def predict_lstm(request: TextRequest):

    text = clean_text(request.text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)

    pred = lstm_model.predict(padded)[0]

    label_index = np.argmax(pred)
    confidence = float(np.max(pred))

    label = le.inverse_transform([label_index])[0]

    return {
        "model": "LSTM",
        "text": request.text,
        "sentiment": label,
        "confidence": round(confidence, 3)
    }

# -------------------------------
# ML Endpoint
# -------------------------------
@app.post("/predict-ml")
def predict_ml(request: TextRequest):

    text = clean_text(request.text)

    vec = tfidf.transform([text])
    pred = ml_model.predict(vec)[0]

    return {
        "model": "ML (TF-IDF)",
        "text": request.text,
        "sentiment": str(pred)
    }