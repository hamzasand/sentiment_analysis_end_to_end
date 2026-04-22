# 🚀 Sentiment Analysis API (ML + DL)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 📌 Overview

This project is an **end-to-end Sentiment Analysis system** that combines:

- 🧠 Deep Learning (LSTM)
- 🤖 Machine Learning (TF-IDF + Classical ML Models)

It is deployed using **FastAPI** and provides real-time sentiment prediction via REST APIs.

---

## ⚙️ Project Structure


src/
├── api/
│ ├── main.py
├── artifacts/
│ ├── sentiment_lstm.h5
│ ├── tokenizer.pkl
│ ├── label_encoder.pkl
│ ├── best_sentiment_model.pkl
│ ├── tfidf_vectorizer.pkl


---

## 🚀 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-api.git
cd sentiment-analysis-api
2️⃣ Install Dependencies
pip install fastapi uvicorn numpy pandas scikit-learn tensorflow joblib
3️⃣ Run Server
uvicorn main:app --reload

Server will run at:

http://127.0.0.1:8000
📡 API Endpoints
🟢 Home
GET /

Check API status.

{
  "message": "Sentiment API (LSTM + ML) is running"
}
🧠 LSTM Prediction
POST /predict-lstm
Request:
{
  "text": "I love this product!"
}
Response:
{
  "model": "LSTM",
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.94
}
🤖 ML Prediction
POST /predict-ml
Request:
{
  "text": "This is very bad experience"
}
Response:
{
  "model": "ML (TF-IDF)",
  "text": "This is very bad experience",
  "sentiment": "negative"
}
📊 Model Performance Comparison
🤖 Machine Learning Models
Model	Accuracy
Logistic Regression	85.62%
Naive Bayes	88.75%
Linear SVM	88.12%
Random Forest	83.75%

👉 Best ML Model: Naive Bayes

🧠 Deep Learning (LSTM)
Metric	Score
Accuracy	0.89
Precision	0.89
Recall	0.88
F1-score	0.89
Class-wise Performance
Class	Precision	Recall	F1-score
Negative	0.78	0.92	0.85
Neutral	1.00	0.97	0.98
Positive	0.88	0.76	0.81
📈 Final Comparison
Feature	ML Models	LSTM (DL)
Accuracy	~88.75%	~89%
Training Speed	Fast	Slower
Context Learning	Limited	Strong
Feature Handling	TF-IDF Manual	Auto Embeddings
🏁 Conclusion
🔹 ML models are efficient and fast.
🔹 LSTM performs better in understanding context.
🔹 Best choice depends on use case:
⚡ ML → Fast production systems
🧠 DL → Better accuracy & context understanding
📌 Use Cases
Social media sentiment analysis
Product review classification
Customer feedback systems
Chat sentiment monitoring
👨‍💻 Author

Muhammad Hamza
AI/ML Engineer | Computer Vision | NLP | LLMs