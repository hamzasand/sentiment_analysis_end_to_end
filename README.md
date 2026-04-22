# 🚀 Sentiment Analysis API (ML + DL)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 📌 Overview

This project is a **production-ready Sentiment Analysis API** that integrates:

- 🧠 **Deep Learning** → LSTM-based model (context-aware)
- 🤖 **Machine Learning** → TF-IDF + classical models (fast & efficient)
- ⚡ **FastAPI** → high-performance REST API for real-time inference

It supports **dual-model inference**, allowing users to choose between speed (ML) and contextual understanding (DL).

---

## ✨ Features

- 🔄 Dual prediction pipelines (ML + DL)
- ⚡ FastAPI-based REST endpoints
- 📦 Pre-trained model artifacts
- 🧪 Ready for production deployment
- 📊 Performance comparison included
- 🔌 Easy integration with frontend / services

---

## 🏗️ Project Structure


src/
├── api/
│ └── main.py
├── artifacts/
│ ├── sentiment_lstm.h5
│ ├── tokenizer.pkl
│ ├── label_encoder.pkl
│ ├── best_sentiment_model.pkl
│ ├── tfidf_vectorizer.pkl


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/hamzasand/sentiment_analysis_end_to_end.git
cd sentiment-analysis-api
2️⃣ Create Virtual Environment
python -m venv venv
Activate Environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Running the Application
uvicorn src.api.main:app --reload

Server will be available at:

👉 http://127.0.0.1:8000

Interactive API Docs:

Swagger UI → http://127.0.0.1:8000/docs
ReDoc → http://127.0.0.1:8000/redoc
📡 API Endpoints
🟢 Health Check

GET /

{
  "message": "Sentiment API (LSTM + ML) is running"
}
🧠 LSTM Prediction

POST /predict-lstm

Request
{
  "text": "I love this product!"
}
Response
{
  "model": "LSTM",
  "text": "I love this product!",
  "sentiment": "positive"
}
🤖 ML Prediction

POST /predict-ml

Request
{
  "text": "This is a very bad experience"
}
Response
{
  "model": "ML (TF-IDF)",
  "text": "This is a very bad experience",
  "sentiment": "negative"
}
📊 Model Performance
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
📈 ML vs DL Comparison
Feature	ML Models	LSTM (DL)
Accuracy	~88.75%	~89%
Training Speed	Fast	Slower
Context Learning	Limited	Strong
Feature Handling	Manual (TF-IDF)	Automatic
🏁 Conclusion
⚡ ML models → faster and lightweight
🧠 LSTM model → better context understanding
🎯 Choose based on your use case:
High-speed systems → ML
Better NLP understanding → DL
📌 Use Cases
📱 Social media sentiment analysis
🛍️ Product review classification
💬 Customer feedback systems
🤖 Chat sentiment monitoring
📊 Business intelligence dashboards
🧪 Future Improvements
Add Transformer models (BERT, DistilBERT)
Docker containerization
CI/CD with GitHub Actions
Model versioning (MLflow)
Streaming inference support
🤝 Contributing

Contributions are welcome!

Steps:
Fork the repository
Create a new branch
git checkout -b feature/your-feature-name
Commit changes
git commit -m "Add new feature"
Push branch
git push origin feature/your-feature-name
Open Pull Request
📝 License

This project is licensed under the MIT License.

👨‍💻 Author

Muhammad Hamza
AI/ML Engineer | NLP | Computer Vision | LLMs
