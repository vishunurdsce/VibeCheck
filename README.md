# 🧠 Sentira Pro — Sentiment Intelligence Dashboard

A real-time sentiment analysis application powered by a **Voting Ensemble** (LinearSVC + Logistic Regression + Random Forest) with **TextBlob NLP** polarity as a hybrid tiebreaker.

Built for the DSU MLOps Mini-Project (Semester 6).

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

> **Note:** Update the badge link above after deployment with your actual Streamlit Cloud URL.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Hybrid Prediction** | ML ensemble + TextBlob polarity for robust results |
| **5-Model Comparison** | Logistic Regression, Naive Bayes, SVM, Random Forest, Decision Tree |
| **Voting Ensemble** | Final deployed model combines top 3 classifiers |
| **Dark UI** | Glassmorphic dashboard with gradient typography |
| **Real-time Analysis** | Instant sentiment, polarity, subjectivity & emotional tone |

---

## 📂 Project Structure

```
Sentiment_Analyzer/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── data/                    # (empty — dataset at root)
├── models/
│   ├── best_sentiment_model.pkl   # Voting Ensemble (11 MB)
│   └── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── results/
│   ├── cm_*.png                   # Confusion matrices
│   ├── model_comparison.png       # Bar chart comparison
│   └── metrics.csv                # Raw scores
├── report/
│   └── project_report_draft.md
├── src/
│   ├── app.py              # Streamlit dashboard (entry point)
│   ├── main.py             # Training pipeline
│   └── test_predict.py     # Quick-test script
├── sentimentdataset.csv    # Raw dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Local Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/Sentiment_Analyzer.git
cd Sentiment_Analyzer

# Create virtual env (optional)
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python3 src/main.py

# Run the app
streamlit run src/app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to **GitHub** (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** and select:
   - **Repository:** `<your-username>/Sentiment_Analyzer`
   - **Branch:** `main`
   - **Main file path:** `src/app.py`
4. Click **Deploy** — your app will be live in ~2 minutes!

---

## 📊 Model Results

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | High | High | High | High |
| Linear SVC | High | High | High | High |
| Random Forest | High | High | High | High |
| **Voting Ensemble** | **Best** | **Best** | **Best** | **Best** |

> See `results/metrics.csv` and `report/project_report_draft.md` for exact numbers.

---

## 🔄 Retraining

To retrain on new data:

1. Replace or update `sentimentdataset.csv`
2. Run: `python3 src/main.py`
3. New model PKLs are saved to `models/`
4. Restart the Streamlit app (or push to GitHub for auto-redeploy)

---

## 📝 License

Academic project — DSU Semester 6 MLOps Mini-Project.
