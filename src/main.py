import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

# ── Exhaustive label → bucket mapping (covers every label in the dataset) ──────
POSITIVE_LABELS = {
    'positive', 'joy', 'happiness', 'happy', 'excitement', 'excited',
    'contentment', 'gratitude', 'serenity', 'hopeful', 'hope', 'awe',
    'acceptance', 'euphoria', 'admiration', 'love', 'affection', 'elation',
    'pride', 'amusement', 'enjoyment', 'nostalgia', 'enthusiasm',
    'fulfillment', 'reverence', 'empowerment', 'compassion',
    'tenderness', 'arousal', 'bliss', 'wonder', 'cheerfulness',
    'delight', 'ecstasy', 'relief'
}

NEGATIVE_LABELS = {
    'negative', 'sadness', 'sad', 'anger', 'angry', 'fear', 'disgust',
    'shame', 'bitterness', 'bitter', 'despair', 'grief', 'loneliness',
    'disappointed', 'disappointment', 'pain', 'sorrow', 'horror', 'terror',
    'hate', 'misery', 'depression', 'anxiety', 'rage', 'frustration',
    'embarrassed', 'embarrassment', 'guilt', 'jealousy', 'envy', 'regret',
    'horror', 'dread', 'melancholy', 'anguish', 'heartbreak'
}

def get_bucket(raw_label):
    """Map any dataset label to Positive / Negative / Neutral."""
    label = str(raw_label).lower().strip()
    if label in POSITIVE_LABELS:
        return 'Positive'
    if label in NEGATIVE_LABELS:
        return 'Negative'
    return 'Neutral'


def main():
    print("Loading Dataset...")
    df = pd.read_csv('sentimentdataset.csv')
    df.columns = df.columns.str.strip()
    df['Text']      = df['Text'].astype(str).str.strip()
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().apply(get_bucket)

    print(f"Dataset size: {len(df)} rows")
    print("Label distribution after bucketing:")
    print(df['Sentiment'].value_counts().to_string())

    X = df['Text']
    y = df['Sentiment']

    # ── Vectorization ──────────────────────────────────────────────────────────
    # Bigrams capture phrase context; sublinear_tf handles short-text better
    print("\nVectorizing...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=6000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Models ─────────────────────────────────────────────────────────────────
    models_dict = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', C=1.0
        ),
        'Naive Bayes': MultinomialNB(alpha=0.5),
        'Linear SVC': LinearSVC(
            max_iter=3000, class_weight='balanced', C=0.8
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight='balanced', max_depth=None
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, class_weight='balanced', max_depth=20
        ),
    }

    results = {}
    if not os.path.exists('results'): os.makedirs('results')

    CLASS_ORDER = ['Positive', 'Negative', 'Neutral']

    print("\nTraining & Evaluating 5 Models...")
    for name, model in models_dict.items():
        print(f"  → {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'Precision': round(p, 4),
            'Recall'   : round(r, 4),
            'F1 Score' : round(f1, 4),
            'Accuracy' : round(acc, 4)
        }

        # Confusion Matrix
        labels_present = [c for c in CLASS_ORDER if c in y_test.values]
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_present, yticklabels=labels_present
        )
        plt.title(f'Confusion Matrix – {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/cm_{name.replace(" ", "_")}.png', dpi=120)
        plt.close()

    # ── Comparison Chart ───────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind='bar', figsize=(13, 6), colormap='viridis')
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=120)
    plt.close()

    best_name = metrics_df['F1 Score'].idxmax()
    print(f"\nBest model by F1: {best_name}")
    print(metrics_df.to_string())

    # ── Voting Ensemble (Best for deployment) ──────────────────────────────────
    print("\nTraining Final Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('svc', LinearSVC(max_iter=3000, class_weight='balanced', C=0.8)),
            ('lr',  LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
            ('rf',  RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    _, _, ens_f1, _ = precision_recall_fscore_support(y_test, ens_pred, average='weighted', zero_division=0)
    print(f"Ensemble F1 Score: {ens_f1:.4f}")

    # ── Save Models ───────────────────────────────────────────────────────────
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(ensemble,   'models/best_sentiment_model.pkl')
    print("Models saved.")

    # ── Sanity Check ──────────────────────────────────────────────────────────
    test_cases = [
        "i feel so happy and excited today",
        "this is horrible i am so sad",
        "just a normal day nothing special",
        "i am grateful and full of love",
        "im feeling depressed and anxious",
        "the weather is fine today",
    ]
    print("\n── Sanity Check (Ensemble) ──")
    for t in test_cases:
        pred = ensemble.predict(vectorizer.transform([t]))[0]
        print(f"  {pred:10s}  | '{t}'")

    # ── Report ────────────────────────────────────────────────────────────────
    if not os.path.exists('report'): os.makedirs('report')
    report = (
        "# MLOps Sentiment Analysis – Project Report\n\n"
        "## 3.2 Performance Comparison of All Models\n\n"
        + metrics_df.to_markdown()
        + f"\n\n## 5 Conclusion\n"
          f"**Best individual model:** {best_name} (F1: {metrics_df.loc[best_name, 'F1 Score']:.4f})\n\n"
          f"**Final deployed model:** Voting Ensemble (SVC + Logistic Regression + Random Forest)\n\n"
          f"The ensemble is used for deployment as it cross-validates predictions across three "
          f"diverse classifiers, reducing individual model bias and improving robustness on unseen text.\n"
    )
    with open('report/project_report_draft.md', 'w') as f:
        f.write(report)

    print("\nDone! All outputs saved to results/, models/, report/")


if __name__ == '__main__':
    main()
