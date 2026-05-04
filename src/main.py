import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set MLflow experiment
mlflow.set_experiment("VibeCheck_Sentiment_Analysis")

# ── Exhaustive label → bucket mapping (covers every label in the dataset) ──────
POSITIVE_LABELS = {
    'positive', 'joy', 'happiness', 'happy', 'excitement', 'excited',
    'contentment', 'gratitude', 'serenity', 'hopeful', 'hope', 'awe',
    'acceptance', 'euphoria', 'admiration', 'love', 'affection', 'elation',
    'pride', 'amusement', 'enjoyment', 'nostalgia', 'enthusiasm',
    'fulfillment', 'reverence', 'empowerment', 'compassion', 'compassionate',
    'tenderness', 'arousal', 'bliss', 'wonder', 'cheerfulness',
    'delight', 'ecstasy', 'relief', 'determination', 'inspiration', 'inspired',
    'playful', 'enchantment', 'calmness', 'thrill', 'grateful', 'proud',
    'accomplishment', 'satisfaction', 'anticipation', 'creative', 'eager',
    'kind', 'peaceful', 'vibrant', 'wonderful', 'amazing', 'fantastic',
    'delighted', 'perfect', 'bravery', 'courage', 'success', 'successful'
}

NEGATIVE_LABELS = {
    'negative', 'sadness', 'sad', 'anger', 'angry', 'fear', 'disgust',
    'shame', 'bitterness', 'bitter', 'despair', 'grief', 'loneliness',
    'disappointed', 'disappointment', 'pain', 'sorrow', 'horror', 'terror',
    'hate', 'misery', 'depression', 'anxiety', 'rage', 'frustration',
    'embarrassed', 'embarrassment', 'guilt', 'jealousy', 'envy', 'regret',
    'horror', 'dread', 'melancholy', 'anguish', 'heartbreak', 'bad',
    'frustrated', 'betrayal', 'desolation', 'boredom', 'devastated',
    'dismissive', 'envious', 'broken', 'defeated', 'overwhelmed',
    'numbness', 'devastation', 'frustrated', 'shameful', 'worried',
    'upset', 'unhappy', 'lonely', 'scary', 'suffering'
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

    # ── Models & Hyperparameter Tuning ─────────────────────────────────────────
    # Define search spaces for each model (Manual AutoML)
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'params': {'C': [0.1, 1.0, 10.0]}
        },
        'Naive Bayes': {
            'model': MultinomialNB(),
            'params': {'alpha': [0.1, 0.5, 1.0]}
        },
        'Linear SVC': {
            'model': LinearSVC(max_iter=3000, class_weight='balanced'),
            'params': {'C': [0.1, 0.8, 2.0]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'params': {'max_depth': [10, 20, None]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {'max_depth': [10, 20, 30]}
        },
    }

    results = {}
    if not os.path.exists('results'): os.makedirs('results')

    CLASS_ORDER = ['Positive', 'Negative', 'Neutral']

    print("\nTraining & Fine-Tuning 5 Models (GridSearch)...")
    for name, config in models_config.items():
        with mlflow.start_run(run_name=f"Tuned_{name}"):
            print(f"  → Tuning {name}")
            
            # Perform Grid Search
            grid = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            # Log Best Params to MLflow
            mlflow.log_params(grid.best_params_)
            print(f"    Best Params: {grid.best_params_}")

            p, r, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            acc = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metrics({
                'precision': p,
                'recall': r,
                'f1_score': f1,
                'accuracy': acc
            })
            
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
            plt.title(f'Confusion Matrix – {name} (Tuned)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = f'results/cm_{name.replace(" ", "_")}.png'
            plt.savefig(cm_path, dpi=120)
            plt.close()
            
            # Log artifacts
            mlflow.log_artifact(cm_path)
            
            # Log best model
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

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
    with mlflow.start_run(run_name="Voting_Ensemble"):
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
        ens_acc = accuracy_score(y_test, ens_pred)
        
        print(f"Ensemble F1 Score: {ens_f1:.4f}")
        
        # Log ensemble metrics
        mlflow.log_metrics({'f1_score': ens_f1, 'accuracy': ens_acc})
        
        # Log and Register Model
        signature = infer_signature(X_train, y_train)
        model_info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            artifact_path="sentiment_ensemble",
            signature=signature,
            registered_model_name="VibeCheck_Production_Model"
        )
        
        # Save local backup
        if not os.path.exists('models'): os.makedirs('models')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(ensemble,   'models/best_sentiment_model.pkl')
        print(f"Models saved locally and registered in MLflow: {model_info.model_uri}")

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
