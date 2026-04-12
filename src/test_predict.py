import joblib
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/test_predict.py \"<Your text to analyze>\"")
        print("Example: python src/test_predict.py \"I am absolutely thrilled about this news!\"")
        sys.exit(1)
        
    text_input = sys.argv[1]
    
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/best_sentiment_model.pkl')
    except Exception as e:
        print("Error loading models! Run 'python3 src/main.py' to train and save them first.")
        sys.exit(1)
        
    # Process text
    text_vec = vectorizer.transform([text_input])
    prediction = model.predict(text_vec)
    
    print("-" * 50)
    # The sentiment models in our dataset use values that have leading/trailing whitespaces (like " Positive  ")
    # so we'll strip them to make the print look cleaner.
    clean_prediction = str(prediction[0]).strip()
    
    print(f"Input Text:         '{text_input}'")
    print(f"Predicted Sentiment: {clean_prediction}")
    print("-" * 50)

if __name__ == '__main__':
    main()
