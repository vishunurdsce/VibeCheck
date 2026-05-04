# MLOps Sentiment Analysis – Project Report

## 3.2 Performance Comparison of All Models

|                     |   Precision |   Recall |   F1 Score |   Accuracy |
|:--------------------|------------:|---------:|-----------:|-----------:|
| Logistic Regression |      0.7384 |   0.7415 |     0.7184 |     0.7415 |
| Naive Bayes         |      0.7387 |   0.7347 |     0.7294 |     0.7347 |
| Linear SVC          |      0.7681 |   0.7483 |     0.7188 |     0.7483 |
| Random Forest       |      0.7376 |   0.7007 |     0.6757 |     0.7007 |
| Decision Tree       |      0.7145 |   0.6463 |     0.5662 |     0.6463 |

## 5 Conclusion
**Best individual model:** Naive Bayes (F1: 0.7294)

**Final deployed model:** Voting Ensemble (SVC + Logistic Regression + Random Forest)

The ensemble is used for deployment as it cross-validates predictions across three diverse classifiers, reducing individual model bias and improving robustness on unseen text.
