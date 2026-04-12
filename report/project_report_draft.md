# MLOps Sentiment Analysis – Project Report

## 3.2 Performance Comparison of All Models

|                     |   Precision |   Recall |   F1 Score |   Accuracy |
|:--------------------|------------:|---------:|-----------:|-----------:|
| Logistic Regression |      0.7291 |   0.7211 |     0.7192 |     0.7211 |
| Naive Bayes         |      0.702  |   0.6871 |     0.6572 |     0.6871 |
| Linear SVC          |      0.7415 |   0.7347 |     0.7331 |     0.7347 |
| Random Forest       |      0.75   |   0.6599 |     0.6324 |     0.6599 |
| Decision Tree       |      0.5579 |   0.4898 |     0.3692 |     0.4898 |

## 5 Conclusion
**Best individual model:** Linear SVC (F1: 0.7331)

**Final deployed model:** Voting Ensemble (SVC + Logistic Regression + Random Forest)

The ensemble is used for deployment as it cross-validates predictions across three diverse classifiers, reducing individual model bias and improving robustness on unseen text.
