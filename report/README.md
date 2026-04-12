# Sentiment Analysis Project - MLOps Miniproject

I have successfully structured your sentiment analyzer project and generated all the required details based on the provided `MLOps Mini-project Report Format.docx`.

## 📁 Project Structure Generated
We created a clear directory structure so you can document the full end-to-end pipeline:
* `src/main.py`: The single Python pipeline script that handles data processing, model training, evaluation, and plotting.
* `results/`: Contains the generated performance graphs and model comparison metrics.
* `report/project_report_draft.md`: Contains text perfectly formatted and matched to Section 2, 3.2, and 5 of your reporting template.

## 🤖 5 Models Included
We have successfully trained and evaluated 5 distinct ML models as required:
1. **Logistic Regression**
2. **Naive Bayes (Multinomial)**
3. **Linear Support Vector Classifier (SVC)** *(Best Performing)*
4. **Random Forest** 
5. **Decision Tree**

## 📊 Graphs & Documentation
When you run the pipeline (`src/main.py`), it automatically generates:
1. **Individual Confusion Matrices**: `results/cm_Logistic_Regression.png`, `results/cm_Random_Forest.png`, etc.
2. **Collage/Combined Bar Chart**: `results/model_comparison.png`
3. **Metrics Table**: Evaluated Precision, Recall, F1 Score, and Accuracy for each.

## 📝 How to complete your report using this:
1. **Introduction / Pipeline (Section 1 & 3.1)**: We established a basic text machine-learning pipeline (Load Data -> Clean -> TF-IDF Vectorization -> Train & Test Split -> Train Models -> Evaluate).
2. **Model Selection (Section 2)**: Open `report/project_report_draft.md`. You can directly copy the descriptions, advantages, and disadvantages for all 5 models into your `.docx`.
3. **Performance (Section 3.2)**: A complete markdown table summarizing the accuracy, precision, recall, and F1-score is available in `report/project_report_draft.md`.
4. **Graphs (Section 4)**: Drag and drop the `.png` images generated in the `results/` folder into your document.
5. **Conclusion (Section 5)**: Also drafted in the report text, detailing why the best model (`Linear SVC`) performed better than the others on this dataset.
