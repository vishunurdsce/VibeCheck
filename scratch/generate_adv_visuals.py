import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

def generate_advanced_visuals():
    results_dir = 'results'
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    # Data from metrics.csv or the previous run
    # For a radar chart, we need Accuracy, Precision, Recall, F1 for each model
    data = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'Logistic Regression': [0.7291, 0.7211, 0.7192, 0.7211],
        'Naive Bayes': [0.7020, 0.6871, 0.6572, 0.6871],
        'Linear SVC': [0.7415, 0.7347, 0.7331, 0.7347],
        'Random Forest': [0.7500, 0.6599, 0.6324, 0.6599],
        'Decision Tree': [0.5579, 0.4898, 0.3692, 0.4898]
    }
    df = pd.DataFrame(data)

    # 1. RADAR CHART (Spider Plot)
    labels = df['Metric'].values
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    models = df.columns[1:]
    
    for i, model in enumerate(models):
        values = df[model].values.flatten().tolist()
        values += values[:1] # Close the circle
        ax.plot(angles, values, color=colors[i], linewidth=2, label=model)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    plt.title('Model Performance Comparison (Radar Chart)', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_radar_chart.png'), dpi=150)
    plt.close()

    print("Radar chart generated: results/performance_radar_chart.png")

    # 2. PRECISION-RECALL TRADE-OFF (Dot Plot)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.scatter(df.loc[1, model], df.loc[0, model], s=100, color=colors[i], label=model, edgecolors='black')
        plt.text(df.loc[1, model]+0.005, df.loc[0, model]+0.005, model, fontsize=9)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision vs Recall Trade-off', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0.4, 0.8)
    plt.ylim(0.5, 0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'precision_recall_tradeoff.png'), dpi=150)
    plt.close()
    
    print("Trade-off plot generated: results/precision_recall_tradeoff.png")

if __name__ == "__main__":
    generate_advanced_visuals()
