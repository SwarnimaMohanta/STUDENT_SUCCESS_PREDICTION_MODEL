from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, output_dir='outputs'):
    """
    Generates metrics and plots.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    # Create output dirs
    viz_dir = os.path.join(output_dir, 'visualizations')
    report_dir = os.path.join(output_dir, 'reports')
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 1. Save Text Report
    with open(os.path.join(report_dir, 'model_performance_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Accuracy: {acc:.2%}")
    print(f"Report saved to {report_dir}")

    # 2. Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. Feature Importance Plot
    # XGBoost built-in importance
    importance = model.feature_importances_
    feature_names = X_test.columns
    
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10) # Top 10 features

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Top 10 Features Predicting Student Success')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
    plt.close()