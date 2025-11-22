import xgboost as xgb
import joblib
import os

def train_model(X_train, y_train):
    """
    Trains an XGBoost Classifier.
    Optimized for speed using 'hist' tree method.
    """
    print("Training XGBoost Model...")
    
    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,           # Sufficient for this data size
        learning_rate=0.1,
        max_depth=5,
        tree_method='hist',         # FAST training method
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    print("Training Complete.")
    return model

def save_model(model, scaler, output_dir='models'):
    """Saves the model and scaler for future use."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump(model, os.path.join(output_dir, 'student_success_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"Model saved to {output_dir}/")