import os
import warnings
from data_preprocessing import load_data, preprocess_data
from feature_engineering import engineer_features, split_and_scale
from model_training import train_model, save_model
from evaluation import evaluate_model

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("--- Starting Student Success Prediction Pipeline ---")
    
    # 1. Define Paths
    # Go up one level from src/ to root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    
    # 2. Load and Preprocess
    df_raw = load_data(DATA_DIR)
    df_clean = preprocess_data(df_raw)
    
    # 3. Feature Engineering
    X, y = engineer_features(df_clean)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    # 4. Train Model (XGBoost)
    model = train_model(X_train, y_train)
    
    # 5. Evaluate
    evaluate_model(model, X_test, y_test, output_dir=OUTPUT_DIR)
    
    # 6. Save Artifacts
    model_dir = os.path.join(BASE_DIR, 'models')
    save_model(model, scaler, output_dir=model_dir)
    
    print("\n--- Pipeline Finished Successfully ---")
    print(f"Check {OUTPUT_DIR} for reports and plots.")

if __name__ == "__main__":
    main()