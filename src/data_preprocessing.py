import pandas as pd
import numpy as np
import os

def load_data(data_path):
    """
    Loads student-mat and student-por, combines them, and handles initial cleaning.
    """
    mat_path = os.path.join(data_path, 'student-mat.csv')
    por_path = os.path.join(data_path, 'student-por.csv')

    # Note: UCI dataset uses ';' as delimiter
    df_mat = pd.read_csv(mat_path, sep=';')
    df_por = pd.read_csv(por_path, sep=';')

    # Add subject column to distinguish
    df_mat['subject'] = 'Math'
    df_por['subject'] = 'Portuguese'

    # Combine datasets
    df = pd.concat([df_mat, df_por], ignore_index=True)
    
    print(f"Data Loaded. Total records: {df.shape[0]}")
    return df

def preprocess_data(df):
    """
    Clean data, handle binary mappings, and create target variable.
    """
    # 1. Create Target Variable (Binary Classification)
    # G3 >= 10 is a pass in this grading system (0-20 scale)
    df['success'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    # 2. Binary mapping for 'address', 'famsize', 'Pstatus', 'schoolsup', etc.
    # Converting strict binary yes/no strings to 0/1 for efficiency
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    return df

# Add this to the bottom of src/data_preprocessing.py to test it
if __name__ == "__main__":
    # Adjust path to where your data is
    data_path = "../data" 
    if os.path.exists(data_path):
        df = load_data(data_path)
        df_clean = preprocess_data(df)
        print("Preprocessing test successful!")
        print(df_clean.head())
    else:
        print("Data folder not found. Run from the src directory or check paths.")