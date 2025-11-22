import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Prepares X (features) and y (target).
    Handles Categorical encoding and Scaling.
    """
    # Drop Target and 'G3' (Leakage). 
    # We keep G1 and G2 for 'mid-term' prediction.
    # If the column doesn't exist (e.g. dropped already), ignore errors
    drop_cols = ['G3', 'success']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(existing_drop_cols, axis=1)
    y = df['success']

    # Identify Categorical Columns (Object types)
    categorical_cols = X.select_dtypes(include=['object']).columns

    # One-Hot Encoding for nominal variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y

def split_and_scale(X, y):
    """
    Splits data into Train/Test and Scales numerical features.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to keep column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_df, X_test_df, y_train, y_test, scaler