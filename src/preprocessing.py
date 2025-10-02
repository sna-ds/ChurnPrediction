import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load churn dataset from CSV."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn"):
    """Preprocess dataset: encode categorical variables, scale numerical features."""
    df = df.copy()
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Split features & target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
