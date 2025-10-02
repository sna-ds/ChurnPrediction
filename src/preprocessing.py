import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """Load dataset from CSV."""
    return pd.read_csv(path)

def preprocess_data(df, target="Churn"):
    """Clean and preprocess churn dataset."""
    df = df.copy()
    
    # Drop customerID if exists
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns.drop(target)
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Encode target variable
    df[target] = le.fit_transform(df[target])
    
    # Split features/target
    X = df.drop(target, axis=1)
    y = df[target]
    
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Train-test split and scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
