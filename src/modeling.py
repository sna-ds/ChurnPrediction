from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """Train Random Forest model."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with classification report & confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

def save_model(model, path: str = "results/model.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, path)

def load_model(path: str = "results/model.pkl"):
    """Load trained model from disk."""
    return joblib.load(path)
