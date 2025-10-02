import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance

def plot_feature_importance(model, feature_names, output_path="results/feature_importance.png"):
    """Plot Random Forest feature importance."""
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_permutation_importance(model, X_test, y_test, feature_names, output_path="results/permutation_importance.png"):
    """Plot permutation importance."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=perm_imp, y=perm_imp.index)
    plt.title("Permutation Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
