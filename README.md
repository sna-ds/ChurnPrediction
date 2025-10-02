# Customer Churn Prediction with Hyperparameter Tuning

## 📌 Overview

This project aims to build a reliable **customer churn prediction model** using machine learning techniques and **hyperparameter tuning with cross-validation**. The goal is not only to maximize performance metrics but also to **interpret results for actionable business insights**.

* **Business Objective**: Detect customers who are likely to churn in order to design targeted retention strategies.
* **Approach**: Data preprocessing, feature engineering, encoding, scaling, model training, hyperparameter tuning, and evaluation.
* **Best Model Identified**: Random Forest (Balanced + Tuned), optimized for **Recall** to minimize false negatives.

---

## 📊 Dataset

The dataset contains customer subscription details, demographics, and billing information.

**Data Dictionary (selected columns):**

| Column           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `customerID`     | Unique customer ID                                     |
| `Gender`         | Gender (Male/Female)                                   |
| `SeniorCitizen`  | Senior citizen flag (0 = No, 1 = Yes)                  |
| `Partner`        | Has a partner (Yes/No)                                 |
| `Dependents`     | Has dependents (Yes/No)                                |
| `Tenure`         | Duration of subscription (months)                      |
| `Contract`       | Contract type (Month-to-month, One year, Two year)     |
| `PaymentMethod`  | Payment method (e.g., Electronic check, Bank transfer) |
| `MonthlyCharges` | Monthly charges                                        |
| `TotalCharges`   | Total charges                                          |
| `Churn`          | Target variable (Yes/No)                               |

---

## ⚙️ Workflow

1. **Data Understanding & Cleaning**

   * Removed duplicates and irrelevant columns (`customerID`)
   * Handled missing values (median for numerical, mode for categorical)
   * Encoded categorical variables (Label Encoding, Ordinal Encoding, One-Hot Encoding)

2. **Exploratory Data Analysis (EDA)**

   * Distribution plots for categorical and numerical features
   * Correlation analysis and feature reduction (dropped `TotalCharges`)

3. **Feature Engineering**

   * Scaling applied to numerical variables (`MonthlyCharges`, `Tenure`)

4. **Modeling & Hyperparameter Tuning**
   Models tested:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * XGBoost

   Metrics evaluated: **Accuracy, Precision, Recall, F1, AUC**

   * Special focus on **Recall & F1** (due to class imbalance).

5. **Best Model Selection**

   * **Random Forest (Balanced + Tuned)** achieved the highest Recall (0.87)
   * Trade-off: Lower Precision, but Recall is prioritized to reduce false negatives.

---

## 📈 Results

| Model               | Version  | Accuracy | Precision | Recall   | F1   |
| ------------------- | -------- | -------- | --------- | -------- | ---- |
| Logistic Regression | Baseline | 0.79     | 0.54      | 0.80     | 0.65 |
| Decision Tree       | Tuned    | 0.76     | 0.57      | 0.78     | 0.66 |
| Random Forest       | Tuned    | 0.67     | 0.44      | **0.87** | 0.59 |
| XGBoost             | Tuned    | 0.78     | 0.62      | 0.51     | 0.55 |

* **Business Interpretation**:

  * Prioritizing Recall ensures more churn cases are detected.
  * Better to wrongly target some non-churn customers than miss actual churners.

---

## 🛠️ Installation & Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/HW_HPTUNING_ChurnPrediction.git
cd HW_HPTUNING_ChurnPrediction
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

---

## 📌 Key Takeaways

* **Data imbalance** requires careful metric selection (Recall & F1 over Accuracy).
* **Hyperparameter tuning** significantly improves model performance.
* **Random Forest (Balanced + Tuned)** is the best choice for this churn prediction case.

---

## Thank You
**👤 Author**

**Suciningtyas Nur Alifah**

