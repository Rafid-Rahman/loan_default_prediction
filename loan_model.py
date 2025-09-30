import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1. Load Data
df = pd.read_excel("default of credit card clients.xls", skiprows=1, engine="xlrd")
# Standardize target column name
for col in df.columns:
    if "default" in col.lower():
        df.rename(columns={col: "default_next_month"}, inplace=True)
        break

if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)

# 2. Feature Engineering
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
df['avg_bill'] = df[bill_cols].mean(axis=1)
df['bill_limit_ratio'] = df['avg_bill'] / df['LIMIT_BAL']

pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
df['avg_pay'] = df[pay_cols].mean(axis=1)
df['pay_bill_ratio'] = df['avg_pay'] / df['avg_bill'].replace(0, np.nan)
df['pay_bill_ratio'] = df['pay_bill_ratio'].fillna(0)

# 3. Features and Target
features = [
    'LIMIT_BAL', 'AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'avg_bill','bill_limit_ratio','avg_pay','pay_bill_ratio'
]
target = 'default_next_month'

X = df[features]
y = df[target]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Models
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf_clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

# 7. Evaluation Function
def evaluate_model(name, model, X_te, scaled=False):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    clf_report = classification_report(y_test, y_pred, target_names=['No Default','Default'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save results to text
    with open("model_results.txt", "a") as f:
        f.write(f"\n=== {name} ===\n")
        f.write(f"Accuracy: {acc:.2%}\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['No Default','Default']))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n" + "-"*50 + "\n")

    # Save classification report as CSV
    pd.DataFrame(clf_report).transpose().to_csv(f"{name}_classification_report.csv")

    # Save confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default','Default'],
                yticklabels=['No Default','Default'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# Evaluate both models
evaluate_model("LogisticRegression", log_reg, X_test_scaled)
evaluate_model("RandomForest", rf_clf, X_test)

# 8. Feature Importance from Random Forest
importances = rf_clf.feature_importances_
fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fi.to_csv("feature_importance.csv", index=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=fi)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig("uci_feature_importance.png")
plt.close()
