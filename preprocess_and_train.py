import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Setup
# ----------------------------
np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("report", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------
file_path = 'data/diabetes.csv'

# Just read the CSV normally if it has a header row
df = pd.read_csv(file_path)

# ----------------------------
# EDA Visualizations
# ----------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(5, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Distribution of Diabetes Outcome")
plt.savefig("plots/outcome_distribution.png")
plt.close()

numeric_features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.savefig(f"plots/distribution_{feature}.png")
    plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("plots/feature_correlation.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Outcome", y="BMI", data=df)
plt.title("BMI by Diabetes Outcome")
plt.savefig("plots/bmi_vs_outcome.png")
plt.close()

# ----------------------------
# Data Preprocessing
# ----------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model Training & Evaluation
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

scores = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    scores[name] = (acc, m)

# ----------------------------
# Select Best Model
# ----------------------------
best_model_name = max(scores, key=lambda x: scores[x][0])
best_model = scores[best_model_name][1]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_model, "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved in /model")

# ----------------------------
# Feature Importances (if available)
# ----------------------------
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df.sort_values(by="Importance", ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("report/feature_importances.png")
    plt.close()

# ----------------------------
# Save Evaluation Report
# ----------------------------
with open("report/metrics.txt", "w") as f:
    for name, m in models.items():
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        f.write(f"{name}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"Precision: {prec:.3f}\n")
        f.write(f"Recall: {rec:.3f}\n\n")
