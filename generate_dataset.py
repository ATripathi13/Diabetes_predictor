import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# --------------------------
# Generate Data
# --------------------------
np.random.seed(0)
rows = 500

os.makedirs("data", exist_ok=True)

data = {
    "Pregnancies": np.random.randint(0, 10, rows),
    "Glucose": np.random.normal(120, 30, rows).astype(int),
    "BloodPressure": np.random.normal(70, 10, rows).astype(int),
    "BMI": np.round(np.random.normal(32, 5, rows), 1),
    "Age": np.random.randint(20, 80, rows),
    "Outcome": np.random.randint(0, 2, rows)
}

df = pd.DataFrame(data)

df.to_csv("data/diabetes.csv", index=False)
print("/n---Synthetic data generated and saved to data/diabetes.csv---")

# --------------------------
# Data Exploration
# --------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())

print("\nOutcome Distribution:")
print(df['Outcome'].value_counts())

df.dropna(inplace=True)

df = df[(df["Glucose"] > 0) & (df["BMI"] > 0) & (df["BloodPressure"] > 0)]

features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

df.to_csv("data/diabetes_preprocessed.csv", index=False)
print("---Preprocessing done and saved to data/diabetes_preprocessed.csv---")
