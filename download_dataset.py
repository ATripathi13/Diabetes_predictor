import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

os.makedirs("data", exist_ok=True)

df = pd.read_csv(url, header=None, names=columns)
df.to_csv("data/diabetes.csv", index=False)

print("---Downloaded and saved as data/diabetes.csv---")

# ------------------------------
# Data Exploration Begins Here
# ------------------------------

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Basic Statistics:")
print(df.describe())

print("\n First 5 Rows:")
print(df.head())

print("\n Outcome Distribution:")
print(df['Outcome'].value_counts())

df.dropna(inplace=True)

df = df[(df["Glucose"] > 0) & (df["BMI"] > 0) & (df["BloodPressure"] > 0)]

features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

df.to_csv("data/diabetes_preprocessed.csv", index=False)
print("---Preprocessing done and saved to data/diabetes_preprocessed.csv---")