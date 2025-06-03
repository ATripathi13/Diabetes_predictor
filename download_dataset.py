import pandas as pd
import os

# URL of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names (add these manually, if the original file has no headers)
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

os.makedirs("data", exist_ok=True)

df = pd.read_csv(url, header=None, names=columns)
df.to_csv("data/diabetes.csv", index=False)

print("Downloaded and saved as data/diabetes.csv")
