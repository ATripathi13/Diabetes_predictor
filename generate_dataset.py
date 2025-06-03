import pandas as pd
import numpy as np
import os

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
