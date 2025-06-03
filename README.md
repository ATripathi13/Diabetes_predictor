Diabetes Predictor

A machine learning project designed to predict the likelihood of diabetes in individuals based on medical data. The project encompasses data acquisition, preprocessing, model training, evaluation, and deployment via an API.

📁 Project Structure

Diabetes_predictor/

├── api/                      # API deployment scripts

├── model/                    # Saved machine learning models

├── plots/                    # Visualizations and plots

├── report/                   # Evaluation reports and documentation

├── download_dataset.py       # Script to download the dataset

├── generate_dataset.py       # Script to generate synthetic data

├── preprocess_and_train.py   # Data preprocessing and model training

├── requirements.txt          # Python dependencies

└── README.md                 # Project overview and instructions

🚀 Features

1. Data Acquisition: Automated scripts to download and generate datasets.
2. Data Preprocessing: Handling missing values, feature scaling, and encoding.
3. Model Training: Implementation of machine learning algorithms to predict diabetes.
4. Evaluation: Performance metrics and visualizations to assess model accuracy.
5. API Deployment: Expose the trained model via a RESTful API for real-time predictions.

🛠️ Installation

1. Clone the repository:
   git clone https://github.com/ATripathi13/Diabetes_predictor.git
   cd Diabetes_predictor
2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required dependencies:
   pip install -r requirements.txt

📊 Usage

1. Download the dataset:
   python download_dataset.py
2. Generate synthetic data (if needed):
   python generate_dataset.py
3. Preprocess data and train the model:
   python preprocess_and_train.py
4. Run the API server:
   uvicorn api.main:app --reload

🌐 API Overview: Diabetes Prediction API
This API provides a RESTful interface to predict whether a person is diabetic based on health-related input features like glucose level, BMI, insulin, etc. It allows seamless real-time predictions using the trained machine learning model.

⚙️ How It Works
1. 📦 Model Loading

    * When the API server starts (main.py inside the api/ directory), it loads the trained machine learning model (e.g., model.pkl) from disk using joblib or pickle.

2. 🚪 API Endpoint

    * POST /predict

        * Accepts a JSON payload containing medical features (like glucose, BMI, age, etc.).
        * Converts the input into the format expected by the model (e.g., numpy array or pandas DataFrame).
        * Runs the prediction using the preloaded model.
        * Returns the result (e.g., {"prediction": "Diabetic"} or {"prediction": "Non-Diabetic"}).

🧠 Input Format Example

{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 32.0,
  "Age": 35
}

🔁 Output Example

{
  "prediction": "1"
}

🧰 Requirements

✅ Python Packages

Install the requirements (libraries):
pip install -r requirements.txt

✅ Running the API Server

uvicorn main:app --reload
This command runs the API locally at http://127.0.0.1:8000.

✅ Try It Out

Once running, you can:
Visit the interactive Swagger docs at: http://127.0.0.1:8000/docs

📌 Summary

Feature	Description
Endpoint	POST /predict
Input	JSON with patient health data
Output	JSON with prediction result
Backend	Python with FastAPI
Model	Pre-trained ML model (e.g., RandomForest)
Use case	Real-time diabetes prediction API


✅ Steps and What’s Happening

📥 Dataset Download (download_dataset.py)

Downloads the Pima Indians Diabetes dataset from a public source (Kaggle or similar).
This dataset contains features like glucose level, BMI, blood pressure, etc., labeled with whether a patient has diabetes or not.

🧪 Data Generation (generate_dataset.py)

1. Optionally creates a synthetic dataset with similar structure for testing or experimentation.
2. Helps in creating varied data inputs if required.

🧹 Preprocessing and Training (preprocess_and_train.py)

1. Loads the dataset.
2. Cleans the data (handles missing values, scales features).
3. Splits the data into training and testing sets.
4. Trains a machine learning model (e.g., Random Forest or Logistic Regression).
5. Saves the trained model for future use.

📊 Evaluation

1. Calculates metrics like accuracy, precision, recall, and F1-score.
2. Generates and saves visual plots (confusion matrix, ROC curve, etc.) to assess model performance.

🌐 API Integration (api/)

1. Loads the saved ML model.
2. Uses FastAPI or Flask to expose a REST API.
3. Accepts patient data as input and returns prediction (diabetic or not).
4. This allows real-time predictions via HTTP requests (e.g., from a frontend or mobile app).

📈 Results

The trained model achieves high accuracy in predicting diabetes cases. Detailed evaluation metrics and visualizations can be found in the report/ and plots/ directories.

