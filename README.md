🤖 AutoML Machine Learning Dashboard
Train • Test • Predict — Automatically (Flask + Scikit-Learn + Render)

A powerful, production-ready AutoML web application that automatically handles:

ML task detection (Classification / Regression)

Data preprocessing

Model training

Model testing

Real-time predictions

JSON API response

Upload ANY CSV → The system figures out the rest.

🧠 Tech Stack

Frontend: HTML, CSS, Bootstrap (Glassmorphism UI)

Backend: Flask (Python)

Machine Learning: Scikit-Learn, Pandas, NumPy

Deployment: Render

Model Storage: Joblib

🚀 Features
🔹 Auto Task Detection

System automatically decides whether dataset is:

Classification (e.g., Purchased, Churn, Label)

Regression (e.g., Salary, Charges, Price)

No user input needed — it's fully automated.

🔹 Automated Preprocessing

Missing value handling

Outlier removal (IQR)

Standard Scaling

One-Hot Encoding

Column alignment for predictions

🔹 AutoML Model Training

Trains 10+ ML models and selects the best performing model automatically.

🔹 Testing Module

Upload test CSV → Get

Accuracy (classification)

R² Score (regression)

🔹 Prediction Module

Supports:

UI Prediction (key=value input)

JSON API Prediction

🔹 Modern UI + Cloud Deployment

Glass UI

Responsive design

Fast & lightweight

Fully deployed on Render

🌐 Live Demo

🔗 Add your Render link here

📦 Installation
git clone https://github.com/YOUR_USERNAME/AutoML-Dashboard.git
cd AutoML-Dashboard

python -m venv automl
automl\Scripts\activate       # Windows
# source automl/bin/activate  # Mac/Linux

pip install -r requirements.txt
python app.py

🖼 Usage
1️⃣ Train a Model

Upload training CSV

App auto-detects:

Target column

Task type (classification/regression)

Data types

Trains multiple models

Picks the best one

2️⃣ Test the Model

Upload test CSV → Get Accuracy / R².

3️⃣ Predict via UI
Age=35
EstimatedSalary=60000

4️⃣ Predict via JSON API

POST → /predict_json

{ "Age": 35, "EstimatedSalary": 60000 }


Response:

{ "prediction": "1" }

⚡ Highlights

Fully automated AutoML pipeline

Smart task detection (classification/regression)

Trains & compares 10+ machine learning models

Automatic preprocessing (no manual work)

UI input + JSON API support

Clean and responsive Bootstrap interface

Works on ANY dataset (generic ML system)

Deployed online using Render