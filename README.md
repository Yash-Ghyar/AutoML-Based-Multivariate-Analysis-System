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
=======
🤖 AutoML-Based Multivariate Analysis System

An advanced Flask-based Machine Learning application that automatically preprocesses data, analyzes multivariate relationships, trains multiple ML models, and selects the best-performing algorithm — all through a clean, responsive Bootstrap interface.

📊 The system performs end-to-end ML automation, including missing value handling, outlier removal, encoding, scaling, model evaluation, multivariate analysis (PCA, VIF, Correlation), and real-time predictions.

🧠 Tech Stack

Frontend:
• HTML
• CSS
• Bootstrap (No JavaScript)
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009

Backend:
• Flask (Python)

<<<<<<< HEAD
Machine Learning: Scikit-Learn, Pandas, NumPy

Deployment: Render

Model Storage: Joblib

🚀 Features
🔹 Auto Task Detection

System automatically decides whether dataset is:

Classification (e.g., Purchased, Churn, Label)

Regression (e.g., Salary, Charges, Price)

No user input needed — it's fully automated.
=======
Machine Learning:
• Scikit-learn
• Pandas
• NumPy
• Statsmodels
• Matplotlib
• Seaborn

🚀 Features
🔹 Automatic Target Detection

System intelligently identifies the target column — no manual input required.

🔹 Automatic ML Task Identification

Detects whether the dataset is for:
✔ Classification
✔ Regression

🔹 Complete Data Preprocessing

Handles all essential preprocessing steps:
• Missing value treatment (Median/Mode)
• Outlier removal using IQR
• Encoding categorical variables
• Scaling numeric features

🔹 AutoML Best Model Selection

Trains multiple ML models and automatically selects the one with the highest accuracy (classification) or R² score (regression).

🔹 Multivariate Analysis

Generates professional statistical insights:
📊 Correlation Heatmap
🧩 PCA (Dimensionality Reduction)
📈 VIF (Multicollinearity Detection)

🔹 Interactive Prediction Engine

Enter values in key=value format → get real-time predictions instantly.

🔹 Modern Bootstrap UI

Clean, elegant, responsive — perfect for presentations and industry demos.
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009

🔹 Automated Preprocessing

<<<<<<< HEAD
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

=======
1️⃣ Upload your dataset (CSV)
2️⃣ System preprocesses the data automatically
3️⃣ Multiple ML models are trained
4️⃣ Best model is selected and saved
5️⃣ Multivariate analysis visualizations are generated
6️⃣ You can input values to get real-time predictions

📦 Installation
git clone https://github.com/Yash-Ghyar/Flask-AutoML-Multivariate-Analysis.git
cd Flask-AutoML-Multivariate-Analysis
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
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
