🤖 AutoML-Based Multivariate Analysis System

An advanced Flask-based Machine Learning application that automatically preprocesses data, analyzes multivariate relationships, trains multiple ML models, and selects the best-performing algorithm — all through a clean, responsive Bootstrap interface.

📊 The system performs end-to-end ML automation, including missing value handling, outlier removal, encoding, scaling, model evaluation, multivariate analysis (PCA, VIF, Correlation), and real-time predictions.

🧠 Tech Stack

Frontend:
• HTML
• CSS
• Bootstrap (No JavaScript)

Backend:
• Flask (Python)

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

⚙️ How It Works

1️⃣ Upload your dataset (CSV)
2️⃣ System preprocesses the data automatically
3️⃣ Multiple ML models are trained
4️⃣ Best model is selected and saved
5️⃣ Multivariate analysis visualizations are generated
6️⃣ You can input values to get real-time predictions

📦 Installation
git clone https://github.com/Yash-Ghyar/Flask-AutoML-Multivariate-Analysis.git
cd Flask-AutoML-Multivariate-Analysis
pip install -r requirements.txt
python app.py
