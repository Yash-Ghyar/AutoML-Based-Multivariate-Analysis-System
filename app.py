import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

# Flask setup
app = Flask(__name__)
app.secret_key = "super_secret_key"

# Folder setup
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

MODEL_PATH = "models/best_model.pkl"


# ---------------------------------------------------------
# Auto target detection
# ---------------------------------------------------------
def detect_target(df):
    possible = ["target", "label", "output", "class", "churn"]
    for col in df.columns:
        if col.lower() in possible:
            return col
    return df.columns[-1]

<<<<<<< HEAD

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
=======
# -------------------------------------------------
# Home page
# -------------------------------------------------
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
@app.route("/")
def index():
    return render_template("index.html")

<<<<<<< HEAD

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
=======
# -------------------------------------------------
# Train model (auto-select best + task display)
# -------------------------------------------------
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("file")
    if not file:
        flash("Upload a training CSV file", "danger")
        return redirect(url_for("index"))

    path = os.path.join("uploads", f"{uuid.uuid4().hex}.csv")
    file.save(path)
    df = pd.read_csv(path)

    # Detect target
    target = detect_target(df)
    if target not in df.columns:
        flash("Target column missing!", "danger")
        return redirect(url_for("index"))

    # Split X and y
    X = df.drop(columns=[target])
    y = df[target]

<<<<<<< HEAD
    # Missing values
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Outlier removal (IQR)
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
        X, y = X[mask], y[mask]

    # Task type
    task = "classification" if y.nunique() <= 20 or y.dtype == "object" else "regression"

    # Preprocessing
=======
    # ------------------------------------------------------
    # 1️⃣ MISSING VALUE HANDLING
    # ------------------------------------------------------
    # Numeric → median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Categorical → mode
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])

    # ------------------------------------------------------
    # 2️⃣ OUTLIER DETECTION & REMOVAL (IQR METHOD)
    # ------------------------------------------------------
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Keeping only non-outliers
        mask = (X[col] >= lower) & (X[col] <= upper)
        X = X[mask]
        y = y[mask]

    # ------------------------------------------------------
    # Detect if Classification or Regression
    # ------------------------------------------------------
    task = "classification" if y.nunique() <= 20 or y.dtype == "object" else "regression"

    # Handle categorical + numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

<<<<<<< HEAD
    # Model options
=======
    # ------------------------------------------------------
    # MODEL DEFINITIONS
    # ------------------------------------------------------
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
    if task == "classification":
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=200),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "DecisionTree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }
        metric = accuracy_score
    else:
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=200),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "SVR": SVR()
        }
        metric = r2_score

<<<<<<< HEAD
    # Train model suite
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
=======
    # ------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009

    best_model, best_name, best_score = None, None, -999

    # ------------------------------------------------------
    # AUTO MODEL SELECTION
    # ------------------------------------------------------
    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            score = metric(y_test, preds)
            if score > best_score:
                best_score, best_model, best_name = score, pipe, name
        except:
            pass

    # Save final model
    joblib.dump({
        "model": best_model,
        "task": task,
        "target": target,
        "columns": X.columns.tolist()
    }, MODEL_PATH)

    return render_template(
        "result.html",
        training=True,
        task=task,
        best_model=best_name,
        score=round(best_score, 4),
        target=target
    )

<<<<<<< HEAD

# ---------------------------------------------------------
# TEST MODEL
# ---------------------------------------------------------
@app.route("/test", methods=["POST"])
def test():
    if not os.path.exists(MODEL_PATH):
        flash("Train a model first!", "danger")
        return redirect(url_for("index"))

    file = request.files.get("file")
    if not file:
        flash("Upload a test CSV!", "danger")
=======
# -------------------------------------------------
# Multivariate analysis
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        flash("Please upload a dataset for analysis.", "danger")
>>>>>>> 514d4e28b53b133ef376ac4f06063e24ed89f009
        return redirect(url_for("index"))

    path = os.path.join("uploads", f"{uuid.uuid4().hex}.csv")
    file.save(path)
    df = pd.read_csv(path)

    model_data = joblib.load(MODEL_PATH)
    pipe, target, task = model_data["model"], model_data["target"], model_data["task"]

    if target not in df.columns:
        flash(f"Test CSV must include target column: {target}", "danger")
        return redirect(url_for("index"))

    X_test = df.drop(columns=[target])
    y_test = df[target]
    preds = pipe.predict(X_test)

    if task == "classification":
        score = accuracy_score(y_test, preds)
        metric_name = "Accuracy"
    else:
        score = r2_score(y_test, preds)
        metric_name = "R² Score"

    return render_template(
        "result.html",
        testing=True,
        test_score=round(score, 4),
        metric_name=metric_name
    )


# ---------------------------------------------------------
# PREDICT FIXED VERSION (SKIP BLANK LINES)
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        flash("Train a model first!", "danger")
        return redirect(url_for("index"))

    text = request.form.get("inputdata")
    if not text:
        flash("Enter input values", "danger")
        return redirect(url_for("index"))

    model_data = joblib.load(MODEL_PATH)
    pipe, columns = model_data["model"], model_data["columns"]

    try:
        data = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line:     # skip empty blank lines
                continue
            if "=" not in line:
                continue

            k, v = line.split("=")
            data[k.strip()] = v.strip()

        # Convert numbers
        for k in data:
            try:
                data[k] = float(data[k])
            except:
                pass

        X = pd.DataFrame([data])

        # Add missing columns = 0
        for col in columns:
            X[col] = X.get(col, 0)

        pred = pipe.predict(X)[0]

        return render_template(
            "result.html",
            predicting=True,
            prediction=pred
        )

    except Exception as e:
        flash(f"Invalid input: {e}", "danger")
        return redirect(url_for("index"))


# ---------------------------------------------------------
# JSON API
# ---------------------------------------------------------
@app.route("/predict_json", methods=["POST"])
def predict_json():
    model_data = joblib.load(MODEL_PATH)
    pipe, columns = model_data["model"], model_data["columns"]

    data = request.json
    X = pd.DataFrame([data])

    for col in columns:
        X[col] = X.get(col, 0)

    pred = pipe.predict(X)[0]
    return jsonify({"prediction": str(pred)})


# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

