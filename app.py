import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
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
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = "super_secret_key"

os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

MODEL_PATH = "models/best_model.pkl"


# -------------------------------------------------
# Helper: Detect target column automatically
# -------------------------------------------------
def detect_target(df):
    """Automatically detect potential target column"""
    for col in df.columns:
        if col.lower() in ["target", "churn", "label", "class", "output", "price"]:
            return col
    return df.columns[-1]


# -------------------------------------------------
# Home page
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------
# Train model (auto-select best + task display)
# -------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("file")
    if not file:
        flash("Please upload a CSV file.", "danger")
        return redirect(url_for("index"))

    path = os.path.join("uploads", f"{uuid.uuid4().hex}.csv")
    file.save(path)
    df = pd.read_csv(path)

    target = detect_target(df)
    if target not in df.columns:
        flash("Target column not found automatically. Please rename target column properly.", "danger")
        return redirect(url_for("index"))

    X = df.drop(columns=[target])
    y = df[target]

    # Determine task type
    task = "classification" if y.nunique() <= 20 or y.dtype == "object" else "regression"

    # Handle categorical/numeric data automatically
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Define models
    if task == "classification":
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "DecisionTree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
        }
        metric = accuracy_score
    else:
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "DecisionTree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "SVR": SVR()
        }
        metric = r2_score

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_name, best_score = None, None, -999

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            score = metric(y_test, preds)
            if score > best_score:
                best_score, best_model, best_name = score, pipe, name
        except Exception as e:
            print(f"Model {name} failed: {e}")

    # Save best model
    joblib.dump({
        "model": best_model,
        "target": target,
        "task": task,
        "columns": X.columns.tolist()
    }, MODEL_PATH)

    metric_name = "Accuracy" if task == "classification" else "R² Score"
    flash(f"✅ Detected Task: {task.title()} | Best Model: {best_name} ({metric_name} = {best_score:.4f})", "success")
    return redirect(url_for("index"))


# -------------------------------------------------
# Multivariate analysis
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        flash("Please upload a dataset for analysis.", "danger")
        return redirect(url_for("index"))

    # Save uploaded file
    path = os.path.join("uploads", f"{uuid.uuid4().hex}.csv")
    file.save(path)
    df = pd.read_csv(path)

    # Drop useless ID-like columns automatically
    df = df.loc[:, ~df.columns.str.contains("id", case=False)]

    # Select numeric columns
    numeric = df.select_dtypes(include=[np.number]).fillna(0)
    if numeric.shape[1] < 2:
        flash("Need at least 2 numeric columns for analysis.", "danger")
        return redirect(url_for("index"))

    # ✅ 1. Scale numeric features before calculating VIF
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    # ✅ 2. Correlation heatmap
    corr = pd.DataFrame(scaled, columns=numeric.columns).corr()
    corr_img = f"corr_{uuid.uuid4().hex}.png"
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Scaled Data)")
    plt.tight_layout()
    plt.savefig(os.path.join("static", corr_img))
    plt.close()

    # ✅ 3. PCA plot
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)
    pca_img = f"pca_{uuid.uuid4().hex}.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (2 Components)")
    plt.tight_layout()
    plt.savefig(os.path.join("static", pca_img))
    plt.close()

    # ✅ 4. VIF Calculation (After Scaling)
    vif_df = pd.DataFrame()
    vif_df["Feature"] = numeric.columns
    vif_df["VIF"] = [variance_inflation_factor(scaled, i) for i in range(scaled.shape[1])]

    # Round off for better readability
    vif_df["VIF"] = vif_df["VIF"].round(3)

    return render_template(
        "index.html",
        analyzed=True,
        corr_image=corr_img,
        pca_image=pca_img,
        vif_summary=vif_df.to_dict(orient="records")
    )


# -------------------------------------------------
# Predict endpoint
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        flash("Please train a model first.", "danger")
        return redirect(url_for("index"))

    model_data = joblib.load(MODEL_PATH)
    pipe = model_data["model"]
    columns = model_data["columns"]
    task = model_data["task"]

    text = request.form.get("inputdata")
    if not text:
        flash("Please enter input data.", "danger")
        return redirect(url_for("index"))

    try:
        data = {}
        for line in text.strip().split("\n"):
            k, v = line.split("=")
            data[k.strip()] = v.strip()

        # Convert numeric where possible
        for k in data:
            try:
                data[k] = float(data[k])
            except:
                pass

        X = pd.DataFrame([data])

        # Handle missing columns
        missing_cols = [c for c in columns if c not in X.columns]
        for col in missing_cols:
            X[col] = 0

        X = X[columns]
        pred = pipe.predict(X)
        flash(f"📊 Task: {task.title()} | Prediction: {pred[0]}", "info")
    except Exception as e:
        flash(f"Invalid input format: {e}", "danger")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
