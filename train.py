"""
train.py
--------
Model Training Script for Heart Disease Risk Estimator.

Loads the Heart Disease dataset, trains a Logistic Regression classifier
wrapped with feature scaling in a scikit-learn Pipeline, evaluates model
performance, and exports the serialized model as 'heart_disease_model.joblib'.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

DATA_PATH = "data/heart.csv"
MODEL_PATH = "heart_disease_model.joblib"

# Canonical feature names matching UCI Cleveland Heart Disease dataset
FEATURE_COLS = [
    "age",       # Age in years
    "sex",       # Sex (1 = male; 0 = female)
    "cp",        # Chest pain type (0, 1, 2, 3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",   # Resting electrocardiographic results (0, 1, 2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise-induced angina (1 = yes; 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of peak exercise ST segment (0, 1, 2)
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal"       # Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
]
TARGET_COL = "target"


def get_dataset() -> pd.DataFrame:
    """
    Loads dataset from local file if available.
    If missing, attempts to download the UCI dataset or creates a clean sample.
    """
    if os.path.exists(DATA_PATH):
        print(f"[+] Loading dataset from existing file: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Attempt to download standard Cleveland dataset
    remote_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Heart%20Disease%20UCI.csv"
    try:
        print(f"[!] Local file not found. Fetching dataset from {remote_url}...")
        df = pd.read_csv(remote_url)
        # Rename standard columns if needed
        df.columns = [c.strip().lower() for c in df.columns]
        if "target" not in df.columns and "condition" in df.columns:
            df = df.rename(columns={"condition": "target"})
        df.to_csv(DATA_PATH, index=False)
        print(f"[+] Successfully downloaded and saved to {DATA_PATH}")
        return df
    except Exception as e:
        print(f"[!] Remote download failed ({e}). Generating fallback sample dataset...")
        np.random.seed(42)
        n_samples = 300
        
        # Realistic synthetic distributions based on UCI Cleveland parameters
        age = np.random.randint(30, 78, n_samples)
        sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
        cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
        trestbps = np.random.randint(95, 195, n_samples)
        chol = np.random.randint(130, 390, n_samples)
        fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        restecg = np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.49, 0.02])
        thalach = np.clip(180 - 0.6 * age + np.random.normal(0, 15, n_samples), 80, 205).astype(int)
        exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])
        oldpeak = np.round(np.clip(np.random.exponential(1.0, n_samples), 0.0, 6.2), 1)
        slope = np.random.choice([0, 1, 2], n_samples, p=[0.46, 0.46, 0.08])
        ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.58, 0.22, 0.13, 0.07])
        thal = np.random.choice([1, 2, 3], n_samples, p=[0.06, 0.55, 0.39])
        
        # Generate target probabilities based on risk factors
        logits = (
            -2.5 
            + 0.03 * (age - 50) 
            + 0.5 * sex 
            + 0.5 * cp 
            + 0.015 * (trestbps - 120) 
            + 0.005 * (chol - 200) 
            + 0.5 * exang 
            + 0.6 * oldpeak 
            + 0.4 * ca
        )
        probs = 1 / (1 + np.exp(-logits))
        target = (np.random.rand(n_samples) < probs).astype(int)
        
        df = pd.DataFrame({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal, "target": target
        })
        df.to_csv(DATA_PATH, index=False)
        print(f"[+] Fallback dataset created and saved to {DATA_PATH}")
        return df


def main():
    print("=" * 60)
    print("Heart Disease Risk Estimator - Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    df = get_dataset()
    print(f"[+] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Target balance: {dict(df[TARGET_COL].value_counts())}")

    # Ensure required columns are present
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 2. Train-Test Split (80/20 with stratified target distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split sizes -> Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Model Pipeline: StandardScaler + LogisticRegression
    # Using a Pipeline ensures proper scaling during both training and Streamlit inference
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("[+] Training Logistic Regression classifier with feature scaling...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n" + "-" * 40)
    print("Model Evaluation Metrics:")
    print("-" * 40)
    print(f"Accuracy Score : {acc * 100:.2f}%")
    print(f"ROC-AUC Score  : {roc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "High Risk (1)"]))

    # 5. Export Model & Metadata
    model_payload = {
        "pipeline": pipeline,
        "feature_names": FEATURE_COLS,
        "accuracy": acc,
        "roc_auc": roc
    }

    joblib.dump(model_payload, MODEL_PATH)
    print(f"[+] Model successfully exported to: {os.path.abspath(MODEL_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
