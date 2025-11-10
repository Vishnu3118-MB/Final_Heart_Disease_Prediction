
import json
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

BASE = Path(__file__).resolve().parent
ROOT = BASE
DATA = ROOT / "data" / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
IMG_DIR = ROOT / "app" / "static" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA)
target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Feature columns (all numeric here)
num_cols = X.columns.tolist()

# Train/Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Define models (with sensible defaults)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Gaussian NB": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # scaler safe but optional
        ("clf", GaussianNB())
    ]),
}

metrics_summary = {}
bar_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "model": []}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback via decision_function if available
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # convert to [0,1] via sigmoid-like transform
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            # no probabilities; use predictions
            y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        rocauc = roc_auc_score(y_test, y_prob)
    except Exception:
        rocauc = float("nan")

    metrics_summary[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(rocauc, 4) if not np.isnan(rocauc) else None
    }

    # Save model
    joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=150)
    plt.close()

    # ROC curve plot (if possible)
    try:
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name)
        plt.title(f"ROC Curve — {name}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"roc_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # For bar charts
    bar_metrics["model"].append(name)
    bar_metrics["accuracy"].append(acc)
    bar_metrics["precision"].append(prec)
    bar_metrics["recall"].append(rec)
    bar_metrics["f1"].append(f1)
    bar_metrics["roc_auc"].append(rocauc if not np.isnan(rocauc) else 0.0)

# Save metrics.json
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

# Bar charts of metric variation
def plot_bar(metric_key, values):
    plt.figure()
    x = np.arange(len(bar_metrics["model"]))
    plt.bar(x, values)
    plt.xticks(x, bar_metrics["model"], rotation=30, ha="right")
    plt.ylabel(metric_key.capitalize())
    plt.title(f"{metric_key.capitalize()} by Model")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"bar_{metric_key}.png", dpi=150)
    plt.close()

for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    plot_bar(k, bar_metrics[k])

print("Training complete. Models and plots saved.")
