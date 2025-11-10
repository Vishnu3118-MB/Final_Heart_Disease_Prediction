import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Safe path handling ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load dataset ---
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_clean.csv")
df = pd.read_csv(DATA_PATH)

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# --- Define models ---
models = {
    "Logistic_Regression": LogisticRegression(max_iter=200),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
    "Naive_Bayes": GaussianNB(),
}

# --- Train and evaluate models ---
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results.append([name, acc, prec, rec, f1, roc])

    # Save model to models folder
    model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Saved {name} to {model_path}")

# --- Save results summary ---
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
results_df.to_csv(os.path.join(MODELS_DIR, "model_performance.csv"), index=False)

print("\n🎯 All models trained and saved successfully!")
print(results_df)
