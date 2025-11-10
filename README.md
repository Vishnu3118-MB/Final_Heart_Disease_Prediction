# Heart Disease Prediction — Final Year Project

A creative, design-forward ML web app that predicts heart disease from patient inputs using multiple algorithms,
compares their performance with metrics and charts, and lets you download a patient report as a PDF.

## Features
- Trains **multiple algorithms** (Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, Gaussian NB)
- Shows **performance variations** (Accuracy, Precision, Recall, F1, ROC-AUC) + charts (ROC, Confusion Matrix, bar charts)
- **Interactive web UI** (Flask) with a clean design
- **Algorithm selector** next to the Predict button
- **PDF report** download for each patient prediction
- Fully reproducible with your dataset at `data/heart_clean.csv`

## Project Structure
```
heart_ai_project/
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── layout.html
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       ├── css/style.css
│       └── img/   (generated plots)
├── data/heart_clean.csv
├── models/  (trained models + metrics.json)
├── reports/ (generated PDFs)
├── downloads/
├── notebooks/ (optional experimentation)
├── train.py
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1) Create virtual env (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train models and generate metrics/plots
python train.py

# 4) Run the web app
cd app
python app.py
# Open http://127.0.0.1:5000
```
