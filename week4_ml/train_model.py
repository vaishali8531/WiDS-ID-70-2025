import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. Load feature data
# ===============================

DATA_PATH = "../data/image_features.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# IMPORTANT:
# For now, we create dummy labels
# Later, these can be replaced with real annotations
# -------------------------------

np.random.seed(42)
df["label"] = np.random.randint(0, 2, size=len(df))

X = df[["mean_intensity", "std_intensity", "edge_density"]]
y = df["label"]

# ===============================
# 2. Train-test split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ===============================
# 3. Train models
# ===============================

models = {
    "SVM": SVC(kernel="rbf", probability=True),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

best_model = None
best_f1 = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# ===============================
# 4. Save best model
# ===============================

MODEL_PATH = "../data/best_model.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print("\nBest model saved as best_model.pkl")