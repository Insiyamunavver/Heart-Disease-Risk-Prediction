import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("C:\\Users\\Taha\\OneDrive\\Desktop\\heart_risk_prediction\\Heart_Disease_Prediction.csv")

# Target encoding
df["Heart Disease"] = df["Heart Disease"].map({
    "Presence": 1,
    "Absence": 0
})

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Models to compare
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

best_model = None
best_auc = 0
best_model_name = ""

print("\n🔍 Training models...\n")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}\n")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# ----------------------------
# Save best model
# ----------------------------
pickle.dump(best_model, open("best_heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print(f"✅ Best model selected: {best_model_name}")
