import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------------------
# Load data (SESUIKAN PATH)
# ----------------------------------------
train_df = pd.read_csv("preprocessing/train_data.csv")
test_df = pd.read_csv("preprocessing/test_data.csv")

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# ----------------------------------------
# ATTACH KE RUN DARI MLFLOW PROJECT
# ----------------------------------------
run_id = os.environ.get("MLFLOW_RUN_ID")

with mlflow.start_run(run_id=run_id):

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Manual logging
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", acc)

    # Log model (WAJIB untuk Docker)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Training selesai | Accuracy: {acc:.4f}")
