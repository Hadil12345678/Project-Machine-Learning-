import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")

    x_train_path = os.path.join(outputs_dir, "X_train.csv")
    y_train_path = os.path.join(outputs_dir, "y_train.csv")
    x_test_path = os.path.join(outputs_dir, "X_test.csv")
    y_test_path = os.path.join(outputs_dir, "y_test.csv")
    model_out_path = os.path.join(outputs_dir, "best_model_smote.pkl")

    # Load data
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze().astype(int).values
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze().astype(int).values

    # Model pipeline
    model = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=1000,
                random_state=42,
            )),
        ]
    )

    # Train
    model.fit(x_train, y_train)

    # ✅ SAVE MODEL
    joblib.dump(model, model_out_path)

    # ✅ SAVE COLUMNS (🔥 VERY IMPORTANT)
    columns_path = os.path.join(outputs_dir, "columns.pkl")
    joblib.dump(x_train.columns.tolist(), columns_path)

    print(f"Saved model  : {model_out_path}")
    print(f"Saved columns: {columns_path}")

    # Evaluate
    y_prob = model.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = float(np.mean(y_pred_05 == y_test))

    print("=== SMOTE MODEL EVALUATION ===")
    print(f"Samples test : {len(y_test)}")
    print(f"Positives    : {int(np.sum(y_test))}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"PR-AUC       : {pr_auc:.4f}")
    print(f"Brier score  : {brier:.4f}")
    print(f"Accuracy@0.5 : {acc_05:.4f}")


if __name__ == "__main__":
    main()