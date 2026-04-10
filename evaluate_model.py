import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary classification model.")
    parser.add_argument(
        "--model",
        default="best_model_tuned.pkl",
        help="Model file name located in outputs/.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "outputs", args.model)
    x_test_path = os.path.join(base_dir, "outputs", "X_test.csv")
    y_test_path = os.path.join(base_dir, "outputs", "y_test.csv")

    model = joblib.load(model_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze().astype(int).values

    y_prob = model.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    base_rate = float(np.mean(y_test))

    print("=== MODEL EVALUATION ===")
    print(f"Samples      : {len(y_test)}")
    print(f"Positives    : {int(np.sum(y_test))}")
    print(f"Base rate    : {base_rate:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"PR-AUC       : {pr_auc:.4f}")
    print(f"Brier score  : {brier:.4f}")
    print()

    for threshold in [0.20, 0.50, 0.90]:
        m = metrics_at_threshold(y_test, y_prob, threshold)
        print(f"--- Threshold {m['threshold']:.2f} ---")
        print(f"Confusion matrix: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
        print(
            "Accuracy={:.3f} Precision={:.3f} Recall={:.3f} Specificity={:.3f} F1={:.3f}".format(
                m["accuracy"],
                m["precision"],
                m["recall"],
                m["specificity"],
                m["f1"],
            )
        )
        print()

    print("Tip: choose threshold based on your goal.")
    print("- Higher recall (catch more positives): lower threshold.")
    print("- Higher precision (fewer false alarms): higher threshold.")


if __name__ == "__main__":
    main()
