"""
Shared evaluation utilities. Prints a report and saves:
  - metrics.json    (accuracy/precision/recall/f1)
  - predictions.csv (y_true, y_pred)
  - model.pkl       (the trained estimator)

so that you can re-load results for the paper without retraining.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_and_save(model_name, model, y_true, y_pred,
                      results_dir="results"):
    """Print metrics + save artifacts for downstream analysis."""
    results_dir = Path(results_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Print
    print(f"\n{'='*52}")
    print(f"{model_name}")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"  Confusion Matrix:\n{cm}")

    # Save metrics
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}) \
        .to_csv(results_dir / "predictions.csv", index=False)

    # Save trained model
    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"\n  → saved to {results_dir}/")
    return metrics