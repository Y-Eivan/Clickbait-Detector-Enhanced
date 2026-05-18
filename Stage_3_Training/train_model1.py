"""
Model 1 — Baseline SVM on Block A only.

Purpose: replicate Vierti et al. SVM performance on the 5722-video balanced
subset. Hyperparameters fixed to Vierti's reported values (C=3.7, gamma=4.1)
so this is a true replication, not a re-tuned baseline.
"""

from sklearn.svm import SVC

from data_loader import load_features
from evaluate import evaluate_and_save


def main():
    X_train_A, X_test_A, _, _, y_train, y_test, _ = load_features()

    print("Training Model 1: Baseline SVM (Block A only)...")
    model = SVC(C=3.7, gamma=4.1, kernel="rbf", random_state=42)
    model.fit(X_train_A, y_train)

    y_pred = model.predict(X_test_A)
    evaluate_and_save("model1_baseline_svm", model, y_test, y_pred)


if __name__ == "__main__":
    main()