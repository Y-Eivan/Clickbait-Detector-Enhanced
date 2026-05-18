"""
Model 2 — Enhanced SVM on Block A + B + C with tuned hyperparameters.

Per the framing (A) methodology decision: each enhanced model is tuned
on its own feature space via 5-fold CV grid search on the training set.
This gives the enhanced feature set its best representation, instead of
inheriting Vierti's gamma=4.1 which was tuned for a 28-dim space.

The Vierti hyperparameters (C=3.7, gamma=4.1) are included in the grid
so this configuration is still reachable if it happens to be optimal.
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from data_loader import load_features
from evaluate import evaluate_and_save


def main():
    _, _, X_train_ABC, X_test_ABC, y_train, y_test, _ = load_features()

    print("Training Model 2: Enhanced SVM (Block A+B+C) with grid search...")

    # Small grid: 5 × 6 = 30 combos × 5-fold CV = 150 fits. Fast enough.
    # Includes Vierti's (C=3.7, gamma=4.1) so we can recover it if optimal.
    param_grid = {
        "C": [0.1, 1, 3.7, 10, 30],
        "gamma": ["scale", "auto", 0.01, 0.1, 1, 4.1],
    }

    grid = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_ABC, y_train)

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")

    model = grid.best_estimator_
    y_pred = model.predict(X_test_ABC)
    evaluate_and_save("model2_enhanced_svm", model, y_test, y_pred)


if __name__ == "__main__":
    main()