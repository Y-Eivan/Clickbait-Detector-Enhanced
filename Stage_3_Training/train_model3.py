"""
Model 3 — Enhanced Random Forest on Block A + B + C with tuned hyperparameters.

Per the framing (A) methodology decision: each enhanced model is tuned
on its own feature space via 5-fold CV grid search on the training set.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data_loader import load_features
from evaluate import evaluate_and_save


def main():
    _, _, X_train_ABC, X_test_ABC, y_train, y_test, _ = load_features()

    print("Training Model 3: Enhanced RF (Block A+B+C) with grid search...")

    # Small RF grid: 3 × 3 × 2 = 18 combos × 5-fold CV = 90 fits.
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 20, 50],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
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
    evaluate_and_save("model3_enhanced_rf", model, y_test, y_pred)


if __name__ == "__main__":
    main()