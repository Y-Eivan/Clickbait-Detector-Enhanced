"""
Loads feature blocks A, B, C and merges them on video_id.

This is the single source of truth for the train/test split — all three
model scripts import from here to guarantee identical splits.
"""

import pandas as pd
from pathlib import Path

# Expected dimensions per Stage 3 design.
# A = Word2Vec 25-dim + log1p(views, likes, dislikes, comments) = 29
# B = VADER aggregates: mean_compound, std_compound, pct_positive, pct_negative = 4
# C = SBERT cosine similarity title<->transcript = 1
EXPECTED_A = 29
EXPECTED_B = 4
EXPECTED_C = 1
EXPECTED_ROWS = 5722


def load_features(data_dir: str = "."):
    """
    Returns:
        X_train_A, X_test_A           — Block A only (baseline features)
        X_train_ABC, X_test_ABC       — Blocks A+B+C (enhanced features)
        y_train, y_test
        feature_info                  — dict with column names per block
    """
    data_dir = Path(data_dir)

    csv_dir = data_dir / "csv"
    split = pd.read_csv(csv_dir / "split_indices.csv")   
    block_a = pd.read_csv(csv_dir / "block_a_features.csv")
    block_b = pd.read_csv(csv_dir / "block_b_features.csv")
    block_c = pd.read_csv(csv_dir / "block_c_features.csv")


    for name, df in [("A", block_a), ("B", block_b), ("C", block_c)]:
        assert len(df) == EXPECTED_ROWS, \
            f"Block {name}: expected {EXPECTED_ROWS} rows, got {len(df)}"
        assert df.isna().sum().sum() == 0, f"Block {name}: contains NaN"

    # Merge on video_id
    df = split.copy()
    df = df.merge(block_a.drop(columns=["label"]), on="video_id", how="left")
    df = df.merge(block_b.drop(columns=["label"], errors="ignore"),
                  on="video_id", how="left")
    df = df.merge(block_c, on="video_id", how="left")
    assert df.isna().sum().sum() == 0, "NaN after merge — video_id alignment issue"

    # Define column groups
    a_cols = [c for c in df.columns if c.startswith("w2v_") or c.startswith("log_")]
    b_cols = ["mean_compound", "std_compound", "pct_positive", "pct_negative"]
    c_cols = ["sbert_similarity"]
    abc_cols = a_cols + b_cols + c_cols

    # Dimension assertions — fail loud if blocks drifted from spec
    assert len(a_cols) == EXPECTED_A, \
        f"Block A dim mismatch: expected {EXPECTED_A}, got {len(a_cols)} ({a_cols})"
    assert len(b_cols) == EXPECTED_B, f"Block B dim mismatch: got {len(b_cols)}"
    assert len(c_cols) == EXPECTED_C, f"Block C dim mismatch: got {len(c_cols)}"

    print(f"Feature dimensions — A:{len(a_cols)}, B:{len(b_cols)}, "
          f"C:{len(c_cols)}, Total:{len(abc_cols)}")

    # Split
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    # Sanity check: 1:1 ratio preserved by stratified split
    print(f"Train: {len(train)} rows, class balance: "
          f"{train['label'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Test:  {len(test)} rows, class balance: "
          f"{test['label'].value_counts(normalize=True).round(3).to_dict()}")

    feature_info = {"A": a_cols, "B": b_cols, "C": c_cols, "ABC": abc_cols}

    return (
        train[a_cols].values, test[a_cols].values,
        train[abc_cols].values, test[abc_cols].values,
        train["label"].values, test["label"].values,
        feature_info,
    )


if __name__ == "__main__":
    # Quick smoke test
    X_train_A, X_test_A, X_train_ABC, X_test_ABC, y_train, y_test, info = load_features()
    print(f"\nShapes:")
    print(f"  X_train_A:   {X_train_A.shape}")
    print(f"  X_test_A:    {X_test_A.shape}")
    print(f"  X_train_ABC: {X_train_ABC.shape}")
    print(f"  X_test_ABC:  {X_test_ABC.shape}")