
import pandas as pd
from sklearn.model_selection import train_test_split

#Config
INPUT_CSV       = "filtered_data.csv"
SPLIT_OUT       = "split_indices.csv"
MASTER_OUT      = "master_5722.csv"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
EXPECTED_ROWS   = 5722


df = pd.read_csv(INPUT_CSV)

#Validate input
assert "video_id" in df.columns, "filtered_data.csv must have a 'video_id' column"
assert "label" in df.columns,    "filtered_data.csv must have a 'label' column"
assert len(df) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, got {len(df)}. "
    f"Make sure you're using the final filtered dataset."
)

#onfirm 1:1 balance
vc = df["label"].value_counts()
print(f"Label distribution:\n{vc.to_string()}\n")
assert len(vc) == 2, "Expected exactly 2 label classes"
assert abs(vc.iloc[0] - vc.iloc[1]) <= 2, (
    f"Dataset is not balanced — found {vc.to_dict()}. Check your filtered_data.csv."
)

#Split the data into 80% train and 20% test
train_ids, test_ids = train_test_split(
    df["video_id"],
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

split_col = df["video_id"].map(
    {vid: "train" for vid in train_ids} | {vid: "test" for vid in test_ids}
)

split_df = pd.DataFrame({
    "video_id": df["video_id"],
    "label":    df["label"],
    "split":    split_col,
})

#Sanity checking 
n_train = (split_df["split"] == "train").sum()
n_test  = (split_df["split"] == "test").sum()
 
assert n_train + n_test == EXPECTED_ROWS, "Row count mismatch after split"
assert abs(n_test - round(EXPECTED_ROWS * TEST_SIZE)) <= 1, (
    f"Test size mismatch: expected ~{round(EXPECTED_ROWS * TEST_SIZE)}, got {n_test}"
)


#Check for class balance 
for split_name in ["train", "test"]:
    sub = split_df[split_df["split"] == split_name]["label"].value_counts()
    diff = abs(sub.iloc[0] - sub.iloc[1])
    assert diff <= 2, f"Class imbalance in {split_name} split: {sub.to_dict()}"
 
print(f"Split summary")
print(f"  Train : {n_train} rows  ({n_train/EXPECTED_ROWS*100:.1f}%)")
print(f"  Test  : {n_test}  rows  ({n_test/EXPECTED_ROWS*100:.1f}%)")
print()
for split_name in ["train", "test"]:
    sub = split_df[split_df["split"] == split_name]["label"].value_counts().to_dict()
    print(f"  {split_name} label distribution: {sub}")


#Save to csv
split_df.to_csv(SPLIT_OUT, index=False)
print(f"\nSaved: {SPLIT_OUT}  ({len(split_df)} rows)")

master_df = split_df[["video_id"]].copy()
master_df.to_csv(MASTER_OUT, index=False)
print(f"Saved: {MASTER_OUT}  ({len(master_df)} rows)")
