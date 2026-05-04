
import pandas as pd


df = pd.read_csv("availability_verified.csv")

#Filter videos with both comments and transcript actually available
df["comments_actually_available"]   = df["comments_actually_available"].astype(str).str.lower().eq("true")
df["transcript_actually_available"] = df["transcript_actually_available"].astype(str).str.lower().eq("true")

both_ok = df[df["comments_actually_available"] & df["transcript_actually_available"]].copy()
print(f"Videos with both available: {len(both_ok)}")

both_ok["label_name"] = both_ok["label"].map({0: "non-clickbait", 1: "clickbait"})

print(f"\nLabel distribution before balancing:")
print(both_ok["label_name"].value_counts().to_string())

#Balance clickbait vs non-clickbait
clickbait     = both_ok[both_ok["label"] == 1]
non_clickbait = both_ok[both_ok["label"] == 0]

min_count = min(len(clickbait), len(non_clickbait))
print(f"\nBalancing to {min_count} samples per class")

balanced = pd.concat([
    clickbait.sample(n=min_count, random_state=42),
    non_clickbait.sample(n=min_count, random_state=42),
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nLabel distribution after balancing:")
print(balanced["label_name"].value_counts().to_string())
print(f"\nTotal rows saved: {len(balanced)}")

balanced.to_csv("filtered_data.csv", index=False)
print("Saved to filtered_data.csv")
