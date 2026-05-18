"""
Rebuild Block A with the dislike count recovered from Vierti's original
dataset. filtered_data.csv lost the video_dislikes column upstream, but
vierti_dataset.csv still has it. We merge on video_id, dedupe Vierti's
279 duplicate rows first, then proceed with the original Block A
construction protocol.

Result: block_a_features.csv with 29 features (25 w2v + 4 engagement).
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler


# --- Recover dislikes ---
df = pd.read_csv("filtered_data.csv")
vierti = pd.read_csv("vierti_dataset.csv")

# Vierti has 279 video_ids appearing twice with identical metadata; dedupe
vierti_dedup = vierti.drop_duplicates(subset="video_id", keep="first")

df = df.merge(
    vierti_dedup[["video_id", "video_dislikes"]],
    on="video_id", how="left"
)
assert df["video_dislikes"].isna().sum() == 0, \
    "Some filtered videos have no dislike data in Vierti — investigate"
print(f"Merged dislikes: {len(df)} rows, {df['video_dislikes'].isna().sum()} missing")


# --- Word2Vec on video titles (unchanged from original) ---
def tokenize(title: str) -> list[str]:
    return str(title).lower().split()

sentences = [tokenize(t) for t in df["video_title"]]

w2v = Word2Vec(
    sentences,
    vector_size=25,
    window=20,
    min_count=1,
    workers=4,
    epochs=30,
    seed=42,
)

def title_to_vector(title: str) -> np.ndarray:
    tokens = tokenize(title)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(25)

title_vecs = np.vstack([title_to_vector(t) for t in df["video_title"]])


# --- Engagement features (now includes dislikes) ---
ENG_COLS = ["video_views", "video_likes", "video_dislikes", "video_comments"]
engagement = df[ENG_COLS].fillna(0).values.astype(float)
engagement_log = np.log1p(engagement)


# --- Combine and scale ---
block_a_raw = np.hstack([title_vecs, engagement_log])  # (5722, 29)
scaler = MinMaxScaler()
block_a_scaled = scaler.fit_transform(block_a_raw)


# --- Save ---
feat_cols = (
    [f"w2v_{i}" for i in range(25)] +
    [f"log_{c}" for c in ENG_COLS]
)
out = pd.DataFrame(block_a_scaled, columns=feat_cols)
out.insert(0, "video_id", df["video_id"].values)
out.insert(1, "label", df["label"].values)
out.to_csv("block_a_features.csv", index=False)

print(f"Block A saved: {out.shape}")
print(f"Feature columns: {len(feat_cols)} (expected: 29)")