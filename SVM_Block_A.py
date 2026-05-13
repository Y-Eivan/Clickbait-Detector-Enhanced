import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

#load data
df = pd.read_csv("filtered_data.csv")

#word2vec on vid titles
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


ENG_COLS = ["video_views", "video_likes", "video_comments"]
engagement = df[ENG_COLS].fillna(0).values.astype(float)
engagement_log = np.log1p(engagement)

#combine and scaling
block_a_raw = np.hstack([title_vecs, engagement_log])  # (5722, 28)

scaler = MinMaxScaler()
block_a_scaled = scaler.fit_transform(block_a_raw)

#save to csv
feat_cols = (
    [f"w2v_{i}" for i in range(25)] +
    [f"log_{c}" for c in ENG_COLS]
)
out = pd.DataFrame(block_a_scaled, columns=feat_cols)
out.insert(0, "video_id", df["video_id"].values)
out.insert(1, "label", df["label"].values)
out.to_csv("block_a_features.csv", index=False)

print(f"Block A saved:{out.shape}") 