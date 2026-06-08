"""
One-time setup — run this before using predict.py.

Trains the Word2Vec model and fits the MinMaxScaler on the same data
used during training (Stage_2_SVM_Vader_SBert/A), then saves them to
demo/artifacts/ alongside copies of the two trained SVM models.

Run from the Clickbait-Detector root:
    python demo/build_artifacts.py
"""

import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ── Load the same data used in SVM_Block_A.py ──────────────────────
print("Loading data...")
filtered = pd.read_csv(ROOT / "Stage_1_Extraction_Validation_Filtering/filtered_data.csv")
vierti   = pd.read_csv(ROOT / "Stage_2_SVM_Vader_SBert/A/vierti_dataset.csv")
vierti   = vierti.drop_duplicates(subset="video_id", keep="first")

# Merge dislikes (not in filtered_data)
df = filtered.merge(vierti[["video_id", "video_dislikes"]], on="video_id", how="left")
df["video_dislikes"] = df["video_dislikes"].fillna(0)
print(f"  {len(df)} videos loaded")

# ── Word2Vec (identical config to SVM_Block_A.py) ──────────────────
def tokenize(title: str) -> list:
    return str(title).lower().split()

sentences = [tokenize(t) for t in df["video_title"]]
print("Training Word2Vec (vector_size=25, window=20, epochs=30)...")
w2v = Word2Vec(
    sentences,
    vector_size=25,
    window=20,
    min_count=1,
    workers=4,
    epochs=30,
    seed=42,
)
w2v.save(str(ARTIFACTS / "w2v.model"))
print(f"  Saved w2v.model  ({len(w2v.wv)} vocab tokens)")

# ── MinMaxScaler (fitted on same Block A raw features) ─────────────
ENG_COLS = ["video_views", "video_likes", "video_dislikes", "video_comments"]
engagement_log = np.log1p(df[ENG_COLS].fillna(0).values.astype(float))

def avg_vec(title: str) -> np.ndarray:
    tokens = tokenize(title)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(25)

print("Building title vectors for scaler fit...")
title_vecs = np.vstack([avg_vec(t) for t in df["video_title"]])
block_a_raw = np.hstack([title_vecs, engagement_log])  # (5722, 29)

scaler = MinMaxScaler()
scaler.fit(block_a_raw)
with open(ARTIFACTS / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Saved scaler.pkl")

# ── Copy trained SVM model PKLs ────────────────────────────────────
for name in ["model1_baseline_svm", "model2_enhanced_svm"]:
    src = ROOT / f"Stage_3_Training/results/{name}/model.pkl"
    dst = ARTIFACTS / f"{name}.pkl"
    shutil.copy(src, dst)
    print(f"  Copied {name}.pkl")

print("\nAll artifacts saved to demo/artifacts/  — ready to run predict.py")
