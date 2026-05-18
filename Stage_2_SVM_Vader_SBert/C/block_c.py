"""
Block C — SBERT cosine similarity feature
Input:  master_5722.csv   (video_id)
        transcripts.csv   (video_id, transcript_text, ...)
        filtered_data.csv (video_id, video_title, ...)
Output: block_c_features.csv (video_id, sbert_similarity)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ── Load ───────────────────────────────────────────────────────────
master      = pd.read_csv("master_5722.csv")
transcripts = pd.read_csv("transcripts.csv")
filtered    = pd.read_csv("filtered_data.csv")[["video_id", "video_title"]]

master["video_id"]      = master["video_id"].astype(str)
transcripts["video_id"] = transcripts["video_id"].astype(str)
filtered["video_id"]    = filtered["video_id"].astype(str)

# Merge title into transcripts, then align to master's 5722
df = transcripts[["video_id", "transcript_text"]].merge(
    filtered, on="video_id", how="inner"
)
df = master.merge(df, on="video_id", how="left")

# Warn and fill if anything is missing
missing_titles      = df["video_title"].isna().sum()
missing_transcripts = df["transcript_text"].isna().sum()
if missing_titles > 0:
    print(f"WARNING:{missing_titles} videos have no title — filling with empty string")
if missing_transcripts > 0:
    print(f"WARNING:{missing_transcripts} videos have no transcript — filling with empty string")

df["video_title"]     = df["video_title"].fillna("").astype(str)
df["transcript_text"] = df["transcript_text"].fillna("").astype(str)

# ── Model ──────────────────────────────────────────────────────────
# DO NOT manually truncate the transcript.
# model.encode() truncates at 256 subword tokens internally via the tokenizer.
# Word-count truncation gives a different (wrong) cutoff.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
assert model.max_seq_length == 256, f"Unexpected max_seq_length:{model.max_seq_length}"

print(f"GPU available:{torch.cuda.is_available()}")
print(f"Encoding{len(df)} title+transcript pairs...")

titles      = df["video_title"].tolist()
transcripts_text = df["transcript_text"].tolist()

# Batched encode — do not loop per-row, this is ~50-100x faster
title_embs = model.encode(
    titles,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True,
)
transcript_embs = model.encode(
    transcripts_text,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True,
)

# Vectorised pairwise cosine: title[i] with transcript[i]
# F.cosine_similarity normalises internally — do NOT use raw dot product
similarities = F.cosine_similarity(title_embs, transcript_embs, dim=1).cpu().numpy()

# ── Validate ───────────────────────────────────────────────────────
out = pd.DataFrame({
    "video_id":        df["video_id"].values,
    "sbert_similarity": similarities.astype(float),
})

assert len(out) == 5722,            f"Expected 5722 rows, got{len(out)}"
assert out.isna().sum().sum() == 0, "NaN found"
assert (out["sbert_similarity"].abs() <= 1.0 + 1e-6).all(), \
    "Cosine value out of [-1, 1] — something is wrong"

# If mean is near 1.0, you probably encoded the same column twice
mean_sim = out["sbert_similarity"].mean()
if mean_sim > 0.95:
    print(f"WARNING: mean similarity is{mean_sim:.4f} — did you encode title vs title by mistake?")

# ── Save ───────────────────────────────────────────────────────────
print(out["sbert_similarity"].describe())
out.to_csv("block_c_features.csv", index=False)
print(f"Saved block_c_features.csv —{len(out)} rows,{out.shape[1]} columns")