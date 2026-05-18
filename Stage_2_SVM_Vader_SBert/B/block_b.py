import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Load data
master = pd.read_csv("master_5722.csv")
comments = pd.read_csv("comments.csv")

master["video_id"] = master["video_id"].astype(str)
comments["video_id"] = comments["video_id"].astype(str)
comments = comments.dropna(subset=["text"])

comments_by_vid = comments.groupby("video_id")["text"].apply(list).to_dict()

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
results = []

for vid in tqdm(master["video_id"], desc="VADER Processing"):
    vid_comments = comments_by_vid.get(vid, [])
    if len(vid_comments) == 0:
        results.append({
            "video_id": vid, "mean_compound": 0.0, "std_compound": 0.0,
            "pct_positive": 0.0, "pct_negative": 0.0,
        })
        continue

    compounds = np.array([analyzer.polarity_scores(str(c))["compound"] for c in vid_comments])

    results.append({
        "video_id": vid,
        "mean_compound": float(compounds.mean()),
        "std_compound": float(compounds.std()),
        "pct_positive": float((compounds > 0.3).mean()),
        "pct_negative": float((compounds < -0.3).mean()),
    })

# Save output
out = pd.DataFrame(results)
out.to_csv("block_b_features.csv", index=False)