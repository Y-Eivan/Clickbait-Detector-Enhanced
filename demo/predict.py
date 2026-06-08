"""
YouTube Clickbait Detector — Demo CLI

Two pools are supported:
  Working set (5,722 videos) — full pipeline: Block A + VADER + SBERT -> Model 2
                               Features loaded from pre-computed CSVs (fast, no GPU).
  Vierti leftovers (~31k)   — metadata only:  Block A only            -> Model 1
                               Block A re-computed on the fly via saved W2V + scaler.

Usage:
  python predict.py <video_id>
  python predict.py --random [--clickbait | --notclickbait] [--vierti]

Flags:
  --random        Pick a random video from the pool
  --clickbait     Restrict random pick to clickbait videos
  --notclickbait  Restrict random pick to non-clickbait videos
  --vierti        Use Vierti leftover pool instead of working set

Run build_artifacts.py once before first use.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pickle
import argparse
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec

ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
CSV_DIR   = ROOT / "Stage_3_Training" / "csv"

# ── Terminal colours (no external dependency) ──────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Artifact + data loading ────────────────────────────────────────

def load_artifacts():
    needed = [
        ARTIFACTS / "w2v.model",
        ARTIFACTS / "scaler.pkl",
        ARTIFACTS / "model1_baseline_svm.pkl",
        ARTIFACTS / "model2_enhanced_svm.pkl",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("ERROR: missing artifacts — run  python demo/build_artifacts.py  first.")
        for p in missing:
            print(f"  missing: {p.name}")
        sys.exit(1)

    print("Loading artifacts...", end=" ", flush=True)
    w2v = Word2Vec.load(str(ARTIFACTS / "w2v.model"))
    with open(ARTIFACTS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS / "model1_baseline_svm.pkl", "rb") as f:
        model1 = pickle.load(f)
    with open(ARTIFACTS / "model2_enhanced_svm.pkl", "rb") as f:
        model2 = pickle.load(f)
    print("done.")
    return w2v, scaler, model1, model2


def load_data():
    print("Loading CSVs...", end=" ", flush=True)

    filtered = pd.read_csv(
        ROOT / "Stage_1_Extraction_Validation_Filtering/filtered_data.csv",
        dtype={"video_id": str},
    )
    vierti = pd.read_csv(
        ROOT / "Stage_2_SVM_Vader_SBert/A/vierti_dataset.csv",
        dtype={"video_id": str},
    ).drop_duplicates(subset="video_id", keep="first")

    # Merge dislikes into filtered (filtered_data lacks video_dislikes)
    filtered = filtered.merge(
        vierti[["video_id", "video_dislikes"]], on="video_id", how="left"
    )
    filtered["video_dislikes"] = filtered["video_dislikes"].fillna(0)

    # Pre-computed feature blocks (used for working set predictions)
    block_a = pd.read_csv(CSV_DIR / "block_a_features.csv", dtype={"video_id": str})
    block_b = pd.read_csv(CSV_DIR / "block_b_features.csv", dtype={"video_id": str})
    block_c = pd.read_csv(CSV_DIR / "block_c_features.csv", dtype={"video_id": str})

    print("done.")
    return filtered, vierti, block_a, block_b, block_c


# ── Feature computation (Vierti-only path) ─────────────────────────

def tokenize(title: str) -> list:
    return str(title).lower().split()


def avg_vec(title: str, w2v) -> np.ndarray:
    tokens = tokenize(title)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(25)


def compute_block_a(title, views, likes, dislikes, n_comments, w2v, scaler) -> np.ndarray:
    vec = avg_vec(title, w2v)
    eng = np.log1p([views, likes, dislikes, n_comments])
    raw = np.hstack([vec, eng]).reshape(1, -1)
    return scaler.transform(raw)


# ── Core prediction logic ──────────────────────────────────────────

def predict(video_id, filtered, vierti, block_a_df, block_b_df, block_c_df,
            w2v, scaler, model1, model2):

    # ── Working set path (full pipeline, Model 2) ──────────────────
    fd_row = filtered[filtered["video_id"] == video_id]
    if len(fd_row) > 0:
        row = fd_row.iloc[0]

        a_row = block_a_df[block_a_df["video_id"] == video_id]
        b_row = block_b_df[block_b_df["video_id"] == video_id]
        c_row = block_c_df[block_c_df["video_id"] == video_id]

        if len(a_row) == 0 or len(b_row) == 0 or len(c_row) == 0:
            return None  # data alignment issue

        a_cols = [c for c in block_a_df.columns if c.startswith("w2v_") or c.startswith("log_")]
        b_cols = ["mean_compound", "std_compound", "pct_positive", "pct_negative"]
        c_cols = ["sbert_similarity"]

        A = a_row[a_cols].values
        B = b_row[b_cols].values
        C = c_row[c_cols].values
        X = np.hstack([A, B, C])

        pred  = int(model2.predict(X)[0])
        score = float(model2.decision_function(X)[0])

        vader = {k: float(b_row.iloc[0][k]) for k in b_cols}

        return {
            "video_id":        video_id,
            "title":           row["video_title"],
            "channel":         row.get("channel_name", "N/A"),
            "views":           int(row.get("video_views") or 0),
            "likes":           int(row.get("video_likes") or 0),
            "dislikes":        int(row.get("video_dislikes") or 0),
            "comments_count":  int(row.get("video_comments") or 0),
            "true_label":      int(row["label"]),
            "predicted_label": pred,
            "score":           score,
            "model_name":      "Enhanced SVM  (Model 2 — Block A + VADER + SBERT)",
            "vader":           vader,
            "sbert":           float(c_row.iloc[0]["sbert_similarity"]),
            "pool":            "working",
        }

    # ── Vierti leftover path (Block A only, Model 1) ───────────────
    v_row = vierti[vierti["video_id"] == video_id]
    if len(v_row) > 0:
        row = v_row.iloc[0]

        A = compute_block_a(
            row["video_title"],
            float(row.get("video_views") or 0),
            float(row.get("video_likes") or 0),
            float(row.get("video_dislikes") or 0),
            float(row.get("video_comments") or 0),
            w2v, scaler,
        )
        pred  = int(model1.predict(A)[0])
        score = float(model1.decision_function(A)[0])

        return {
            "video_id":        video_id,
            "title":           row["video_title"],
            "channel":         row.get("channel_name", "N/A"),
            "views":           int(row.get("video_views") or 0),
            "likes":           int(row.get("video_likes") or 0),
            "dislikes":        int(row.get("video_dislikes") or 0),
            "comments_count":  int(row.get("video_comments") or 0),
            "true_label":      int(row["label"]),
            "predicted_label": pred,
            "score":           score,
            "model_name":      "Baseline SVM  (Model 1 — Block A only)",
            "vader":           None,
            "sbert":           None,
            "pool":            "vierti",
        }

    return None


# ── Display ────────────────────────────────────────────────────────

def display(info: dict):
    W   = 62
    bar = "-" * W

    pred_label = "CLICKBAIT"     if info["predicted_label"] == 1 else "NOT CLICKBAIT"
    true_label = "CLICKBAIT"     if info["true_label"]      == 1 else "NOT CLICKBAIT"
    pred_color = RED              if info["predicted_label"] == 1 else GREEN
    correct    = info["predicted_label"] == info["true_label"]
    verdict    = f"{GREEN}correct{RESET}" if correct else f"{RED}wrong{RESET}"

    conf = sigmoid(abs(info["score"])) * 100

    print()
    print(f"{BOLD}{bar}{RESET}")
    print(f"{BOLD}  YouTube Clickbait Detector{RESET}")
    print(bar)
    print(f"  Video ID  : {CYAN}{info['video_id']}{RESET}")
    print(f"  Title     : {info['title']}")
    print(f"  Channel   : {info['channel']}")
    print()
    print(f"  {BOLD}Prediction : {pred_color}{pred_label}{RESET}  ({verdict})")
    print(f"  True label : {true_label}")
    print(f"  Confidence : {conf:.1f}%  {DIM}(decision score {info['score']:+.3f}){RESET}")
    print(f"  Model      : {info['model_name']}")
    print()
    print(f"  {BOLD}Engagement{RESET}")
    print(f"    Views    : {info['views']:>12,}")
    print(f"    Likes    : {info['likes']:>12,}")
    print(f"    Dislikes : {info['dislikes']:>12,}")
    print(f"    Comments : {info['comments_count']:>12,}")

    if info["vader"] is not None:
        v = info["vader"]
        bar_pos = "|" * max(1, int(v["pct_positive"] * 20))
        bar_neg = "|" * max(1, int(v["pct_negative"] * 20))
        print()
        print(f"  {BOLD}VADER Comment Sentiment{RESET}")
        print(f"    Mean compound : {v['mean_compound']:+.3f}")
        print(f"    % Positive    : {v['pct_positive']*100:5.1f}%  {GREEN}{bar_pos}{RESET}")
        print(f"    % Negative    : {v['pct_negative']*100:5.1f}%  {RED}{bar_neg}{RESET}")

    if info["sbert"] is not None:
        s = info["sbert"]
        alignment = "HIGH" if s > 0.5 else "LOW"
        a_color   = GREEN  if s > 0.5 else RED
        note      = "title matches content" if s > 0.5 else "title diverges from content"
        print()
        print(f"  {BOLD}SBERT Title <-> Transcript Similarity{RESET}")
        print(f"    Score     : {s:.4f}")
        print(f"    Alignment : {a_color}{alignment}{RESET}  ({note})")

    if info["pool"] == "vierti":
        print()
        print(f"  {YELLOW}No transcript or comments on file for this video.{RESET}")
        print(f"  {YELLOW}Prediction uses title + engagement features only.{RESET}")

    print(bar)
    print()


# ── Entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Clickbait Detector — demo CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py St8vFW9Kg-c\n"
            "  python predict.py --random\n"
            "  python predict.py --random --clickbait\n"
            "  python predict.py --random --notclickbait\n"
            "  python predict.py --random --vierti\n"
            "  python predict.py --random --clickbait --vierti\n"
        ),
    )
    parser.add_argument("video_id", nargs="?", help="YouTube video ID")
    parser.add_argument("--random",       action="store_true", help="Pick a random video")
    parser.add_argument("--clickbait",    action="store_true", help="Restrict to clickbait videos")
    parser.add_argument("--notclickbait", action="store_true", help="Restrict to non-clickbait videos")
    parser.add_argument("--vierti",       action="store_true",
                        help="Use Vierti leftover pool (metadata only, Model 1)")
    args = parser.parse_args()

    if not args.video_id and not args.random:
        parser.print_help()
        sys.exit(0)

    w2v, scaler, model1, model2 = load_artifacts()
    filtered, vierti, block_a_df, block_b_df, block_c_df = load_data()

    if args.random:
        if args.vierti:
            pool = vierti[~vierti["video_id"].isin(filtered["video_id"])].copy()
        else:
            pool = filtered.copy()

        if args.clickbait:
            pool = pool[pool["label"] == 1]
        elif args.notclickbait:
            pool = pool[pool["label"] == 0]

        if len(pool) == 0:
            print("No videos match the given filters.")
            sys.exit(1)

        video_id = str(random.choice(pool["video_id"].tolist()))
        print(f"Randomly selected: {CYAN}{video_id}{RESET}")
    else:
        video_id = args.video_id

    info = predict(
        video_id, filtered, vierti,
        block_a_df, block_b_df, block_c_df,
        w2v, scaler, model1, model2,
    )

    if info is None:
        print(f"\nVideo ID '{video_id}' not found.")
        print("  Working set : 5,722 videos with transcript + comments (full pipeline)")
        print("  Vierti pool : ~31,000 videos metadata-only  (use --vierti to pick from here)")
        sys.exit(1)

    display(info)


if __name__ == "__main__":
    main()
