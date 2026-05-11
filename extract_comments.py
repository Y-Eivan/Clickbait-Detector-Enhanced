import csv
import os
import sys
import time
import signal
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ============ CONFIG ============
API_KEYS = [
    "AIzaSyCqJMOEtA5alkIbyXRqkp6tX8n4ZTZnQ9c",
    "AIzaSyBS08Jm7ksgU91zNlO_Q2T1drMGVP6frVw",
    "AIzaSyCK2hSLFqFQYugMbkNcIwxx09-LZjxeXrc",
    "AIzaSyBU3cnQ1uXMaq9-gXyCqI9EdJL8rbbeE7Q"
]

INPUT_CSV = "filtered_data.csv"
OUTPUT_CSV = "comments.csv"
FAILED_CSV = "comments_failed.csv"

COMMENTS_PER_VIDEO = 100
ORDER = "relevance"        # 'relevance' or 'time'. Relevance = top-comments view.
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 60.0
PROGRESS_EVERY = 1
# ================================

COMMENT_FIELDS = [
    "video_id", "comment_id", "text",
    "like_count", "reply_count", "published_at", "author",
]
FAILED_FIELDS = ["video_id", "reason"]


# Rotates API keys when one runs out of quota
class KeyRotator:
    def __init__(self, keys):
        valid = [k for k in keys if k and not k.startswith("YOUR_KEY")]
        if not valid:
            raise ValueError("Set your real API keys in API_KEYS first.")
        self.keys = valid
        self.idx = 0
        self.exhausted = set()

    def current(self):
        return self.keys[self.idx]

    # Move to next live key, None if all dead
    def rotate(self):
        self.exhausted.add(self.idx)
        for _ in range(len(self.keys)):
            self.idx = (self.idx + 1) % len(self.keys)
            if self.idx not in self.exhausted:
                return self.keys[self.idx]
        return None  # all dead


def build_client(key):
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def fetch_comments(youtube, video_id, max_comments, order):
    """Fetch up to max_comments top-level comments. May raise HttpError."""
    items = []
    page_token = None
    # Page through commentThreads until full or no more pages
    while len(items) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(items)),
            order=order,
            textFormat="plainText",
            pageToken=page_token,
        )
        resp = req.execute()
        # Flatten each thread into a row
        for entry in resp.get("items", []):
            top = entry["snippet"]["topLevelComment"]["snippet"]
            items.append({
                "video_id": video_id,
                "comment_id": entry["id"],
                "text": top.get("textOriginal", ""),
                "like_count": top.get("likeCount", 0),
                "reply_count": entry["snippet"].get("totalReplyCount", 0),
                "published_at": top.get("publishedAt", ""),
                "author": top.get("authorDisplayName", ""),
            })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items[:max_comments]


def process_video(youtube, rotator, vid):
    """
    Returns (status, payload, youtube_client).
      status in {'ok', 'fail', 'quota_dead'}
      payload = list of comment dicts (ok), reason str (fail), None (quota_dead)
    The youtube client may have been swapped out due to key rotation.
    """
    backoff = INITIAL_BACKOFF
    order = ORDER
    for attempt in range(MAX_RETRIES):
        try:
            rows = fetch_comments(youtube, vid, COMMENTS_PER_VIDEO, order)
            return "ok", rows, youtube
        except HttpError as e:
            status = e.resp.status
            content = (e.content or b"").decode("utf-8", errors="ignore")

            # Quota exceeded -> rotate key, retry same video
            if status == 403 and "quota" in content.lower():
                new_key = rotator.rotate()
                if new_key is None:
                    return "quota_dead", None, youtube
                print(f"  ↳ quota hit; rotated to key #{rotator.idx}")
                youtube = build_client(new_key)
                continue

            # Comments off / disabled
            if status == 403 and "commentsDisabled" in content:
                return "fail", "commentsDisabled", youtube

            # Bad request -> sometimes order='relevance' rejected; fall back once
            if status == 400 and order == "relevance":
                order = "time"
                continue

            # Video gone
            if status == 404:
                return "fail", "videoNotFound", youtube

            # Transient
            if status == 429 or status >= 500:
                if attempt == MAX_RETRIES - 1:
                    return "fail", f"transient_{status}", youtube
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            return "fail", f"http_{status}: {content[:150]}", youtube

        except Exception as e:
            return "fail", f"{type(e).__name__}: {str(e)[:150]}", youtube

    return "fail", "max_retries", youtube


# Resume helper: video_ids already in the CSV
def load_done_set(path, col="video_id"):
    if not Path(path).exists():
        return set()
    return set(pd.read_csv(path, usecols=[col])[col].unique())


def main():
    rotator = KeyRotator(API_KEYS)
    youtube = build_client(rotator.current())

    # Load input, keep only videos with comments confirmed available
    df = pd.read_csv(INPUT_CSV)
    df = df[df["comments_actually_available"] == True]
    all_ids = df["video_id"].tolist()

    # Skip anything already processed
    done = load_done_set(OUTPUT_CSV)
    failed = load_done_set(FAILED_CSV)
    remaining = [v for v in all_ids if v not in done and v not in failed]

    print(f"Total: {len(all_ids)} | Done: {len(done)} | Failed: {len(failed)} | Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    # Open outputs in append mode, write header only if fresh
    new_out = not Path(OUTPUT_CSV).exists()
    new_fail = not Path(FAILED_CSV).exists()
    out_f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    fail_f = open(FAILED_CSV, "a", newline="", encoding="utf-8")
    out_writer = csv.DictWriter(out_f, fieldnames=COMMENT_FIELDS)
    fail_writer = csv.DictWriter(fail_f, fieldnames=FAILED_FIELDS)
    if new_out:
        out_writer.writeheader()
    if new_fail:
        fail_writer.writeheader()

    # Ctrl-C: close cleanly
    def shutdown(*_):
        out_f.close(); fail_f.close()
        print("\nClean shutdown. Re-run to resume.")
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    processed = 0
    total_comments = 0

    for i, vid in enumerate(remaining, 1):
        status, payload, youtube = process_video(youtube, rotator, vid)

        if status == "ok":
            for r in payload:
                out_writer.writerow(r)
            out_f.flush()
            processed += 1
            total_comments += len(payload)
            if processed % PROGRESS_EVERY == 0:
                print(f"[{i}/{len(remaining)}] {vid} → {len(payload)} comments "
                      f"(total: {total_comments})")
        elif status == "fail":
            fail_writer.writerow({"video_id": vid, "reason": payload})
            fail_f.flush()
        elif status == "quota_dead":
            print("\nAll API keys exhausted. Save progress and resume after quota resets (midnight Pacific).")
            break

    out_f.close()
    fail_f.close()
    print(f"\nDone. Processed {processed} videos, wrote {total_comments} comments.")


if __name__ == "__main__":
    main()