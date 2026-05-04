"""
Double-checks comment + transcript availability for a list of YouTube videos.

Fixes over v1:
  - Distinguishes rate-limit / IP-block errors from genuine "unavailable" for transcripts.
  - Exponential backoff on transient rate-limits; graceful stop if sustained.
  - Row-by-row checkpointing: writes results as it goes, resumes from existing output.
  - Handles malformed input CSVs (double-quoted rows).
  - Ctrl-C safe (progress is already on disk).
  - Separate reason/status columns so you can debug which rows failed and why.
"""

import csv
import os
import signal
import sys
import time

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

# --- import transcript exceptions defensively (library has moved things around across versions)
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
try:
    from youtube_transcript_api._errors import TooManyRequests
except ImportError:
    class TooManyRequests(Exception): pass
try:
    from youtube_transcript_api._errors import YouTubeRequestFailed
except ImportError:
    class YouTubeRequestFailed(Exception): pass
try:
    from youtube_transcript_api._errors import IpBlocked
except ImportError:
    class IpBlocked(Exception): pass
try:
    from youtube_transcript_api._errors import NoTranscriptAvailable
except ImportError:
    class NoTranscriptAvailable(Exception): pass

# --- CONFIGURATION ---
API_KEYS = [
    "AIzaSyCqJMOEtA5alkIbyXRqkp6tX8n4ZTZnQ9c",
    "AIzaSyBS08Jm7ksgU91zNlO_Q2T1drMGVP6frVw",
    "AIzaSyCK2hSLFqFQYugMbkNcIwxx09-LZjxeXrc",
    "AIzaSyBU3cnQ1uXMaq9-gXyCqI9EdJL8rbbeE7Q",
]

INPUT_CSV  = "vierti_dataset.csv"
OUTPUT_CSV = "availability_verified.csv"

# Rate-limit behaviour for the transcript endpoint
BACKOFF_SCHEDULE_SEC = [15, 45, 120]   # three retries on a transient 429
SLEEP_BETWEEN_VIDEOS = 0.25            # be gentle on the endpoint

# The fieldnames we always write to OUTPUT_CSV
OUT_FIELDS = [
    "video_id",
    "comment_count", "comments_status", "view_count", "like_count", "error",
    "comments_actually_available", "comments_reason",
    "transcript_actually_available", "transcript_reason",
]


# ---------------------------------------------------------------------------
# Input loading — tolerate the double-quoted-row malformed CSV
# ---------------------------------------------------------------------------
def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If the whole row ended up as one column (happens when every line is wrapped in "...")
    if len(df.columns) == 1 and "," in df.columns[0]:
        print(f"[!] Detected malformed CSV (single quoted column). Re-parsing.")
        header = df.columns[0].split(",")
        rows = [row[0].split(",") for row in df.itertuples(index=False)]
        df = pd.DataFrame(rows, columns=header)
        # normalise numeric cols we know about
        for c in ("comment_count", "view_count", "like_count"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def already_done(path: str) -> set:
    """Return set of video_ids already present in the output CSV."""
    if not os.path.exists(path):
        return set()
    try:
        done = pd.read_csv(path, usecols=["video_id"])
        return set(done["video_id"].astype(str).tolist())
    except Exception as e:
        print(f"[!] Could not read existing output for resume ({e}); starting fresh.")
        return set()


def open_output_writer(path: str):
    """Open output CSV in append mode, writing header only if file is new/empty."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


# ---------------------------------------------------------------------------
# Comment check (API-key based, quota rotation)
# ---------------------------------------------------------------------------
class QuotaExhausted(Exception):
    pass


class YouTubeClientPool:
    def __init__(self, keys):
        self.keys = keys
        self.idx = 0
        self.client = build("youtube", "v3", developerKey=self.keys[self.idx])

    def rotate(self):
        self.idx += 1
        if self.idx >= len(self.keys):
            raise QuotaExhausted("All API keys exhausted.")
        print(f"    [!] Rotating to API key #{self.idx + 1}")
        self.client = build("youtube", "v3", developerKey=self.keys[self.idx])


def check_comments(pool: YouTubeClientPool, video_id: str):
    """Returns (has_comments: bool, reason: str). Raises QuotaExhausted if we're fully out."""
    while True:
        try:
            pool.client.commentThreads().list(
                part="id", videoId=video_id, maxResults=1
            ).execute()
            return True, "ok"
        except HttpError as e:
            msg = str(e)
            status = getattr(getattr(e, "resp", None), "status", None)
            if status == 403 and "quotaExceeded" in msg:
                pool.rotate()          # try again with the next key
                continue
            if status == 403 and "commentsDisabled" in msg:
                return False, "disabled"
            if status == 404:
                return False, "video_not_found"
            # Anything else (e.g. 400 invalid id): treat as unavailable but log the reason.
            return False, f"http_{status}"
        except Exception as e:
            return False, f"error:{type(e).__name__}"


# ---------------------------------------------------------------------------
# Transcript check — the part that was buggy
# ---------------------------------------------------------------------------
class TranscriptRateLimited(Exception):
    """Signals we should stop the whole run — we're being blocked."""


def check_transcript(video_id: str):
    """
    Returns (has_transcript: bool, reason: str).
    Raises TranscriptRateLimited if, after backoff retries, we still look blocked.
    """
    attempt = 0
    while True:
        try:
            YouTubeTranscriptApi.list_transcripts(video_id)
            return True, "ok"

        # ---- Genuine "no transcript" cases ----
        except TranscriptsDisabled:
            return False, "disabled"
        except (NoTranscriptFound, NoTranscriptAvailable):
            return False, "none"
        except VideoUnavailable:
            return False, "video_unavailable"

        # ---- Rate-limit / block cases — back off and retry ----
        except (TooManyRequests, IpBlocked, YouTubeRequestFailed) as e:
            if attempt >= len(BACKOFF_SCHEDULE_SEC):
                raise TranscriptRateLimited(
                    f"{type(e).__name__} on {video_id} after {attempt} retries"
                )
            wait = BACKOFF_SCHEDULE_SEC[attempt]
            print(f"    [!] {type(e).__name__} — backing off {wait}s (retry {attempt + 1}/{len(BACKOFF_SCHEDULE_SEC)})")
            time.sleep(wait)
            attempt += 1
            continue

        # ---- Unknown errors: inspect the message for 429-ish signals before giving up ----
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "too many requests" in msg
                    or "blocked" in msg or "ip" in msg and "block" in msg):
                if attempt >= len(BACKOFF_SCHEDULE_SEC):
                    raise TranscriptRateLimited(f"{type(e).__name__}: {e}")
                wait = BACKOFF_SCHEDULE_SEC[attempt]
                print(f"    [!] Suspected rate-limit ({type(e).__name__}) — backing off {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            # Actually unknown — log the class and move on, but don't call it "no transcript" blindly.
            return False, f"error:{type(e).__name__}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run():
    df = load_input(INPUT_CSV)
    total = len(df)
    done = already_done(OUTPUT_CSV)
    print(f"Loaded {total} videos. {len(done)} already in {OUTPUT_CSV}; {total - len(done)} to go.")

    pool = YouTubeClientPool(API_KEYS)
    out_f, writer = open_output_writer(OUTPUT_CSV)

    # Ctrl-C: just make sure we flush and exit cleanly. The file has everything so far.
    def _sigint(signum, frame):
        print("\n[!] Ctrl-C received — flushing and exiting. Re-run to resume.")
        out_f.flush(); out_f.close()
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    processed_this_run = 0
    try:
        for index, row in df.iterrows():
            video_id = str(row["video_id"])
            if video_id in done:
                continue

            # -- Comments --
            try:
                has_comments, comments_reason = check_comments(pool, video_id)
            except QuotaExhausted:
                print("\n[!] All API keys exhausted. Saving progress and exiting.")
                print("    Resume by re-running the script tomorrow.")
                break

            # -- Transcript --
            try:
                has_transcript, transcript_reason = check_transcript(video_id)
            except TranscriptRateLimited as e:
                print(f"\n[!] Transcript endpoint is blocking us ({e}).")
                print("    Saving progress and exiting. Try again later or from a different IP.")
                break

            out_row = {
                **{k: row[k] for k in row.index if k in OUT_FIELDS},
                "video_id": video_id,
                "comments_actually_available": has_comments,
                "comments_reason": comments_reason,
                "transcript_actually_available": has_transcript,
                "transcript_reason": transcript_reason,
            }
            writer.writerow(out_row)
            out_f.flush()               # ensure durability row-by-row
            processed_this_run += 1

            print(f"[{index + 1}/{total}] {video_id} | "
                  f"comments={has_comments}({comments_reason}) | "
                  f"transcript={has_transcript}({transcript_reason})")

            time.sleep(SLEEP_BETWEEN_VIDEOS)

    finally:
        out_f.flush(); out_f.close()
        print(f"\nDone for this run. Processed {processed_this_run} videos.")
        print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()