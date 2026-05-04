"""
Extract transcripts for YouTube videos.

Output: transcripts.csv (one row per video)
Schema is designed for downstream SBERT title-transcript similarity computation.

Resume-safe: skips video_ids already in transcripts.csv or transcripts_failed.csv.
Rate-limited: random sleep between requests to reduce IP-ban risk.
Auto-pauses on consecutive failures (likely IP block).

Pinned: youtube-transcript-api==0.6.3 (matches Group 09's existing pipeline).
"""

import csv
import os
import sys
import time
import random
import signal
from pathlib import Path

import pandas as pd

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# ============ CONFIG ============
INPUT_CSV = "filtered_data.csv"
OUTPUT_CSV = "transcripts.csv"
FAILED_CSV = "transcripts_failed.csv"

# Sleep between requests (random in this range) - youtube-transcript-api
# scrapes the timedtext endpoint and IP-bans aggressive callers.
SLEEP_MIN = 1.0
SLEEP_MAX = 2.5

# Pause-and-retry behaviour after streak of unexpected errors.
CONSECUTIVE_FAIL_THRESHOLD = 8
LONG_SLEEP = 600     # 10 min
PROGRESS_EVERY = 1
# ================================

TRANSCRIPT_FIELDS = [
    "video_id", "language", "is_generated",
    "segment_count", "duration_seconds", "transcript_text",
]
FAILED_FIELDS = ["video_id", "reason"]


def fetch_transcript(video_id):
    """
    Returns dict for transcripts.csv, or None if video has no transcripts.
    Raises on disabled/unavailable/network errors.

    Selection policy (multilingual-friendly):
      1) If a manually-created transcript exists in any language → take it.
      2) Else, take an auto-generated transcript in any language.
      3) Do NOT translate. We want the ORIGINAL language so XLM-R / multilingual
         SBERT can do their job on real content.
    """
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    manual, generated = [], []
    for t in transcript_list:
        (generated if t.is_generated else manual).append(t)

    chosen = manual[0] if manual else (generated[0] if generated else None)
    if chosen is None:
        return None

    segments = chosen.fetch()

    parts = [s["text"].replace("\n", " ").strip() for s in segments]
    text = " ".join(p for p in parts if p)
    text = " ".join(text.split())  # collapse whitespace

    duration = sum(s.get("duration", 0.0) for s in segments)

    return {
        "video_id": video_id,
        "language": chosen.language_code,
        "is_generated": chosen.is_generated,
        "segment_count": len(segments),
        "duration_seconds": round(duration, 2),
        "transcript_text": text,
    }


def load_done_set(path, col="video_id"):
    if not Path(path).exists():
        return set()
    return set(pd.read_csv(path, usecols=[col])[col].unique())


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["transcript_actually_available"] == True]
    all_ids = df["video_id"].tolist()

    done = load_done_set(OUTPUT_CSV)
    failed = load_done_set(FAILED_CSV)
    remaining = [v for v in all_ids if v not in done and v not in failed]

    print(f"Total: {len(all_ids)} | Done: {len(done)} | Failed: {len(failed)} | Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    new_out = not Path(OUTPUT_CSV).exists()
    new_fail = not Path(FAILED_CSV).exists()
    out_f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    fail_f = open(FAILED_CSV, "a", newline="", encoding="utf-8")
    out_writer = csv.DictWriter(out_f, fieldnames=TRANSCRIPT_FIELDS)
    fail_writer = csv.DictWriter(fail_f, fieldnames=FAILED_FIELDS)
    if new_out:
        out_writer.writeheader()
    if new_fail:
        fail_writer.writeheader()

    def shutdown(*_):
        out_f.close(); fail_f.close()
        print("\nClean shutdown. Re-run to resume.")
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    processed = 0
    consec_fails = 0

    for i, vid in enumerate(remaining, 1):
        try:
            result = fetch_transcript(vid)
            if result is None:
                fail_writer.writerow({"video_id": vid, "reason": "no_transcript_available"})
                fail_f.flush()
            else:
                out_writer.writerow(result)
                out_f.flush()
                processed += 1
                consec_fails = 0
                if processed % PROGRESS_EVERY == 0:
                    print(f"[{i}/{len(remaining)}] {vid} → "
                          f"lang={result['language']}, gen={result['is_generated']}, "
                          f"{result['segment_count']} segs, {result['duration_seconds']:.0f}s")

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            fail_writer.writerow({"video_id": vid, "reason": type(e).__name__})
            fail_f.flush()
        except VideoUnavailable:
            fail_writer.writerow({"video_id": vid, "reason": "VideoUnavailable"})
            fail_f.flush()
        except CouldNotRetrieveTranscript as e:
            fail_writer.writerow({"video_id": vid, "reason": f"CouldNotRetrieve: {str(e)[:120]}"})
            fail_f.flush()
        except Exception as e:
            consec_fails += 1
            msg = str(e)[:200]
            fail_writer.writerow({"video_id": vid, "reason": f"{type(e).__name__}: {msg}"})
            fail_f.flush()

            if consec_fails >= CONSECUTIVE_FAIL_THRESHOLD:
                print(f"\n⚠ {consec_fails} consecutive failures - probably IP-throttled. "
                      f"Sleeping {LONG_SLEEP}s and resuming...")
                time.sleep(LONG_SLEEP)
                consec_fails = 0

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    out_f.close()
    fail_f.close()
    print(f"\nDone. Processed {processed} new transcripts.")


if __name__ == "__main__":
    main()
