"""
Extract transcripts for YouTube videos. v2.

Differences from v1:
- youtube-transcript-api >= 1.0.0 (new instance-based API)
- 429 / empty-body responses are now correctly classified as IP blocks
  and trigger an ABORT, not a 10-min sleep that does nothing
- Block-failures go to a separate `blocked.csv` file so they can be
  retried later from a different IP, instead of being marked as
  "no transcript available" forever
- More conservative sleep band (4-8s)
- Resume-safe across network changes

Resume strategy:
- transcripts.csv      → permanent: successfully fetched
- transcripts_failed.csv → permanent: video genuinely has no transcript
                          / disabled / unavailable
- transcripts_blocked.csv → transient: IP was blocked. These are NOT
                          treated as done — re-run from a new IP and
                          they'll be picked up again.
"""

import csv
import sys
import time
import random
import signal
from pathlib import Path

import pandas as pd

import http.cookiejar
import requests

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
    YouTubeRequestFailed,
    CouldNotRetrieveTranscript,
)

# ============ CONFIG ============
INPUT_CSV = "filtered_data.csv"
OUTPUT_CSV = "transcripts.csv"
FAILED_CSV = "transcripts_failed.csv"
BLOCKED_CSV = "transcripts_blocked.csv"

# Conservative request rate. Don't be greedy on a residential IP.
SLEEP_MIN = 4
SLEEP_MAX = 8

# Once your IP is flagged, sleeping won't unflag it. Abort and resume
# later from a different network.
ABORT_AFTER_CONSECUTIVE_BLOCKS = 5

PROGRESS_EVERY = 1
# ================================

TRANSCRIPT_FIELDS = [
    "video_id", "language", "is_generated",
    "segment_count", "duration_seconds", "transcript_text",
]
FAILED_FIELDS = ["video_id", "reason"]
BLOCKED_FIELDS = ["video_id", "reason"]


def _build_session(cookie_path: str) -> requests.Session:
    jar = http.cookiejar.MozillaCookieJar(cookie_path)
    jar.load(ignore_discard=True, ignore_expires=True)
    s = requests.Session()
    s.cookies = jar
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    })
    return s

ytt_api = YouTubeTranscriptApi(http_client=_build_session("cookies.txt"))

def fetch_transcript(video_id):
    """
    Returns dict for transcripts.csv, or None if the video genuinely has
    no transcripts (manual or auto). Raises on disabled/unavailable/blocked.

    Selection policy (multilingual-friendly, matches v1):
      1) Manually-created transcript in any language → take it.
      2) Else, auto-generated transcript in any language.
      3) Do NOT translate — keep original language for SBERT/XLM-R.
    """
    transcript_list = ytt_api.list(video_id)

    manual, generated = [], []
    for t in transcript_list:
        (generated if t.is_generated else manual).append(t)

    chosen = manual[0] if manual else (generated[0] if generated else None)
    if chosen is None:
        return None

    fetched = chosen.fetch()  # FetchedTranscript object in 1.x

    # 1.x: fetched.snippets is a list of FetchedTranscriptSnippet
    # with .text, .start, .duration attributes (not dict keys).
    parts = [s.text.replace("\n", " ").strip() for s in fetched.snippets]
    text = " ".join(p for p in parts if p)
    text = " ".join(text.split())

    duration = sum(s.duration for s in fetched.snippets)

    return {
        "video_id": video_id,
        "language": chosen.language_code,
        "is_generated": chosen.is_generated,
        "segment_count": len(fetched.snippets),
        "duration_seconds": round(duration, 2),
        "transcript_text": text,
    }


def load_done_set(path, col="video_id"):
    if not Path(path).exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=[col])[col].unique())
    except (pd.errors.EmptyDataError, ValueError):
        return set()


def is_block_error(exc):
    """
    Classify whether an exception means 'IP blocked, retry later from a
    different network' vs 'video genuinely lacks a transcript'.
    """
    if isinstance(exc, (RequestBlocked, IpBlocked, YouTubeRequestFailed)):
        return True
    # Defensive: in case the library wraps a 429 inside CouldNotRetrieve
    # without raising RequestBlocked, sniff the message.
    msg = str(exc).lower()
    return ("429" in msg
            or "too many requests" in msg
            or "ip" in msg and "block" in msg)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["transcript_actually_available"] == True]
    all_ids = df["video_id"].tolist()

    done = load_done_set(OUTPUT_CSV)
    failed = load_done_set(FAILED_CSV)
    # Note: blocked are intentionally NOT skipped — we want to retry them.
    remaining = [v for v in all_ids if v not in done and v not in failed]

    print(f"Total: {len(all_ids)} | Done: {len(done)} | "
          f"Failed-permanent: {len(failed)} | Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    new_out = not Path(OUTPUT_CSV).exists()
    new_fail = not Path(FAILED_CSV).exists()
    new_block = not Path(BLOCKED_CSV).exists()

    out_f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    fail_f = open(FAILED_CSV, "a", newline="", encoding="utf-8")
    block_f = open(BLOCKED_CSV, "a", newline="", encoding="utf-8")
    out_writer = csv.DictWriter(out_f, fieldnames=TRANSCRIPT_FIELDS)
    fail_writer = csv.DictWriter(fail_f, fieldnames=FAILED_FIELDS)
    block_writer = csv.DictWriter(block_f, fieldnames=BLOCKED_FIELDS)
    if new_out:
        out_writer.writeheader()
    if new_fail:
        fail_writer.writeheader()
    if new_block:
        block_writer.writeheader()

    def shutdown(*_):
        out_f.close(); fail_f.close(); block_f.close()
        print("\nClean shutdown. Re-run to resume.")
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    processed = 0
    consec_blocks = 0

    for i, vid in enumerate(remaining, 1):
        try:
            result = fetch_transcript(vid)
            if result is None:
                # Genuine: video has no transcripts at all.
                fail_writer.writerow({"video_id": vid,
                                      "reason": "no_transcript_available"})
                fail_f.flush()
            else:
                out_writer.writerow(result)
                out_f.flush()
                processed += 1
                consec_blocks = 0
                if processed % PROGRESS_EVERY == 0:
                    print(f"[{i}/{len(remaining)}] {vid} → "
                          f"lang={result['language']}, "
                          f"gen={result['is_generated']}, "
                          f"{result['segment_count']} segs, "
                          f"{result['duration_seconds']:.0f}s")

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            fail_writer.writerow({"video_id": vid, "reason": type(e).__name__})
            fail_f.flush()
            consec_blocks = 0

        except VideoUnavailable:
            fail_writer.writerow({"video_id": vid, "reason": "VideoUnavailable"})
            fail_f.flush()
            consec_blocks = 0

        except Exception as e:
            # Includes RequestBlocked, IpBlocked, YouTubeRequestFailed,
            # CouldNotRetrieveTranscript-with-429, and anything weird.
            blocked = is_block_error(e)
            reason = f"{type(e).__name__}: {str(e)[:160]}"

            if blocked:
                # Transient: write to blocked.csv so it gets retried later.
                block_writer.writerow({"video_id": vid, "reason": reason})
                block_f.flush()
                consec_blocks += 1
                print(f"[{i}/{len(remaining)}] {vid} BLOCKED "
                      f"(consec={consec_blocks}/{ABORT_AFTER_CONSECUTIVE_BLOCKS})")

                if consec_blocks >= ABORT_AFTER_CONSECUTIVE_BLOCKS:
                    print(f"\n⚠ {consec_blocks} consecutive blocks. "
                          f"This IP is flagged. Aborting.\n"
                          f"Resume from a different network "
                          f"(phone hotspot, campus wifi, teammate's machine).")
                    out_f.close(); fail_f.close(); block_f.close()
                    sys.exit(2)
            else:
                # Genuinely weird unknown error → log to failed and continue.
                fail_writer.writerow({"video_id": vid, "reason": reason})
                fail_f.flush()
                consec_blocks = 0

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    out_f.close()
    fail_f.close()
    block_f.close()
    print(f"\nDone. Processed {processed} new transcripts in this run.")


if __name__ == "__main__":
    main()