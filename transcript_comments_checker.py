import csv
from operator import index
import os
import signal
import sys
import time

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
# Older lib versions don't export these, stub them out
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

api_key = [
    "AIzaSyCqJMOEtA5alkIbyXRqkp6tX8n4ZTZnQ9c",
    "AIzaSyBS08Jm7ksgU91zNlO_Q2T1drMGVP6frVw",
    "AIzaSyCK2hSLFqFQYugMbkNcIwxx09-LZjxeXrc",
    "AIzaSyBU3cnQ1uXMaq9-gXyCqI9EdJL8rbbeE7Q",
]

inputFile = "vierti_dataset.csv"
outputFile = "availability_verified.csv"

#ratelimit
backoff_time = [15, 45, 120]
delay_inbetween = 1 #increase if ratelimited

target = 20000

#shuffling input to randomize vierti's data
shuffle_seed = 42

out_fields =  [
    "video_id", "channel_id", "channel_name", "video_title",
    "video_views", "video_likes", "video_comments", "label",
    "comments_actually_available", "comments_reason",
    "transcript_actually_available", "transcript_reason",
]

# Load, clean, and shuffle the input CSV
def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # If the whole row collapsed into one column, re-split it
    if len(df.columns) == 1 and ',' in df.columns[0]:
        print("csv malformed, attempt to fix")
        header = df.columns[0].split(',')
        rows = [row[0].spli(',') for row in df.itertuples(index=False)]
        df = pd.DataFrame(rows, columns=header)

    df = df.dropna(subset=["video_id"]).copy() #drop if row doesnt have video_id
    df["video_id"] = df["video_id"].astype(str) #normalize to string
    df = df.drop_duplicates(subset=["video_id"]) #drop if duplicate video_id
    df = df.sample(frac=1.0, random_state = shuffle_seed).reset_index(drop=True)


    #normalise numeric cols if present
    for c in ("video_comments", "video_views", "video_likes", "comment_counts", "view_count", "like_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c],errors = "coerce")
    return df


#checkpoint implementation

# Read existing output to find done ids and qualifying count
def checkpoint(path: str) -> tuple:
    if not os.path.exists(path):
        return set(), 0
    try :
        done = pd.read_csv(path)
        ids  = set(done["video_id"].astype(str).tolist())
        # Count rows where both flags are True
        if "comments_actually_available" in done.columns and "transcript_actually_available" in done.columns:
            both_ok = (
                done["comments_actually_available"].astype(str).str.lower().eq("true") &
                done["transcript_actually_available"].astype(str).str.lower().eq("true")
            ).sum()
        else:
            both_ok = 0
        return ids, int(both_ok)
    except Exception as e:
        print(f"error loading checkpoint. Proceeding to startover")
        return set(), 0

# Open output in append mode, header only if fresh
def open_output_writer(path : str):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames = out_fields, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


#Comment and transcript checking

class QuotaExhaustedError(Exception):
    pass

# Rotates API keys when quota runs out
class YoutubeClientPool:
    def __init__(self, keys):
        self.keys = keys
        self.idx = 0
        self.client = build("youtube", "v3", developerKey=self.keys[self.idx])

    def rotate(self):
        self.idx += 1
        if self.idx >= len(self.keys):
            raise QuotaExhaustedError("API key quota has been fully used")
        print(f"Rotating to next API key: {self.keys[self.idx]}")
        self.client = build("youtube", "v3", developerKey=self.keys[self.idx])

# Probe one video for comment availability
def check_comments(pool : YoutubeClientPool, video_id: str) :
    while True :
        try:
            pool.client.commentThreads().list(
                part="id", videoId = video_id, maxResults=1
            ).execute()
            return True, "ok"
        except HttpError as e:
            msg = str(e)
            status = getattr(getattr(e, "resp", None), "status", None)
            # Quota gone, rotate and retry
            if status == 403 and "quotaExceeded" in msg:
                pool.rotate()
                continue
            if status == 403 and "commentsDisabled " in msg:
                return False, "disabled"
            if status == 404:
                return False, "video_not_found"
            return False, f"http_{status}"
        except Exception as e:
            return False, f"error:{type(e).__name__}"

class TranscriptRateLimited(Exception):
    pass

# Probe one video for EN transcript availability
def check_transcript(video_id: str):
    attempt = 0
    while True:
        try:
            YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return True, "ok"

        #if no transcript cases
        except TranscriptsDisabled:
            return False, "disabled"
        except (NoTranscriptFound, NoTranscriptAvailable):
            return False, "none"
        except VideoUnavailable:
            return False, "video_unavailable"

        # rate limit case
        #backoff and retry

        except (TooManyRequests, YouTubeRequestFailed, IpBlocked) as e:
            if attempt >= len(backoff_time):
                raise TranscriptRateLimited(
                    f"{type(e).__name__} on {video_id} after {attempt} retries"
                )

            wait = backoff_time[attempt]
            print(f"{type(e).__name__}, waiting {wait}s ({attempt + 1}/{len(backoff_time)})")
            time.sleep(wait)
            attempt += 1
            continue

        #unknown errors
        except Exception as e:
            # Sniff message for 429-ish hints
            msg = str(e).lower()
            if ("429" in msg or "too many requests" in msg
                    or "blocked" in msg or ("ip" in msg and "block" in msg)):
                if attempt >= len(backoff_time):
                    raise TranscriptRateLimited(f"{type(e).__name__}: {e}")
                wait = backoff_time[attempt]
                print(f"likely rate-limit ({type(e).__name__}), waiting {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            # Actually unknown — log the class and move on, but don't call it "no transcript" blindly.
            return False, f"error:{type(e).__name__}"

def run():
    df = load_input(inputFile)
    total = len(df)
    done, available_count = checkpoint(outputFile)
    print(f"Total videos: {total}, already done: {len(done)}, available in checkpoint: {available_count}")
    if target is not None and available_count >= target:
        print("Target achieved")
        return

    pool = YoutubeClientPool(api_key)
    out_f, writer = open_output_writer(outputFile)

    # Ctrl-C: flush and exit
    def _sigint(signum, frame):
        print("\nstopped, re-run to resume")
        out_f.flush(); out_f.close()
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    processed_this_run = 0
    new_available_this_run = 0

    try:
        if target is not None and available_count >= target:
            print("Target achieved")
            return


        for index, row in df.iterrows():
            video_id = str(row["video_id"])
            if video_id in done:
                continue

            try :
                has_comments, comments_reason = check_comments(pool, video_id)
            except QuotaExhaustedError:
                print ("All api keys exhausted, attempt again in 1x24 hrs")
                break

            try :
                has_transcript, transcript_reason = check_transcript(video_id)
            except TranscriptRateLimited as e:
                print(f"\nblocked ({e}), retry later")
                break

            # Merge input columns with new availability results
            out_row = {
                **{k: row[k] for k in row.index if k in out_fields},
                        "video_id": video_id,
                        "comments_actually_available": has_comments,
                        "comments_reason": comments_reason,
                        "transcript_actually_available": has_transcript,
                        "transcript_reason": transcript_reason,
            }

            writer.writerow(out_row)
            out_f.flush()               # ensure durability row-by-row
            processed_this_run += 1

            both_ok = has_comments and has_transcript
            if both_ok :
                available_count += 1
                new_available_this_run += 1

            print(f"{index + 1} - https://youtu.be/{video_id} "
                  f"comments={has_comments} transcript={has_transcript} "
                  f"{{{available_count}}}")

            if target is not None and available_count >= target:
                print("Target achieved")
                break

            time.sleep(delay_inbetween)

    finally:
        out_f.flush();
        out_f.close()

        print(f"\nprocessed {processed_this_run} ({new_available_this_run} avail), total {available_count}")
if __name__ == "__main__":
    run()