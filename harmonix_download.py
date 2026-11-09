#!/usr/bin/env python3
"""Download Harmonix Set audio from YouTube for tracks with alignment score >= threshold."""

import argparse
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yt_dlp

BASE = "https://raw.githubusercontent.com/urinieto/harmonixset/main/dataset/"
SCORES_URL = BASE + "youtube_alignment_scores.csv"
URLS_URL = BASE + "youtube_urls.csv"


def fetch_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as r:
        return pd.read_csv(io.StringIO(r.read().decode("utf-8")))


def build_track_list(threshold: float) -> pd.DataFrame:
    scores = fetch_csv(SCORES_URL)
    urls = fetch_csv(URLS_URL)
    scores["File"] = scores["File"].astype(str).str.strip()
    urls["File"] = urls["File"].astype(str).str.strip()
    df = urls.merge(scores, on="File", how="inner")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"] >= threshold].reset_index(drop=True)
    return df


def download_one(file_id, url, out_dir, fmt, cookies_browser=None, cookies_file=None):
    """Returns (success: bool, error_message: str)."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{file_id}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": fmt,          # "wav" or "mp3"
        }],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return True, ""
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"  FAILED {file_id}: {msg[:200]}", file=sys.stderr)
        return False, msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="harmonix_audio", help="output directory")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--format", choices=["wav", "mp3"], default="wav")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds between downloads")
    p.add_argument("--limit", type=int, default=None, help="only try first N tracks (for testing)")
    p.add_argument("--cookies-browser", default=None,
                   help="browser to read cookies from, e.g. chrome, firefox, safari, edge")
    p.add_argument("--cookies-file", default=None,
                   help="path to a Netscape-format cookies.txt")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = build_track_list(args.threshold)
    if args.limit:
        tracks = tracks.head(args.limit)
    print(f"{len(tracks)} tracks with score >= {args.threshold}")

    tracks.to_csv(out_dir / "selected_tracks.csv", index=False)

    failed = []
    for i, row in tracks.iterrows():
        fid, url = row["File"], row["URL"]
        if (out_dir / f"{fid}.{args.format}").exists():
            print(f"[{i+1}/{len(tracks)}] skip {fid} (exists)")
            continue
        print(f"[{i+1}/{len(tracks)}] {fid}  score={row['score']:.3f}")
        ok, err = download_one(fid, url, out_dir, args.format,
                               args.cookies_browser, args.cookies_file)
        if not ok:
            failed.append((fid, url, err))
        time.sleep(args.sleep)

    if failed:
        with open(out_dir / "failed.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["File", "URL", "error"])
            w.writerows(failed)
        print(f"\n{len(failed)} failures logged to {out_dir / 'failed.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
