#!/usr/bin/env python#!/usr/bin/env python3
"""Download Harmonix Set audio from YouTube for tracks with alignment score >= threshold,
and write a manifest.csv describing every selected track (existing, new, missing or failed)."""

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import soundfile as sf
import yt_dlp

BASE = "https://raw.githubusercontent.com/urinieto/harmonixset/main/dataset/"
SCORES_URL = BASE + "youtube_alignment_scores.csv"
URLS_URL = BASE + "youtube_urls.csv"

MANIFEST_COLUMNS = [
    "file_id", "url", "score", "status", "path", "format",
    "duration_s", "sample_rate", "channels", "size_bytes", "error",
]


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


def inspect_audio(path: Path):
    """Read header info only (no decoding). Returns (info_dict, error_message)."""
    try:
        info = sf.info(str(path))
        return {
            "duration_s": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }, ""
    except Exception as e:
        return {}, f"unreadable: {str(e)[:200]}"


def build_manifest_row(row, out_dir: Path, fmt: str, download_errors: dict) -> dict:
    fid = row["File"]
    path = out_dir / f"{fid}.{fmt}"
    record = {
        "file_id": fid,
        "url": row["URL"],
        "score": row["score"],
        "status": "",
        "path": "",
        "format": fmt,
        "duration_s": "",
        "sample_rate": "",
        "channels": "",
        "size_bytes": "",
        "error": "",
    }

    if path.exists():
        info, err = inspect_audio(path)
        record["path"] = str(path.resolve())
        record["size_bytes"] = path.stat().st_size
        if err:
            record["status"] = "unreadable"
            record["error"] = err
        else:
            record["status"] = "ok"
            record.update(info)
    elif fid in download_errors:
        record["status"] = "failed"
        record["error"] = download_errors[fid]
    else:
        record["status"] = "missing"      # not downloaded yet / run interrupted
    return record


def write_manifest(tracks: pd.DataFrame, out_dir: Path, fmt: str, download_errors: dict):
    records = [build_manifest_row(row, out_dir, fmt, download_errors)
               for _, row in tracks.iterrows()]
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    counts = manifest["status"].value_counts().to_dict()
    print(f"\nManifest written to {manifest_path}  {counts}")


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
    p.add_argument("--manifest-only", action="store_true",
                   help="skip downloading; just (re)build manifest.csv from files on disk")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = build_track_list(args.threshold)
    if args.limit:
        tracks = tracks.head(args.limit)
    print(f"{len(tracks)} tracks with score >= {args.threshold}")

    tracks.to_csv(out_dir / "selected_tracks.csv", index=False)

    download_errors = {}   # file_id -> error message, for this run only

    try:
        if not args.manifest_only:
            for i, row in tracks.iterrows():
                fid, url = row["File"], row["URL"]
                if (out_dir / f"{fid}.{args.format}").exists():
                    print(f"[{i+1}/{len(tracks)}] skip {fid} (exists)")
                    continue
                print(f"[{i+1}/{len(tracks)}] {fid}  score={row['score']:.3f}")
                ok, err = download_one(fid, url, out_dir, args.format,
                                       args.cookies_browser, args.cookies_file)
                if not ok:
                    download_errors[fid] = err
                time.sleep(args.sleep)
    finally:
        # Runs on normal completion, errors, and Ctrl-C
        write_manifest(tracks, out_dir, args.format, download_errors)

    print("Done.")


if __name__ == "__main__":
    main()
