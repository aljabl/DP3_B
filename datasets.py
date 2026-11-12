"""
Dataset adapters. Every dataset exposes the same thing: a list of Track tuples
(file_id, path to the original audio, optional label such as genre).
Also holds the Harmonix YouTube downloader, since it is dataset-specific I/O.
"""

import io
import sys
import time
import urllib.request
from collections import namedtuple
from pathlib import Path

import pandas as pd

from dp3 import config
from dp3.cache import RunLog

Track = namedtuple("Track", ["file_id", "path", "label"])

HARMONIX_BASE = "https://raw.githubusercontent.com/urinieto/harmonixset/main/dataset/"
DOWNLOAD_LOG_COLUMNS = ["file_id", "url", "score", "status", "error"]


def get_tracks(dataset: str, fma_root: str = None, limit: int = None) -> list:
    if dataset == "harmonix":
        tracks = harmonix_tracks()

    elif dataset == "fma":
        if fma_root is None:
            raise SystemExit("--fma-root is required for the fma dataset")
        tracks = fma_tracks(Path(fma_root))

    else:
        raise SystemExit(f"unknown dataset: {dataset}")

    return tracks[:limit] if limit else tracks


# --- Harmonix ---
def harmonix_tracks() -> list:
    """Tracks whose audio has been downloaded (data/audio/harmonix/<file_id>.wav)."""

    return [Track(p.stem, p, None) for p in sorted(config.audio_dir("harmonix").glob("*.wav"))]


def _fetch_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as r:
        return pd.read_csv(io.StringIO(r.read().decode("utf-8")))


def harmonix_selection(threshold: float) -> pd.DataFrame:
    """Harmonix tracks with YouTube alignment score >= threshold (columns File, URL, score)."""

    scores = _fetch_csv(HARMONIX_BASE + "youtube_alignment_scores.csv")
    urls = _fetch_csv(HARMONIX_BASE + "youtube_urls.csv")

    scores["File"] = scores["File"].astype(str).str.strip()
    urls["File"] = urls["File"].astype(str).str.strip()

    df = urls.merge(scores, on="File", how="inner")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df[df["score"] >= threshold].reset_index(drop=True)


def harmonix_download(threshold: float = 0.95, sleep: float = 5.0, limit: int = None,
                      cookies_browser: str = None, cookies_file: str = None):
    """Download audio as wav for every selected track not already on disk.
    Results (including failures) are appended to download_log.csv in the audio folder."""

    import yt_dlp   # imported here so the other stages don't need it

    out_dir = config.audio_dir("harmonix")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "download_log.csv", DOWNLOAD_LOG_COLUMNS)

    selection = harmonix_selection(threshold)
    if limit:
        selection = selection.head(limit)
    print(f"{len(selection)} tracks with score >= {threshold}")

    opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True, "quiet": True, "no_warnings": True, "retries": 3,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        opts["cookiefile"] = cookies_file

    for i, row in selection.iterrows():
        fid, url = row["File"], row["URL"]

        if (out_dir / f"{fid}.wav").exists():
            print(f"[{i + 1}/{len(selection)}] skip {fid} (exists)")
            continue

        print(f"[{i + 1}/{len(selection)}] {fid}  score={row['score']:.3f}")

        try:
            with yt_dlp.YoutubeDL({**opts, "outtmpl": str(out_dir / f"{fid}.%(ext)s")}) as ydl:
                ydl.download([url])
            log.add(file_id=fid, url=url, score=row["score"], status="ok")

        except Exception as e:
            msg = str(e).replace("\n", " ")
            print(f"  FAILED {fid}: {msg[:200]}", file=sys.stderr)
            log.add(file_id=fid, url=url, score=row["score"], status="failed", error=msg[:200])
 
        time.sleep(sleep)

# --- FMA ---

def fma_tracks(fma_root: Path) -> list:
    return []
