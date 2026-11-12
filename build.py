"""Offline index build: download -> separate -> extract -> table, for one dataset.
 
    python build.py --dataset harmonix --stage all
    python build.py --dataset harmonix --stage extract --limit 3
 
Every stage skips tracks that are already cached, so an interrupted run can
simply be started again.
"""

import argparse
import time
import numpy as np
import pandas as pd

from datasets import harmonix_download, get_tracks
from dp3 import config
from dp3.cache import Cache, StemCache, RunLog
from dp3.separation import StemSeparator

SEPARATION_COLUMNS = ["file_id", "status", "time_s", "residual", "error"]


def run_stage(tracks, cache: Cache, log: RunLog, process):
    """Skip cached tracks; for the rest call process(track) -> dict of log fields.
    process is responsible for saving its result to the cache (saves are atomic).
    A track that raises is logged as failed and left uncached, so it is retried next run."""

    for i, track in enumerate(tracks, 1):
        if cache.has(track.file_id):
            # Track has been processed, skip
            print(f"[{i}/{len(tracks)}] skip {track.file_id} (cached)")
            continue

        print(f"[{i}/{len(tracks)}] {track.file_id}")
        start = time.perf_counter()

        try:
            fields = process(track)

        except Exception as e:
            # Log and skip failed track
            error = str(e).replace("\n", " ")[:200]
            print(f"  FAILED {track.file_id}: {error}")
            log.add(file_id=track.file_id, status="failed", error=error)
            continue

        log.add(file_id=track.file_id, status="ok",
                time_s=round(time.perf_counter() - start, 2), **fields)

def stage_separate(tracks, dataset: str):
    # Get existing stems
    cache = StemCache(config.stems_dir(dataset))

    if all(cache.has(t.file_id) for t in tracks):
        print("separate: everything cached")
        return

    separator = StemSeparator()
 
    def process(track):
        stems, residual = separator.separate(track.path)
        cache.save(track.file_id, {name: a.numpy() for name, a in stems.items()},
                   separator.sample_rate)
        return {"residual": round(residual, 4)}
 
    run_stage(tracks, cache, RunLog(cache.root / "separation_log.csv", SEPARATION_COLUMNS), process)

def stage_extract(tracks, dataset:str):
    None

def stage_table():
    None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["harmonix", "fma"], default="harmonix")
    p.add_argument("--stage", choices=["download", "separate", "extract", "table", "all"],
                   default="all", help="'all' = download (harmonix only) + separate + extract + table")
    p.add_argument("--limit", type=int, default=None, help="only first N tracks (for testing)")
    p.add_argument("--fma-root", default="fma/", help="folder containing fma_metadata/ and fma_small/")
    p.add_argument("--threshold", type=float, default=0.95, help="harmonix alignment score cutoff")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds between harmonix downloads")
    p.add_argument("--cookies-browser", default="firefox", help="e.g. chrome, firefox, safari, edge") # required for youtube downloads
    p.add_argument("--cookies-file", default=None, help="Netscape-format cookies.txt")
    args = p.parse_args()
 
    stages = ["download", "separate", "extract", "table"] if args.stage == "all" else [args.stage]
 
    if "download" in stages and args.dataset == "harmonix":
        harmonix_download(args.threshold, args.sleep, args.limit,
                          args.cookies_browser, args.cookies_file)
 
    if "separate" in stages or "extract" in stages:
        tracks = get_tracks(args.dataset, args.fma_root, args.limit)
        print(f"{len(tracks)} tracks")

        if "separate" in stages:
            stage_separate(tracks, args.dataset)

        if "extract" in stages:
            stage_extract(tracks, args.dataset)
 
    if "table" in stages:
        stage_table()

    print("Done.")
 
 
if __name__ == "__main__":
    main()
