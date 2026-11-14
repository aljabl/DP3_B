"""Offline index build: download -> separate -> extract -> table, for one dataset.

    python build.py --dataset harmonix --stage all
    python build.py --dataset harmonix --stage extract --limit 3

Every stage skips tracks that are already cached, so an interrupted run can
simply be started again.
"""

import argparse
import time
import traceback
import numpy as np
import pandas as pd

from pathlib import Path

from datasets import harmonix_download, get_tracks
from dp3 import config
from dp3.cache import Cache, StemCache, FeatureCache, RunLog
from dp3.separation import StemSeparator
from dp3.features import FeatureExtractor, load_mono
from dp3.evaluate import ListeningTest

SEPARATION_COLUMNS = ["file_id", "status", "time_s", "residual", "error"]
EXTRACTION_COLUMNS = ["file_id", "status", "time_s", "tempo", "n_beats",
                      "n_stems_ok", "n_stems_silent", "beat_time_s",
                      "stems_time_s", "chroma_time_s", "mfcc_time_s", "error"]
DATASET_COLUMNS = ["dataset", "track_id", "stem", "status", "rms_db", "row"]

PROJECT_ROOT = Path(__file__).resolve().parent


def describe_error(e: Exception) -> str:
    """One-line description: exception type, message, and where it was raised.

    Locations read "folder/file.py:LINE in function". If the error was raised inside
    a library (e.g. demucs/apply.py), the innermost frame from this project that led
    there is added as "called from ...".
    """
    frames = traceback.extract_tb(e.__traceback__)

    def fmt(f):
        return f"{'/'.join(Path(f.filename).parts[-2:])}:{f.lineno} in {f.name}"

    where = fmt(frames[-1])                                  # innermost frame
    ours = [f for f in frames if Path(f.filename).resolve().is_relative_to(PROJECT_ROOT)]

    if ours and ours[-1] is not frames[-1]:
        where += f", called from {fmt(ours[-1])}"

    message = str(e).replace("\n", " ")[:200]
    return f"{type(e).__name__}: {message} [{where}]"

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
            error = describe_error(e)
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
        cache.save(track.file_id, stems, separator.sample_rate)
        return {"residual": round(residual, 4)}

    run_stage(tracks, cache, RunLog(cache.root / "separation_log.csv", SEPARATION_COLUMNS), process)

def stage_extract(tracks, dataset:str):
    stems_cache = StemCache(config.stems_dir(dataset))
    cache = FeatureCache(config.features_dir(dataset))
    extractor = FeatureExtractor()

    def process(track):
        if not stems_cache.has(track.file_id):
            raise FileNotFoundError("stems not found (run the separate stage first)")

        mix = load_mono(track.path)
        stems = {name: load_mono(p) for name, p in stems_cache.paths(track.file_id).items()}
        result = extractor.analyze(mix, stems)
        cache.save(track.file_id, result)
        statuses = [s["status"] for s in result["stems"].values()]

        return {"tempo": round(result["tempo"], 1), "n_beats": result["n_beats"],
                "n_stems_ok": statuses.count("ok"), "n_stems_silent": statuses.count("silent"),
                "beat_time_s": round(result["beat_time_s"], 2),
                "stems_time_s": round(result["stems_time_s"], 2),
                "chroma_time_s": round(result["chroma_time_s"], 2),
                "mfcc_time_s": round(result["mfcc_time_s"], 2)}

    run_stage(tracks, cache, RunLog(cache.root / "features_log.csv", EXTRACTION_COLUMNS), process)

def stage_table():
    """Collect the pooled vectors of every dataset under data/features into one table:
    data/index/embeddings_index.csv (dataset, track_id, stem, status, rms_db, row) and
    data/index/embeddings.npz (chroma, mfcc; `row` indexes into them, -1 for silent stems).
    Vectors are stored raw: standardizing / combining them is done when the search index is built."""

    rows, chroma, mfcc = [], [], []
    features_root = config.DATA / "features"

    if not features_root.exists():
        raise SystemExit("table: no features found (run the extract stage first)")

    for dataset_dir in sorted(p for p in features_root.iterdir() if p.is_dir()):
        cache = FeatureCache(dataset_dir)

        for track_id in cache.keys():
            try:
                result = cache.load(track_id)
            except (FileNotFoundError):
                print(f"load: file {track_id} not found")
                continue

            for stem, s in result["stems"].items():
                row = {"dataset": dataset_dir.name, "track_id": track_id, "stem": stem,
                       "status": s["status"], "rms_db": s["rms_db"], "row": -1}
                if s["status"] == "ok":

                    row["row"] = len(chroma)
                    chroma.append(s["chroma"])
                    mfcc.append(s["mfcc"])

                rows.append(row)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DATASET_COLUMNS).to_csv(config.INDEX_DIR / "embeddings_index.csv", index=False)
    np.savez_compressed(
        config.INDEX_DIR / "embeddings.npz",
        chroma=np.stack(chroma) if chroma else np.empty((0, config.CHROMA_DIM), np.float32),
        mfcc=np.stack(mfcc) if mfcc else np.empty((0, config.MFCC_DIM), np.float32))

    print(f"table: {len(rows)} stem rows ({len(chroma)} ok) -> {config.INDEX_DIR}")

def latest_ok(log_path: Path, columns: list) -> pd.DataFrame:
    """Successful rows of a run log, one per track (the latest if it was logged more than once)."""

    log = pd.read_csv(log_path)
    log = log[log["status"] == "ok"].drop_duplicates("file_id", keep="last")
    return log.set_index("file_id")[columns]


def stage_timing(dataset: str):
    """Seconds per song for separation, beat tracking, chroma and MFCC, plus each descriptor's cost.
    Beat tracking is shared by all descriptors, so it is not part of their cost."""

    separation = latest_ok(config.stems_dir(dataset) / "separation_log.csv", ["time_s"])
    features = latest_ok(config.features_dir(dataset) / "features_log.csv",
                         ["beat_time_s", "chroma_time_s", "mfcc_time_s"])
    songs = separation.rename(columns={"time_s": "separation_s"}).join(features, how="inner")
    songs["total_s"] = songs.sum(axis=1)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    songs.to_csv(config.INDEX_DIR / "timing_per_song.csv")

    print(songs.describe(percentiles=[0.5, 0.9]).round(2).T)
    chroma, mfcc = songs["chroma_time_s"].mean(), songs["mfcc_time_s"].mean()
    print(f"\nMean extraction cost per song: chroma {chroma:.2f}s, mfcc {mfcc:.2f}s, "
          f"combined {chroma + mfcc:.2f}s")


def stage_listen(dataset: str, score: bool):
    """Prepare the blind listening test (writes data/listening/<dataset>/), or score the ratings."""

    index = pd.read_csv(config.INDEX_DIR / "embeddings_index.csv", dtype={"track_id": str})
    index = index[(index["dataset"] == dataset) & (index["status"] == "ok")]

    with np.load(config.INDEX_DIR / "embeddings.npz") as data:
        vectors = {"chroma": data["chroma"], "mfcc": data["mfcc"]}

    test = ListeningTest(index, vectors, StemCache(config.stems_dir(dataset)),
                         config.DATA / "listening" / dataset)

    if score:
        print(test.score().round(2))
    else:
        test.prepare()
        print(f"listen: fill in the rating column of {test.out_dir / 'ratings.csv'}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["harmonix", "fma", "previews"], default="harmonix")
    p.add_argument("--stage",
                   choices=["download", "separate", "extract", "table", "timing", "listen", "all"],
                   default="all",
                   help="'all' = download (harmonix only) + separate + extract + table")
    p.add_argument("--limit", type=int, default=None, help="only first N tracks (for testing)")
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
        tracks = get_tracks(args.dataset, args.limit)
        print(f"{len(tracks)} tracks")

        if "separate" in stages:
            stage_separate(tracks, args.dataset)

        if "extract" in stages:
            stage_extract(tracks, args.dataset)

    if "table" in stages:
        stage_table()

    if "timing" in stages:
        stage_timing(args.dataset)

    if "listen" in stages or "score" in stages:
        stage_listen(args.dataset, score="score" in stages)

    print("Done.")


if __name__ == "__main__":
    main()
