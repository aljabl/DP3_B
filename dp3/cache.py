"""
Atomic folder cache and CSV run log, shared by every stage.
"""

import csv
import json
import shutil
import soundfile as sf
import numpy as np

from contextlib import contextmanager
from pathlib import Path


class Cache:
    """Stores one folder per key: <root>/<key>/.

    A folder exists only if it was fully written: writes go to <key>.tmp and
    are renamed on success, so an interrupted run never leaves a half-written
    entry behind (the leftover .tmp is cleaned up on the next write).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return self.path(key).is_dir()

    def keys(self) -> list:
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and not p.name.endswith(".tmp"))

    @contextmanager
    def write(self, key: str):
        """Yields an empty temporary folder to fill; commits it when the block ends cleanly."""

        tmp = self.root / f"{key}.tmp"
        shutil.rmtree(tmp, ignore_errors=True) # remove leftovers from an interrupted run
        tmp.mkdir(parents=True)
        yield tmp                       # an exception here skips the rename below
        tmp.rename(self.path(key))

class StemCache(Cache):
    """Stems of each track as <root>/<file_id>/<stem>.flac (16-bit)."""
 
    def save(self, key: str, stems: dict, sample_rate: int):
        with self.write(key) as folder:
            for name, audio in stems.items():
                # sf expects shape (frames, channels), but audio is of shape (channels, samples)
                sf.write(folder / f"{name}.flac", np.asarray(audio).T, sample_rate, subtype="PCM_16")

    def paths(self, key: str) -> dict:
        """Maps stem name -> file path for every stem stored for this track."""
        return {p.stem: p for p in sorted(self.path(key).glob("*.flac"))}

class FeatureCache(Cache):
    """Features of each track as <root>/<file_id>/{features.npz, meta.json}.

    save() takes the dict returned by features.analyze; load() returns the same
    structure (without the timing fields).
    """

    _ARRAYS = ("chroma_beat", "mfcc_beat", "chroma", "mfcc")

    def save(self, key: str, result: dict):
        arrays = {"beat_times": np.asarray(result["beat_times"], dtype=np.float32)}
        stems_meta = {}

        for name, s in result["stems"].items():
            stems_meta[name] = {"status": s["status"], "rms_db": round(s["rms_db"], 2)}

            if s["status"] == "ok":
                for k in self._ARRAYS:
                    arrays[f"{name}__{k}"] = s[k]

        meta = {"tempo": result["tempo"], "n_beats": result["n_beats"], "stems": stems_meta}

        with self.write(key) as folder:
            np.savez_compressed(folder / "features.npz", **arrays)
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, key: str) -> dict:
        folder = self.path(key)
        meta = json.loads((folder).read_text())
        data = np.load(folder)
        stems = {}

        for name, info in meta["stems"].items():
            stems[name] = dict(info)

            if info["status"] == "ok":
                for k in self._ARRAYS:
                    stems[name][k] = data[f"{name}__{k}"]

        return {"tempo": meta["tempo"], "n_beats": meta["n_beats"],
                "beat_times": data["beat_times"], "stems": stems}

class RunLog:
    """Appends one CSV row per processed item. Columns are fixed at construction."""

    def __init__(self, path: Path, columns: list):
        self.path = Path(path)
        self.columns = columns

    def add(self, **row):
        is_new = not self.path.exists()

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, self.columns, restval="")

            if is_new:
                writer.writeheader()

            writer.writerow(row)
