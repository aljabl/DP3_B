"""
Atomic folder cache and CSV run log, shared by every stage.
"""

import csv
import json
import os
import shutil
import soundfile as sf
import numpy as np

from .facets import FacetPolicy

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

    def keys(self) -> list[str]:
        """Find complete feature entries, including IDs containing a slash."""
        return sorted(
            folder.parent.relative_to(self.root).as_posix()
            for folder in self.root.rglob("meta.json")
            if (folder.parent / "features.npz").is_file()
            and not any(part.endswith(".tmp") for part in folder.relative_to(self.root).parts)
        )

    def missing_facets(self, key: str) -> dict[str, tuple[str, ...]]:
        """Return facets absent from an existing track cache, per active stem."""
        meta = json.loads((self.path(key) / "meta.json").read_text())
        with np.load(self.path(key) / "features.npz", allow_pickle=False) as data:
            names = set(data.files)
        policy = FacetPolicy()
        return {
            stem: missing
            for stem, info in meta["stems"].items()
            if info["status"] == "ok"
            if (missing := tuple(f for f in policy.facets_for(stem)
                                 if f"{stem}__{f}" not in names))
        }

    def add_facets(self, key: str, facets: dict[str, dict[str, np.ndarray]],
                   metadata: dict | None = None) -> None:
        """Atomically add computed facets without replacing legacy descriptors."""
        folder = self.path(key)
        with np.load(folder / "features.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        for stem, values in facets.items():
            for facet, vector in values.items():
                arrays[f"{stem}__{facet}"] = np.asarray(vector, dtype=np.float32)
        temporary = folder / "features.npz.tmp"
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(output, **arrays)
            os.replace(temporary, folder / "features.npz")
        finally:
            temporary.unlink(missing_ok=True)
        if metadata:
            meta_path = folder / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(metadata)
            meta_tmp = folder / "meta.json.tmp"
            try:
                meta_tmp.write_text(json.dumps(meta, indent=2))
                os.replace(meta_tmp, meta_path)
            finally:
                meta_tmp.unlink(missing_ok=True)

    def save(self, key: str, result: dict):
        arrays = {"beat_times": np.asarray(result["beat_times"], dtype=np.float32)}
        stems_meta = {}

        for name, s in result["stems"].items():
            stems_meta[name] = {"status": s["status"], "rms_db": round(s["rms_db"], 2)}

            if s["status"] == "ok":
                for k in (*self._ARRAYS, *FacetPolicy().facets_for(name)):
                    arrays[f"{name}__{k}"] = s[k]

        meta = {"tempo": result["tempo"], "n_beats": result["n_beats"],
                "meter_beats": result.get("meter_beats"),
                "meter_confidence": result.get("meter_confidence", 0.0),
                "stems": stems_meta}

        with self.write(key) as folder:
            np.savez_compressed(folder / "features.npz", **arrays)
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, key: str) -> dict:
        folder = self.path(key)
        meta = json.loads((folder / "meta.json").read_text())
        stems = {}
        with np.load(folder / "features.npz", allow_pickle=False) as data:
            for name, info in meta["stems"].items():
                stems[name] = dict(info)
                if info["status"] == "ok":
                    for k in (*self._ARRAYS, *FacetPolicy().facets_for(name)):
                        if f"{name}__{k}" in data:
                            stems[name][k] = data[f"{name}__{k}"]
            beat_times = data["beat_times"]

        return {"tempo": meta["tempo"], "n_beats": meta["n_beats"],
                "meter_beats": meta.get("meter_beats"),
                "meter_confidence": meta.get("meter_confidence", 0.0),
                "beat_times": beat_times, "stems": stems}

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
