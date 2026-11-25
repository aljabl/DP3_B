"""Load cached stems or separate a session clip with Demucs."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import config
from .upload import CLIP_SECONDS, CLIP_TOLERANCE_SECONDS, UploadedAudio, source_matches_file


STEM_NAMES = ("vocals", "drums", "bass", "guitar", "piano", "other")
DATA_ROOT = Path(__file__).resolve().parents[1] / config.DATA


def stem_cache_key(upload: UploadedAudio) -> str:
    """Identify the selected clip and model, regardless of its source filename."""
    if upload.clip_bytes is None:
        raise ValueError("Save a 30-second clip before separating it.")
    digest = sha256()
    digest.update(config.SEPARATION_MODEL.encode("utf-8"))
    digest.update(b"\0")
    digest.update(upload.clip_bytes)
    return digest.hexdigest()


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    output = BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _read_disk_stems(folder: Path, expected_seconds: float) -> dict[str, bytes] | None:
    paths = {name: folder / f"{name}.flac" for name in STEM_NAMES}
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        stems = {}
        stem_shape = None
        for name, path in paths.items():
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            if abs(len(samples) / sample_rate - expected_seconds) > CLIP_TOLERANCE_SECONDS:
                return None
            if stem_shape is not None and (len(samples), sample_rate) != stem_shape:
                return None
            stem_shape = (len(samples), sample_rate)
            stems[name] = _wav_bytes(samples, sample_rate)
        return stems
    except (sf.LibsndfileError, OSError, ValueError):
        return None


def _existing_dataset_stems(upload: UploadedAudio, data_root: Path) -> dict[str, bytes] | None:
    # Dataset stems describe the whole source file. Reuse them only when the
    # selected clip is the entire, byte-identical dataset source.
    if upload.clip_start_seconds != 0 or abs(upload.duration_seconds - CLIP_SECONDS) > CLIP_TOLERANCE_SECONDS:
        return None

    filename = Path(upload.name).name
    stem = Path(filename).stem
    candidates = []
    for dataset in ("previews", "harmonix"):
        source = data_root / "audio" / dataset / filename
        folder = data_root / "stems" / dataset / stem.replace(" ", "")
        candidates.append((source, folder))

    if stem.isdigit() and len(stem) == 6:
        source = data_root / "audio" / "fma" / "fma_small" / stem[:3] / filename
        for folder in (data_root / "stems" / "fma").glob(f"{stem}_*"):
            candidates.append((source, folder))

    for source, folder in candidates:
        if source_matches_file(source, upload):
            cached = _read_disk_stems(folder, upload.duration_seconds)
            if cached is not None:
                return cached
    return None


def load_or_separate(
    upload: UploadedAudio,
    ram_cache: dict[str, dict[str, bytes]],
    separator_factory: Callable,
    *,
    data_root: Path = DATA_ROOT,
) -> tuple[dict[str, bytes], str]:
    """Return six playable stems, always retaining them in the session cache.

    The disk cache is read only. New separation results stay in RAM.
    """
    key = stem_cache_key(upload)
    if key in ram_cache:
        return ram_cache[key], "memory"

    stems = _read_disk_stems(data_root / "stems" / "uploads" / key, upload.duration_seconds)
    if stems is None:
        stems = _existing_dataset_stems(upload, data_root)
    if stems is not None:
        ram_cache[key] = stems
        return stems, "disk"

    clip, sample_rate = sf.read(BytesIO(upload.clip_bytes), dtype="float32", always_2d=True)
    separator = separator_factory()
    separated, _residual = separator.separate_array(clip.T, sample_rate)
    if set(separated) != set(STEM_NAMES):
        raise RuntimeError("Demucs did not return all six expected stems.")
    stems = {
        name: _wav_bytes(np.asarray(separated[name]).T, separator.sample_rate)
        for name in STEM_NAMES
    }
    ram_cache[key] = stems
    return stems, "Demucs"
