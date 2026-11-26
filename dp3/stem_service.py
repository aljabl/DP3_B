"""Load cached stems or separate a session clip with Demucs."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Callable

import numpy as np
import soundfile as sf

from . import config
from .upload import CLIP_SECONDS, CLIP_TOLERANCE_SECONDS, UploadedAudio, source_matches_file


STEM_NAMES = config.SEPARATION_MODELS[config.SEPARATION_MODEL]
DATA_ROOT = Path(__file__).resolve().parents[1] / config.DATA


def stem_names(model_name: str) -> tuple[str, ...]:
    """Return the source names for a supported Demucs model."""
    try:
        return config.SEPARATION_MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unsupported Demucs model: {model_name}") from None


def stem_cache_key(upload: UploadedAudio, model_name: str = config.SEPARATION_MODEL) -> str:
    """Identify the selected clip and model, regardless of its source filename."""
    if upload.clip_bytes is None:
        raise ValueError("Save a 30-second clip before separating it.")
    digest = sha256()
    stem_names(model_name)
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(upload.clip_bytes)
    return digest.hexdigest()


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    output = BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _read_disk_stems(folder: Path, expected_seconds: float,
                     model_name: str) -> dict[str, bytes] | None:
    paths = {name: folder / f"{name}.flac" for name in stem_names(model_name)}
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


def _existing_dataset_stems(upload: UploadedAudio, data_root: Path,
                            model_name: str) -> dict[str, bytes] | None:
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

    # Offline dataset stems were built with the configured six-source model.
    if model_name == config.SEPARATION_MODEL:
        for source, folder in candidates:
            if source_matches_file(source, upload):
                cached = _read_disk_stems(folder, upload.duration_seconds, model_name)
                if cached is not None:
                    return cached
    user_features = data_root / "features" / "user_added"
    for metadata in user_features.glob("*/meta.json"):
        try:
            saved = json.loads(metadata.read_text())
            filename = saved["song_filename"]
            if Path(filename).name != filename:
                continue
            if saved.get("model_name", config.SEPARATION_MODEL) != model_name:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        source = data_root / "audio" / "user_added" / filename
        if source_matches_file(source, upload):
            folder = data_root / "stems" / "user_added" / metadata.parent.name
            cached = _read_disk_stems(folder, upload.duration_seconds, model_name)
            if cached is not None:
                return cached
    return None


def _cached_model_stems(
    upload: UploadedAudio,
    ram_cache: dict[str, dict[str, bytes]],
    data_root: Path,
    model_name: str,
) -> tuple[dict[str, bytes] | None, str | None]:
    """Find an existing result for one model without running Demucs."""
    key = stem_cache_key(upload, model_name)
    if key in ram_cache:
        return ram_cache[key], "memory"

    stems = _read_disk_stems(data_root / "stems" / "uploads" / key,
                             upload.duration_seconds, model_name)
    if stems is None:
        stems = _existing_dataset_stems(upload, data_root, model_name)
    if stems is not None:
        ram_cache[key] = stems
        return stems, "disk"
    return None, None


def _save_four_stem_cache(folder: Path, separated: dict, sample_rate: int) -> None:
    """Publish a complete four-stem upload cache for reuse across sessions."""
    if folder.is_dir():
        expected_seconds = len(next(iter(separated.values()))[0]) / sample_rate
        if _read_disk_stems(folder, expected_seconds, "htdemucs") is not None:
            return
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    try:
        for name, samples in separated.items():
            sf.write(temporary / f"{name}.flac", np.asarray(samples).T,
                     sample_rate, subtype="PCM_16")
        if folder.is_dir():
            shutil.rmtree(folder)
        try:
            temporary.rename(folder)
        except OSError:
            if not folder.is_dir():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_model_stems(
    upload: UploadedAudio,
    ram_cache: dict[str, dict[str, bytes]],
    separator_factory: Callable,
    data_root: Path,
    model_name: str,
) -> tuple[dict[str, bytes], str]:
    """Load or run one model."""
    stems, origin = _cached_model_stems(upload, ram_cache, data_root, model_name)
    if stems is not None:
        return stems, origin

    names = stem_names(model_name)
    key = stem_cache_key(upload, model_name)
    clip, sample_rate = sf.read(BytesIO(upload.clip_bytes), dtype="float32", always_2d=True)
    separator = separator_factory(model_name)
    separated, _residual = separator.separate_array(clip.T, sample_rate)
    if set(separated) != set(names):
        raise RuntimeError(f"Demucs did not return the expected {len(names)} stems.")
    stems = {
        name: _wav_bytes(np.asarray(separated[name]).T, separator.sample_rate)
        for name in names
    }
    if model_name == "htdemucs":
        _save_four_stem_cache(data_root / "stems" / "uploads" / key,
                              separated, separator.sample_rate)
    ram_cache[key] = stems
    return stems, "Demucs"


def load_or_separate(
    upload: UploadedAudio,
    ram_cache: dict[str, dict[str, bytes]],
    separator_factory: Callable,
    *,
    data_root: Path = DATA_ROOT,
    model_name: str = config.SEPARATION_MODEL,
) -> tuple[dict[str, bytes], str]:
    """Return playable stems, preferring four-source estimates for shared types.

    A six-stem request runs only htdemucs_6s unless four-stem estimates are
    already cached for this exact clip. Four-stem uploads are cached on disk
    so later sessions can prefer them for shared stem types.
    """
    stem_names(model_name)
    if model_name == "htdemucs":
        four, origin = _load_model_stems(upload, ram_cache, separator_factory,
                                         data_root, model_name)
        six_key = stem_cache_key(upload, "htdemucs_6s")
        if six_key in ram_cache:
            ram_cache[six_key].update(four)
        return four, origin

    six, six_origin = _load_model_stems(upload, ram_cache, separator_factory,
                                        data_root, "htdemucs_6s")
    four, _four_origin = _cached_model_stems(upload, ram_cache, data_root,
                                             "htdemucs")
    if four is None:
        return six, six_origin
    combined = {**six, **four}
    ram_cache[stem_cache_key(upload, model_name)] = combined
    return combined, six_origin
