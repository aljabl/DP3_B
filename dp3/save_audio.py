"""Persist a selected clip, its stems, and searchable descriptors."""

import json
import shutil
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import soundfile as sf

from . import config
from .facets import FacetPolicy
from .features import FeatureExtractor, to_analysis
from .stem_service import DATA_ROOT, STEM_NAMES, load_or_separate
from .upload import (
    CLIP_SECONDS,
    CLIP_TOLERANCE_SECONDS,
    ALAC_SUFFIXES,
    AudioUploadError,
    UploadedAudio,
    crop_audio,
    inspect_audio,
    whole_clip_audio,
)


USER_ADDED_DIR = Path(__file__).resolve().parents[1] / config.DATA / "user_added"
INVALID_TITLE_CHARS = frozenset('<>:"/\\|?*')


def _saved_audio(
    upload: UploadedAudio,
    title: str,
    directory: Path,
) -> tuple[Path, bytes]:
    """Validate the title and return the exact clip that will be saved."""
    name = title.strip()
    if (
        not name
        or name in {".", ".."}
        or name.endswith(".")
        or any(char in INVALID_TITLE_CHARS or ord(char) < 32 for char in name)
    ):
        raise AudioUploadError("Enter a title without path separators or special filename characters.")

    if upload.clip_bytes is None:
        raise AudioUploadError("Save a 30-second crop before saving this song to disk.")
    if (
        upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS
        and upload.clip_start_seconds is None
    ):
        raise AudioUploadError("Save a 30-second crop before saving this song to disk.")

    if upload.duration_seconds > CLIP_SECONDS:
        data = (
            upload.clip_bytes
            if upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS
            else crop_audio(upload, 0.0)
        )
        suffix = ".wav"
    elif Path(upload.name).suffix.lower() in ALAC_SUFFIXES:
        data = upload.clip_bytes
        suffix = ".wav"
    else:
        data = upload.source_bytes
        suffix = Path(upload.name).suffix.lower()

    return directory / f"{name}{suffix}", data


def save_uploaded_audio(
    upload: UploadedAudio,
    title: str,
    *,
    directory: Path = USER_ADDED_DIR,
) -> Path:
    """Write the selected clip without overwriting an existing song."""
    path, data = _saved_audio(upload, title, directory)
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(data)
    return path


def _analysis_audio(data: bytes) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.T, sample_rate


def save_and_index_upload(
    upload: UploadedAudio,
    title: str,
    ram_cache: dict[str, dict[str, bytes]],
    separator_factory,
    *,
    data_root: Path = DATA_ROOT,
    extractor_factory=FeatureExtractor,
) -> Path:
    """Separate and analyze a clip, then publish its audio and search entry.

    The index entry is committed last. If any step fails, remove the new song
    and stems so the user can retry the save without a partial result.
    """
    song_dir = data_root / "user_added"
    song_path, song_bytes = _saved_audio(upload, title, song_dir)
    if song_path.exists():
        raise FileExistsError(song_path)

    saved_upload = inspect_audio(song_path.name, song_bytes)
    saved_upload.clip_bytes = whole_clip_audio(saved_upload)
    saved_upload.clip_start_seconds = 0.0
    stems, _origin = load_or_separate(
        saved_upload, ram_cache, separator_factory, data_root=data_root
    )
    if set(stems) != set(STEM_NAMES):
        raise ValueError("The song must have all six stems before it can be saved.")

    mix, mix_rate = _analysis_audio(song_bytes)
    stem_audio = {}
    for name in STEM_NAMES:
        audio, rate = _analysis_audio(stems[name])
        stem_audio[name] = to_analysis(audio, rate)
    features = extractor_factory().analyze(to_analysis(mix, mix_rate), stem_audio)
    policy = FacetPolicy()
    facets = policy.indexed_facets()
    rows = []
    vectors = {facet: [] for facet in facets}
    for name in STEM_NAMES:
        result = features["stems"][name]
        row = {"stem": name, "status": result["status"],
               "rms_db": float(result["rms_db"]), "row": -1}
        if result["status"] == "ok":
            row["row"] = len(vectors[facets[0]])
            applicable = policy.facets_for(name)
            for facet in facets:
                vector = (result[facet] if facet in applicable
                          else np.zeros({config.TIMBRE: config.TIMBRE_DIM,
                                         config.RHYTHM: config.RHYTHM_DIM}[facet]))
                vectors[facet].append(np.asarray(vector, dtype=np.float32))
        rows.append(row)
    if not any(row["status"] == "ok" for row in rows):
        raise ValueError("All six stems are too quiet to add to the search index.")

    track_id = uuid4().hex
    stem_dir = data_root / "stems" / "user_added" / track_id
    index_dir = data_root / "index" / "user_added"
    entry_dir = index_dir / track_id
    created_song = False
    try:
        song_path = save_uploaded_audio(upload, title, directory=song_dir)
        created_song = True
        stem_dir.mkdir(parents=True)
        for name in STEM_NAMES:
            audio, rate = _analysis_audio(stems[name])
            sf.write(stem_dir / f"{name}.flac", audio.T, rate, subtype="PCM_16")
        index_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{track_id}-", dir=index_dir) as temp:
            folder = Path(temp)
            (folder / "meta.json").write_text(json.dumps({
                "song_filename": song_path.name,
                "rows": rows,
                "meter_beats": features.get("meter_beats"),
                "meter_confidence": features.get("meter_confidence", 0.0),
            }))
            np.savez_compressed(folder / "vectors.npz", **{
                facet: np.stack(values) for facet, values in vectors.items()
            })
            folder.rename(entry_dir)
    except Exception:
        if not entry_dir.exists():
            if created_song:
                song_path.unlink(missing_ok=True)
            shutil.rmtree(stem_dir, ignore_errors=True)
        raise
    return song_path
