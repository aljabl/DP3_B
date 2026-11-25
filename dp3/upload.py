"""Validate and crop uploaded audio without writing it to disk."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import soundfile as sf


CLIP_SECONDS = 30
MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


class AudioUploadError(ValueError):
    """An uploaded file cannot be used as a 30-second audio clip."""


@dataclass
class UploadedAudio:
    name: str
    source_bytes: bytes
    sample_rate: int
    frames: int
    mime_type: str
    clip_bytes: bytes | None = None
    clip_start_seconds: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def _audio_buffer(data: bytes, name: str) -> BytesIO:
    buffer = BytesIO(data)
    buffer.name = name
    return buffer


def inspect_audio(name: str, data: bytes) -> UploadedAudio:
    """Read metadata from an in-memory WAV or MP3 upload."""
    suffix = Path(name).suffix.lower()
    if suffix not in MIME_TYPES:
        raise AudioUploadError("Choose a WAV or MP3 file.")
    if not data:
        raise AudioUploadError("The selected file is empty.")

    try:
        with sf.SoundFile(_audio_buffer(data, name)) as audio:
            if audio.samplerate <= 0 or audio.frames <= 0 or audio.channels <= 0:
                raise AudioUploadError("The selected file contains no playable audio.")
            return UploadedAudio(
                name=name,
                source_bytes=data,
                sample_rate=audio.samplerate,
                frames=audio.frames,
                mime_type=MIME_TYPES[suffix],
            )
    except (sf.LibsndfileError, OSError) as exc:
        raise AudioUploadError("This file could not be decoded as audio.") from exc


def crop_audio(upload: UploadedAudio, start_seconds: float) -> bytes:
    """Return exactly 30 seconds of PCM WAV audio as bytes."""
    clip_frames = CLIP_SECONDS * upload.sample_rate
    max_start_frame = upload.frames - clip_frames
    if max_start_frame < 0:
        raise AudioUploadError("The audio is shorter than 30 seconds.")
    if not 0 <= start_seconds <= max_start_frame / upload.sample_rate:
        raise AudioUploadError("The crop must fit within the uploaded audio.")

    start_frame = min(round(start_seconds * upload.sample_rate), max_start_frame)
    try:
        with sf.SoundFile(_audio_buffer(upload.source_bytes, upload.name)) as audio:
            audio.seek(start_frame)
            samples = audio.read(clip_frames, dtype="float32", always_2d=True)
        if len(samples) != clip_frames:
            raise AudioUploadError("The full 30-second crop could not be decoded.")

        output = BytesIO()
        sf.write(output, samples, upload.sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()
    except (sf.LibsndfileError, OSError) as exc:
        raise AudioUploadError("The crop could not be decoded as audio.") from exc
