"""Validate and crop uploaded audio without writing it to disk."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import soundfile as sf


CLIP_SECONDS = 30
CLIP_TOLERANCE_SECONDS = 0.1  # Account for encoder padding in 30-second previews.
MIME_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".caf": "audio/x-caf",
}
ALAC_SUFFIXES = frozenset({".m4a", ".mp4", ".caf"})


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
    playback_bytes: bytes | None = None

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def _audio_buffer(data: bytes, name: str) -> BytesIO:
    buffer = BytesIO(data)
    buffer.name = name
    return buffer


def inspect_audio(name: str, data: bytes) -> UploadedAudio:
    """Read metadata from an in-memory WAV, MP3, or ALAC upload."""
    suffix = Path(name).suffix.lower()
    if suffix not in MIME_TYPES:
        raise AudioUploadError("Choose a WAV, MP3, or ALAC file (.m4a, .mp4, or .caf).")
    if not data:
        raise AudioUploadError("The selected file is empty.")

    if suffix in ALAC_SUFFIXES:
        try:
            with av.open(BytesIO(data)) as container:
                if not container.streams.audio:
                    raise AudioUploadError("This file contains no audio stream.")
                stream = container.streams.audio[0]
                if stream.codec_context.name != "alac":
                    raise AudioUploadError("This file does not contain ALAC audio.")
                sample_rate = stream.rate
                if not sample_rate or not stream.codec_context.layout.channels:
                    raise AudioUploadError("The selected file contains no playable audio.")
                if stream.duration is not None:
                    seconds = float(stream.duration * stream.time_base)
                    frames = round(seconds * sample_rate)
                elif container.duration is not None:
                    frames = round(container.duration / av.time_base * sample_rate)
                else:
                    frames = sum(frame.samples for frame in container.decode(stream))
                if frames <= 0:
                    raise AudioUploadError("The selected file contains no playable audio.")
                return UploadedAudio(name, data, sample_rate, frames, MIME_TYPES[suffix])
        except AudioUploadError:
            raise
        except (av.FFmpegError, OSError, ValueError) as exc:
            raise AudioUploadError("This ALAC file could not be decoded as audio.") from exc

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


def _decode_alac(upload: UploadedAudio, start_frame: int = 0,
                 frame_count: int | None = None) -> bytes:
    """Decode only the requested frames into an in-memory PCM WAV."""
    output = BytesIO()
    written = 0
    skipped = 0
    try:
        with av.open(BytesIO(upload.source_bytes)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout,
                                           rate=upload.sample_rate)
            channels = len(stream.codec_context.layout.channels)
            with sf.SoundFile(output, mode="w", samplerate=upload.sample_rate,
                              channels=channels, format="WAV", subtype="PCM_16") as wav:
                for frame in container.decode(stream):
                    for converted in resampler.resample(frame):
                        samples = converted.to_ndarray().T
                        if skipped < start_frame:
                            drop = min(start_frame - skipped, len(samples))
                            samples = samples[drop:]
                            skipped += drop
                        if frame_count is not None:
                            samples = samples[:frame_count - written]
                        if len(samples):
                            wav.write(np.asarray(samples, dtype=np.float32))
                            written += len(samples)
                    if frame_count is not None and written == frame_count:
                        break
        if frame_count is not None and written != frame_count:
            raise AudioUploadError("The full 30-second crop could not be decoded.")
        if (frame_count is None and
                abs(written - upload.frames) / upload.sample_rate > CLIP_TOLERANCE_SECONDS):
            raise AudioUploadError("The ALAC audio ended before its reported duration.")
        if written == 0:
            raise AudioUploadError("This ALAC file contains no playable audio.")
        return output.getvalue()
    except (av.FFmpegError, sf.LibsndfileError, OSError, ValueError) as exc:
        if isinstance(exc, AudioUploadError):
            raise
        raise AudioUploadError("This ALAC file could not be decoded as audio.") from exc


def source_audio_for_playback(upload: UploadedAudio) -> tuple[bytes, str]:
    """Return browser-playable source audio, decoding ALAC once per session."""
    if Path(upload.name).suffix.lower() not in ALAC_SUFFIXES:
        return upload.source_bytes, upload.mime_type
    if upload.playback_bytes is None:
        upload.playback_bytes = _decode_alac(upload)
    return upload.playback_bytes, "audio/wav"


def crop_audio(upload: UploadedAudio, start_seconds: float) -> bytes:
    """Return exactly 30 seconds of PCM WAV audio as bytes."""
    clip_frames = CLIP_SECONDS * upload.sample_rate
    max_start_frame = upload.frames - clip_frames
    if max_start_frame < 0:
        raise AudioUploadError("The audio is shorter than 30 seconds.")
    if not 0 <= start_seconds <= max_start_frame / upload.sample_rate:
        raise AudioUploadError("The crop must fit within the uploaded audio.")

    start_frame = min(round(start_seconds * upload.sample_rate), max_start_frame)
    if Path(upload.name).suffix.lower() in ALAC_SUFFIXES:
        return _decode_alac(upload, start_frame, clip_frames)
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


def whole_clip_audio(upload: UploadedAudio) -> bytes:
    """Convert an already 30-second upload to playable WAV bytes."""
    if abs(upload.duration_seconds - CLIP_SECONDS) > CLIP_TOLERANCE_SECONDS:
        raise AudioUploadError("This file needs a 30-second crop.")
    if Path(upload.name).suffix.lower() in ALAC_SUFFIXES:
        return _decode_alac(upload)
    try:
        with sf.SoundFile(_audio_buffer(upload.source_bytes, upload.name)) as audio:
            samples = audio.read(dtype="float32", always_2d=True)
        output = BytesIO()
        sf.write(output, samples, upload.sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()
    except (sf.LibsndfileError, OSError) as exc:
        raise AudioUploadError("This file could not be decoded as audio.") from exc


def bytes_match_file(path: Path, data: bytes) -> bool:
    """Check byte identity without loading an entire disk file into memory."""
    try:
        if not path.is_file() or path.stat().st_size != len(data):
            return False
        digest = sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.digest() == sha256(data).digest()
    except OSError:
        return False


def source_matches_file(path: Path, upload: UploadedAudio) -> bool:
    """Check whether a disk source is the uploaded file."""
    return bytes_match_file(path, upload.source_bytes)
