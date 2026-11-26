"""Validate and crop uploaded audio without writing it to disk."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

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
    artist: str | None = None
    title: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate

    @property
    def display_artist(self) -> str:
        return self.artist or "Unknown Artist"


def normalize_artist(value: str | None) -> str | None:
    """Keep a real artist name or represent missing metadata as None."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


_TRAILING_CREDIT = re.compile(r"^(?P<title>.+?)\s*\((?P<credit>[^()]*)\)\s*$")
_TITLE_SEPARATOR = re.compile(r"\s+[-–—]\s+")
_FEATURED = re.compile(r"^(?:feat\.?|ft\.?|featuring)\s+", re.IGNORECASE)
_NON_ARTIST_CREDIT = re.compile(
    r"^(?:live|remix|mix|edit|version|acoustic|instrumental|demo|radio|"
    r"remaster(?:ed)?|reprise|explicit|clean|mono|stereo|official|cover|"
    r"karaoke|sped up|slowed)(?:\b|\s|$)", re.IGNORECASE,
)
_PRODUCTION_SUFFIX = re.compile(r"\b(?:remix|mix|edit|version|remaster(?:ed)?)$", re.IGNORECASE)


def _same_name(left: str, right: str) -> bool:
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _artist_credit_matches(credit: str, artist: str) -> bool:
    credit = _FEATURED.sub("", credit).strip()
    names = re.split(r"\s*(?:,|&|\band\b|\bfeat\.?\b|\bft\.?\b)\s*",
                     artist, flags=re.IGNORECASE)
    return any(_same_name(credit, name) for name in (artist, *names) if name)


def normalize_title_artist(title: str, artist: str | None) -> tuple[str, str | None]:
    """Separate clear artist credits from a song title."""
    title = title.strip()
    artist = normalize_artist(artist)
    if artist:
        title = re.sub(
            r"\s*[\(\[]([^()\[\]]+)[\)\]]",
            lambda match: "" if _artist_credit_matches(match[1], artist) else match[0],
            title,
        ).strip()
    match = _TRAILING_CREDIT.fullmatch(title)
    if match:
        credit = match["credit"].strip()
        if (not artist and credit and not _FEATURED.match(credit)
              and not _NON_ARTIST_CREDIT.match(credit)
              and not _PRODUCTION_SUFFIX.search(credit)
              and not re.match(r"^\d{4}\b", credit)):
            title, artist = match["title"].strip(), credit

    parts = _TITLE_SEPARATOR.split(title, maxsplit=1)
    if len(parts) == 2 and all(part.strip() for part in parts):
        left, right = (part.strip() for part in parts)
        if artist and _artist_credit_matches(left, artist):
            title = right
        elif artist and _artist_credit_matches(right, artist):
            title = left
        elif not artist:
            title, artist = right, left
    return title, artist


def song_identity(filename: str, metadata_title: str | None = None,
                  metadata_artist: str | None = None) -> tuple[str, str | None]:
    """Prefer embedded title tags while also using filename artist credits."""
    tagged_title = normalize_artist(metadata_title)
    if tagged_title and not metadata_artist:
        parts = _TITLE_SEPARATOR.split(Path(filename).stem, maxsplit=1)
        if len(parts) == 2 and _same_name(parts[0], tagged_title) and parts[1].strip():
            return normalize_title_artist(tagged_title, parts[1])
    filename_title, artist = normalize_title_artist(Path(filename).stem, metadata_artist)
    title = tagged_title or filename_title
    return normalize_title_artist(title, artist)


def _song_tags_from_container(container) -> tuple[str | None, str | None]:
    title = artist = None
    for tags in (container.metadata, *(stream.metadata for stream in container.streams.audio)):
        for key, value in tags.items():
            key = key.casefold()
            if key in {"artist", "tpe1"} and artist is None:
                artist = normalize_artist(value)
            elif key in {"title", "tit2"} and title is None:
                title = normalize_artist(value)
    return title, artist


def song_tags_from_media(source: bytes | Path) -> tuple[str | None, str | None]:
    """Read title and artist tags without decoding the audio stream."""
    try:
        with av.open(BytesIO(source) if isinstance(source, bytes) else str(source)) as container:
            return _song_tags_from_container(container)
    except (av.FFmpegError, OSError, ValueError):
        return None, None


def artist_from_media(source: bytes | Path) -> str | None:
    """Read an artist tag without decoding the audio stream."""
    return song_tags_from_media(source)[1]


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
                tagged_title, tagged_artist = _song_tags_from_container(container)
                title, artist = song_identity(name, tagged_title, tagged_artist)
                return UploadedAudio(name, data, sample_rate, frames, MIME_TYPES[suffix],
                                     artist=artist, title=title)
        except AudioUploadError:
            raise
        except (av.FFmpegError, OSError, ValueError) as exc:
            raise AudioUploadError("This ALAC file could not be decoded as audio.") from exc

    try:
        with sf.SoundFile(_audio_buffer(data, name)) as audio:
            if audio.samplerate <= 0 or audio.frames <= 0 or audio.channels <= 0:
                raise AudioUploadError("The selected file contains no playable audio.")
            tagged_title, tagged_artist = song_tags_from_media(data)
            title, artist = song_identity(name, tagged_title, tagged_artist)
            return UploadedAudio(
                name=name,
                source_bytes=data,
                sample_rate=audio.samplerate,
                frames=audio.frames,
                mime_type=MIME_TYPES[suffix],
                artist=artist,
                title=title,
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
