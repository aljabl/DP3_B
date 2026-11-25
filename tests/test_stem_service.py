"""Checks for disk reuse, in-memory caching, and Demucs fallback."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from dp3.stem_service import STEM_NAMES, load_or_separate, stem_cache_key
from dp3.upload import inspect_audio, whole_clip_audio


SAMPLE_RATE = 8000


def make_upload(name: str = "song.wav"):
    output = BytesIO()
    sf.write(output, np.zeros(30 * SAMPLE_RATE, dtype="float32"), SAMPLE_RATE, format="WAV")
    upload = inspect_audio(name, output.getvalue())
    upload.clip_bytes = whole_clip_audio(upload)
    upload.clip_start_seconds = 0.0
    return upload


def write_stems(folder: Path, *, count: int = 6) -> None:
    folder.mkdir(parents=True)
    for name in STEM_NAMES[:count]:
        sf.write(folder / f"{name}.flac", np.zeros(30 * SAMPLE_RATE), SAMPLE_RATE)


class FakeSeparator:
    sample_rate = SAMPLE_RATE

    def __init__(self):
        self.calls = 0

    def separate_array(self, audio, sample_rate):
        self.calls += 1
        assert sample_rate == SAMPLE_RATE
        assert audio.shape == (1, 30 * SAMPLE_RATE)
        return {name: np.zeros_like(audio) for name in STEM_NAMES}, 0.0


class StemServiceTests(unittest.TestCase):
    def test_disk_stems_are_loaded_into_ram_without_model(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            source = data_root / "audio" / "previews" / upload.name
            source.parent.mkdir(parents=True)
            source.write_bytes(upload.source_bytes)
            write_stems(data_root / "stems" / "previews" / "song")

            cache = {}
            def no_model():
                self.fail("Demucs should not load for a verified disk cache hit")

            stems, origin = load_or_separate(upload, cache, no_model, data_root=data_root)
            self.assertEqual(origin, "disk")
            self.assertEqual(set(stems), set(STEM_NAMES))
            self.assertIs(cache[stem_cache_key(upload)], stems)

            cached, origin = load_or_separate(upload, cache, no_model, data_root=data_root)
            self.assertEqual(origin, "memory")
            self.assertIs(cached, stems)

    def test_same_name_with_different_audio_runs_model(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            source = data_root / "audio" / "previews" / upload.name
            source.parent.mkdir(parents=True)
            source.write_bytes(upload.source_bytes + b"different")
            write_stems(data_root / "stems" / "previews" / "song")

            separator = FakeSeparator()
            cache = {}
            stems, origin = load_or_separate(upload, cache, lambda: separator, data_root=data_root)
            self.assertEqual(origin, "Demucs")
            self.assertEqual(separator.calls, 1)
            self.assertEqual(len(stems), 6)
            self.assertTrue(all(sf.info(BytesIO(stem)).frames == 30 * SAMPLE_RATE for stem in stems.values()))

            load_or_separate(upload, cache, lambda: separator, data_root=data_root)
            self.assertEqual(separator.calls, 1)

    def test_incomplete_disk_cache_runs_model(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            write_stems(data_root / "stems" / "uploads" / stem_cache_key(upload), count=5)
            separator = FakeSeparator()
            _stems, origin = load_or_separate(upload, {}, lambda: separator, data_root=data_root)
            self.assertEqual(origin, "Demucs")
            self.assertEqual(separator.calls, 1)


if __name__ == "__main__":
    unittest.main()
