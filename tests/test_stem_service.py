"""Checks for disk reuse, in-memory caching, and Demucs fallback."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from dp3.stem_service import STEM_NAMES, load_or_separate, stem_cache_key, stem_names
from dp3.upload import inspect_audio, whole_clip_audio


SAMPLE_RATE = 8000


def make_upload(name: str = "song.wav"):
    output = BytesIO()
    sf.write(output, np.zeros(30 * SAMPLE_RATE, dtype="float32"), SAMPLE_RATE, format="WAV")
    upload = inspect_audio(name, output.getvalue())
    upload.clip_bytes = whole_clip_audio(upload)
    upload.clip_start_seconds = 0.0
    return upload


def write_stems(folder: Path, *, count: int = 6, model_name: str = "htdemucs_6s") -> None:
    folder.mkdir(parents=True)
    for name in stem_names(model_name)[:count]:
        sf.write(folder / f"{name}.flac", np.zeros(30 * SAMPLE_RATE), SAMPLE_RATE)


class FakeSeparator:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_name: str = "htdemucs_6s", amplitude: float = 0.0):
        self.calls = 0
        self.names = stem_names(model_name)
        self.amplitude = amplitude

    def separate_array(self, audio, sample_rate):
        self.calls += 1
        assert sample_rate == SAMPLE_RATE
        assert audio.shape == (1, 30 * SAMPLE_RATE)
        return {name: np.full_like(audio, self.amplitude) for name in self.names}, 0.0


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
            def no_model(model_name):
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
            stems, origin = load_or_separate(upload, cache, lambda model: separator, data_root=data_root)
            self.assertEqual(origin, "Demucs")
            self.assertEqual(separator.calls, 1)
            self.assertEqual(len(stems), 6)
            self.assertTrue(all(sf.info(BytesIO(stem)).frames == 30 * SAMPLE_RATE for stem in stems.values()))

            load_or_separate(upload, cache, lambda model: separator, data_root=data_root)
            self.assertEqual(separator.calls, 1)

    def test_incomplete_disk_cache_runs_model(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            write_stems(data_root / "stems" / "uploads" / stem_cache_key(upload), count=5)
            separator = FakeSeparator()
            _stems, origin = load_or_separate(upload, {}, lambda model: separator, data_root=data_root)
            self.assertEqual(origin, "Demucs")
            self.assertEqual(separator.calls, 1)

    def test_four_stem_model_has_separate_cache_and_skips_six_stem_dataset(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            source = data_root / "audio" / "previews" / upload.name
            source.parent.mkdir(parents=True)
            source.write_bytes(upload.source_bytes)
            write_stems(data_root / "stems" / "previews" / "song")
            cache = {}
            six, origin = load_or_separate(upload, cache, lambda model: self.fail("model loaded"),
                                           data_root=data_root)
            self.assertEqual(origin, "disk")
            self.assertEqual(set(six), set(STEM_NAMES))

            model_name = "htdemucs"
            self.assertNotEqual(stem_cache_key(upload), stem_cache_key(upload, model_name))
            separator = FakeSeparator(model_name)
            four, origin = load_or_separate(upload, cache, lambda model: separator,
                                            data_root=data_root, model_name=model_name)
            self.assertEqual(origin, "Demucs")
            self.assertEqual(set(four), set(stem_names(model_name)))
            self.assertEqual(separator.calls, 1)
            again, origin = load_or_separate(upload, cache, lambda model: self.fail("model loaded"),
                                             data_root=data_root, model_name=model_name)
            self.assertEqual(origin, "memory")
            self.assertIs(again, four)

    def test_six_only_runs_one_model_then_later_four_overrides_shared_stems(self) -> None:
        upload = make_upload()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def separator(model_name):
                calls.append(model_name)
                return FakeSeparator(model_name, 0.1 if model_name == "htdemucs" else 0.5)

            cache = {}
            six, _ = load_or_separate(upload, cache, separator, data_root=root)
            self.assertEqual(calls, ["htdemucs_6s"])
            self.assertEqual(sf.read(BytesIO(six["vocals"]))[0][0],
                             sf.read(BytesIO(six["guitar"]))[0][0])

            four, _ = load_or_separate(upload, cache, separator, data_root=root,
                                       model_name="htdemucs")
            self.assertEqual(calls, ["htdemucs_6s", "htdemucs"])
            for name in stem_names("htdemucs"):
                self.assertEqual(six[name], four[name])
            self.assertNotEqual(six["guitar"], four["vocals"])

            # A new session reuses the four-stem disk cache without rerunning it.
            new_cache = {}
            six_again, _ = load_or_separate(upload, new_cache, separator,
                                            data_root=root)
            self.assertEqual(calls, ["htdemucs_6s", "htdemucs", "htdemucs_6s"])
            self.assertEqual(set(six_again), set(STEM_NAMES))
            for name in stem_names("htdemucs"):
                self.assertEqual(sf.read(BytesIO(six_again[name]))[0].shape,
                                 sf.read(BytesIO(four[name]))[0].shape)
                self.assertTrue(np.allclose(sf.read(BytesIO(six_again[name]))[0],
                                            sf.read(BytesIO(four[name]))[0], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
