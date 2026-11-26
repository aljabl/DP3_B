"""Saving user uploads keeps the selected 30-second clip and chosen title."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import soundfile as sf

from dp3 import config
from dp3.save_audio import save_and_index_upload, save_uploaded_audio
from dp3.search import SimilarityIndex
from dp3.stem_service import STEM_NAMES, load_or_separate
from dp3.upload import AudioUploadError, crop_audio, inspect_audio, whole_clip_audio


SAMPLE_RATE = 8000


def make_upload(seconds: float):
    output = BytesIO()
    samples = np.arange(round(seconds * SAMPLE_RATE), dtype="float32") / (seconds * SAMPLE_RATE)
    sf.write(output, samples, SAMPLE_RATE, format="WAV")
    return inspect_audio("original.wav", output.getvalue())


class SaveAudioTests(unittest.TestCase):
    def test_saved_crop_has_six_stems_and_can_be_a_search_result(self) -> None:
        upload = make_upload(31)
        upload.clip_bytes = crop_audio(upload, 1.0)
        upload.clip_start_seconds = 1.0

        class Separator:
            sample_rate = SAMPLE_RATE

            def separate_array(self, audio, sample_rate):
                self.calls += 1
                return {name: np.full((1, audio.shape[1]), 0.1, np.float32)
                        for name in STEM_NAMES}, 0.0

            calls = 0

        class Extractor:
            def analyze(self, mix, stems):
                return {"stems": {
                    name: {"status": "ok", "rms_db": -2.0,
                           "timbre": np.full(config.TIMBRE_DIM, 2, np.float32),
                           "rhythm": np.full(config.RHYTHM_DIM, 2, np.float32)}
                    for name in stems
                }, "meter_beats": 4, "meter_confidence": 0.8}

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_dir = root / "index"
            base_dir.mkdir()
            pd.DataFrame([{
                "dataset": "previews", "track_id": "Existing", "stem": "drums",
                "status": "ok", "row": 0,
            }]).to_csv(base_dir / "embeddings_index.csv", index=False)
            np.savez_compressed(base_dir / "embeddings.npz",
                                timbre=np.ones((1, config.TIMBRE_DIM), np.float32),
                                rhythm=np.ones((1, config.RHYTHM_DIM), np.float32))
            separator = Separator()
            cache = {}
            saved = save_and_index_upload(
                upload, "My Saved Crop", cache, lambda: separator,
                data_root=root, extractor_factory=Extractor,
            )
            self.assertEqual(separator.calls, 1)
            self.assertEqual(saved.read_bytes(), upload.clip_bytes)
            index = SimilarityIndex.load(root)
            user_rows = index.rows[index.rows["dataset"] == "user_added"]
            self.assertEqual(len(user_rows), 6)
            track_id = user_rows.iloc[0]["track_id"]
            for stem in STEM_NAMES:
                path = root / "stems" / "user_added" / track_id / f"{stem}.flac"
                with sf.SoundFile(path) as audio:
                    self.assertEqual(audio.frames, 30 * SAMPLE_RATE)
            query = {"timbre": np.full(config.TIMBRE_DIM, 2, np.float32),
                     "rhythm": np.full(config.RHYTHM_DIM, 2, np.float32)}
            results = index.search("drums", query)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "My Saved Crop")
            self.assertEqual(results[0].mix_path, saved)
            self.assertEqual(results[0].stem_path.name, "drums.flac")
            self.assertEqual(index.search("drums", query, upload=upload), [])
            reupload = inspect_audio(saved.name, saved.read_bytes())
            reupload.clip_bytes = whole_clip_audio(reupload)
            reupload.clip_start_seconds = 0.0
            disk_stems, origin = load_or_separate(
                reupload, {}, Mock(side_effect=AssertionError("Demucs was called")),
                data_root=root,
            )
            self.assertEqual(origin, "disk")
            self.assertEqual(set(disk_stems), set(STEM_NAMES))
            with self.assertRaises(FileExistsError):
                save_and_index_upload(upload, "My Saved Crop", cache, Mock(), data_root=root)

    def test_exact_upload_uses_edited_title_without_overwriting(self) -> None:
        upload = make_upload(30)
        upload.clip_bytes = whole_clip_audio(upload)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "user_added"
            path = save_uploaded_audio(upload, "My New Title", directory=directory)
            self.assertEqual(path.name, "My New Title.wav")
            self.assertEqual(path.read_bytes(), upload.source_bytes)
            with self.assertRaises(FileExistsError):
                save_uploaded_audio(upload, "My New Title", directory=directory)

    def test_long_upload_writes_only_saved_crop(self) -> None:
        upload = make_upload(31)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "user_added"
            with self.assertRaises(AudioUploadError):
                save_uploaded_audio(upload, "Cropped Song", directory=directory)
            self.assertFalse(directory.exists())

            upload.clip_bytes = crop_audio(upload, 1.0)
            upload.clip_start_seconds = 1.0
            path = save_uploaded_audio(upload, "Cropped Song", directory=directory)
            self.assertEqual(path.read_bytes(), upload.clip_bytes)
            with sf.SoundFile(path) as audio:
                self.assertEqual(audio.frames, 30 * SAMPLE_RATE)

    def test_audio_slightly_over_30_seconds_is_cropped(self) -> None:
        upload = make_upload(30.05)
        upload.clip_bytes = whole_clip_audio(upload)
        upload.clip_start_seconds = 0.0
        with TemporaryDirectory() as temporary:
            path = save_uploaded_audio(upload, "Encoder Padded", directory=Path(temporary))
            with sf.SoundFile(path) as audio:
                self.assertEqual(audio.frames, 30 * SAMPLE_RATE)

    def test_title_cannot_escape_save_directory(self) -> None:
        upload = make_upload(30)
        upload.clip_bytes = whole_clip_audio(upload)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "user_added"
            for title in ("../elsewhere", "", "bad\\name"):
                with self.subTest(title=title), self.assertRaises(AudioUploadError):
                    save_uploaded_audio(upload, title, directory=directory)
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
