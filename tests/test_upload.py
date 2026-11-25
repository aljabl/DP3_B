"""Checks for in-memory audio uploads and the crop flow."""

from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

from dp3.upload import AudioUploadError, crop_audio, inspect_audio


APP_PATH = Path(__file__).resolve().parents[1] / "dp3" / "app.py"
SAMPLE_RATE = 8000


def audio_bytes(seconds: int, *, format: str = "WAV") -> bytes:
    samples = np.repeat(np.arange(seconds, dtype="float32") / 32, SAMPLE_RATE)
    output = BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format=format)
    return output.getvalue()


class UploadTests(unittest.TestCase):
    def test_long_upload_requires_saved_crop_and_remains_in_session(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].upload("song.wav", audio_bytes(31), "audio/wav").run()

        self.assertFalse(app.exception)
        upload = next(iter(app.session_state["audio_uploads"].values()))
        self.assertIsNone(upload.clip_bytes)
        self.assertEqual(len(app.button), 1)

        app.slider[0].set_value(1.0).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(upload.clip_start_seconds, 1.0)
        with sf.SoundFile(BytesIO(upload.clip_bytes)) as clip:
            self.assertEqual(clip.frames, 30 * SAMPLE_RATE)
            self.assertAlmostEqual(clip.read(1)[0], 1 / 32, places=3)

        app.get("file_uploader")[0].clear().run()
        self.assertEqual(len(app.session_state["audio_uploads"]), 1)
        self.assertIsNotNone(upload.clip_bytes)

    def test_exact_mp3_is_accepted_and_short_audio_is_not(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].set_value(
            [
                ("ready.mp3", audio_bytes(30, format="MP3"), "audio/mpeg"),
                ("short.wav", audio_bytes(29), "audio/wav"),
            ]
        ).run()

        self.assertFalse(app.exception)
        uploads = {upload.name: upload for upload in app.session_state["audio_uploads"].values()}
        self.assertEqual(len(uploads), 2)
        self.assertIsNotNone(uploads["ready.mp3"].clip_bytes)
        self.assertIsNone(uploads["short.wav"].clip_bytes)

    def test_invalid_and_short_audio_cannot_be_cropped(self) -> None:
        with self.assertRaises(AudioUploadError):
            inspect_audio("broken.mp3", b"not audio")
        short = inspect_audio("short.wav", audio_bytes(29))
        with self.assertRaises(AudioUploadError):
            crop_audio(short, 0)

    def test_preview_just_under_30_seconds_is_accepted(self) -> None:
        output = BytesIO()
        sf.write(output, np.zeros(30 * SAMPLE_RATE - 200), SAMPLE_RATE, format="WAV")
        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].upload("preview.wav", output.getvalue(), "audio/wav").run()

        self.assertFalse(app.exception)
        upload = next(iter(app.session_state["audio_uploads"].values()))
        self.assertIsNotNone(upload.clip_bytes)
        self.assertTrue(any(button.label == "Separate into six stems" for button in app.button))


if __name__ == "__main__":
    unittest.main()
