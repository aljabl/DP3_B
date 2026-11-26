"""Checks for in-memory audio uploads and the crop flow."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

from dp3.save_audio import save_uploaded_audio
from dp3.crop_window import waveform_peaks
from dp3.upload import AudioUploadError, crop_audio, inspect_audio, source_audio_for_playback


APP_PATH = Path(__file__).resolve().parents[1] / "dp3" / "app.py"
SAMPLE_RATE = 8000


def audio_bytes(seconds: int, *, format: str = "WAV") -> bytes:
    samples = np.repeat(np.arange(seconds, dtype="float32") / 32, SAMPLE_RATE)
    output = BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format=format)
    return output.getvalue()


def alac_bytes(seconds: int, *, sample_rate: int = SAMPLE_RATE,
               container_format: str = "ipod") -> bytes:
    """Encode a real, seekable ALAC/M4A fixture entirely in memory."""
    output = BytesIO()
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)
    with av.open(output, mode="w", format=container_format) as container:
        stream = container.add_stream("alac", rate=sample_rate)
        stream.layout = "mono"
        for second in range(seconds):
            samples = (tone * (4000 + second * 100)).astype(np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format="s16p", layout="mono")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


class UploadTests(unittest.TestCase):
    def test_alac_upload_crops_and_saves_as_playable_wav(self) -> None:
        source = alac_bytes(31)
        upload = inspect_audio("lossless.m4a", source)
        self.assertEqual(upload.duration_seconds, 31)
        clip = crop_audio(upload, 1.0)
        first_second = crop_audio(upload, 0.0)
        with sf.SoundFile(BytesIO(clip)) as audio:
            self.assertEqual(audio.frames, 30 * SAMPLE_RATE)
            cropped_peak = max(abs(audio.read(SAMPLE_RATE).ravel()))
        with sf.SoundFile(BytesIO(first_second)) as audio:
            original_peak = max(abs(audio.read(SAMPLE_RATE).ravel()))
        self.assertGreater(cropped_peak, original_peak)
        upload.clip_bytes = clip
        upload.clip_start_seconds = 1.0
        with TemporaryDirectory() as temporary:
            path = save_uploaded_audio(upload, "Lossless", directory=Path(temporary))
            self.assertEqual(path.name, "Lossless.wav")
            self.assertEqual(path.read_bytes(), clip)

        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].upload("lossless.m4a", source, "audio/mp4").run()
        self.assertFalse(app.exception)
        session_upload = next(iter(app.session_state["audio_uploads"].values()))
        self.assertIsNone(session_upload.clip_bytes)
        self.assertTrue(any(button.label == "Save this 30-second crop" for button in app.button))

    def test_alac_exact_and_short_playback(self) -> None:
        exact = inspect_audio("exact.m4a", alac_bytes(30))
        from dp3.upload import whole_clip_audio
        exact.clip_bytes = whole_clip_audio(exact)
        with sf.SoundFile(BytesIO(exact.clip_bytes)) as audio:
            self.assertEqual(audio.frames, 30 * SAMPLE_RATE)
        with TemporaryDirectory() as temporary:
            path = save_uploaded_audio(exact, "Exact ALAC", directory=Path(temporary))
            self.assertEqual(path.name, "Exact ALAC.wav")
            self.assertEqual(path.read_bytes(), exact.clip_bytes)
        short = inspect_audio("short.m4a", alac_bytes(29))
        playable, mime_type = source_audio_for_playback(short)
        self.assertEqual(mime_type, "audio/wav")
        self.assertIs(playable, source_audio_for_playback(short)[0])
        with sf.SoundFile(BytesIO(playable)) as audio:
            self.assertEqual(audio.frames, 29 * SAMPLE_RATE)
        with self.assertRaisesRegex(AudioUploadError, "does not contain ALAC"):
            inspect_audio("aac.m4a", audio_bytes(30))

        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].set_value([
            ("exact.m4a", exact.source_bytes, "audio/mp4"),
            ("short.m4a", short.source_bytes, "audio/mp4"),
        ]).run()
        self.assertFalse(app.exception)
        uploads = {item.name: item for item in app.session_state["audio_uploads"].values()}
        self.assertIsNotNone(uploads["exact.m4a"].clip_bytes)
        self.assertIsNone(uploads["short.m4a"].clip_bytes)

    def test_alac_caf_container_is_accepted(self) -> None:
        upload = inspect_audio(
            "lossless.caf", alac_bytes(30, sample_rate=44100, container_format="caf")
        )
        from dp3.upload import whole_clip_audio
        clip = whole_clip_audio(upload)
        with sf.SoundFile(BytesIO(clip)) as audio:
            self.assertLess(abs(audio.frames - upload.frames), 0.1 * upload.sample_rate)

    def test_long_upload_requires_saved_crop_and_remains_in_session(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].upload("song.wav", audio_bytes(31), "audio/wav").run()

        self.assertFalse(app.exception)
        upload = next(iter(app.session_state["audio_uploads"].values()))
        self.assertIsNone(upload.clip_bytes)
        self.assertTrue(any(button.label == "Save this 30-second crop" for button in app.button))
        self.assertTrue(next(button for button in app.button if button.label == "Save song to disk").disabled)

        crop_key = next(key for key in app.session_state if key.startswith("crop_window_"))
        peaks = waveform_peaks(upload)
        self.assertEqual(len(peaks), 96)
        self.assertLess(peaks[0], peaks[-1])
        app.session_state[crop_key] = {"start": 1.0}
        app.run()
        next(button for button in app.button if button.label == "Save this 30-second crop").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(upload.clip_start_seconds, 1.0)
        self.assertFalse(next(button for button in app.button if button.label == "Save song to disk").disabled)
        with sf.SoundFile(BytesIO(upload.clip_bytes)) as clip:
            self.assertEqual(clip.frames, 30 * SAMPLE_RATE)
            self.assertAlmostEqual(clip.read(1)[0], 1 / 32, places=3)

        app.session_state[crop_key] = {"start": 0.0}
        app.run()
        self.assertTrue(next(button for button in app.button if button.label == "Save song to disk").disabled)
        self.assertEqual(upload.clip_start_seconds, 1.0)

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

    def test_edited_save_title_updates_library_label(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run()
        app.get("file_uploader")[0].upload("original.wav", audio_bytes(30), "audio/wav").run()

        title = next(field for field in app.text_input if field.label == "Title for saved audio")
        title.set_value("My Library Title").run()

        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "My Library Title" for button in app.button))
        upload = next(iter(app.session_state["audio_uploads"].values()))
        self.assertEqual(upload.name, "original.wav")


if __name__ == "__main__":
    unittest.main()
