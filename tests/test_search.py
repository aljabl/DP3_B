"""Checks for same-stem ranking and playable search results."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import soundfile as sf

from dp3.search import SearchError, SimilarityIndex
from dp3.upload import inspect_audio, whole_clip_audio


SAMPLE_RATE = 8000


def wav_bytes(amplitude: float) -> bytes:
    output = BytesIO()
    sf.write(output, np.full(30 * SAMPLE_RATE, amplitude), SAMPLE_RATE, format="WAV")
    return output.getvalue()


class SearchTests(unittest.TestCase):
    def test_returns_three_playable_same_stem_matches(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ("Self", "Good", "Missing", "Medium", "Far", "Last")
            rows = []
            for position, name in enumerate(names):
                rows.append({"dataset": "previews", "track_id": name, "stem": "drums", "status": "ok", "row": position})
                mix = root / "audio" / "previews" / f"{name}.wav"
                mix.parent.mkdir(parents=True, exist_ok=True)
                mix.write_bytes(wav_bytes(position / 20))
                if name != "Missing":
                    stem = root / "stems" / "previews" / name / "drums.flac"
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(stem, np.zeros(30 * SAMPLE_RATE), SAMPLE_RATE)

            rows.append({"dataset": "previews", "track_id": "Self", "stem": "vocals", "status": "ok", "row": 6})
            vectors = np.array(
                [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.0, 0.8], [1.0, 0.0]],
                dtype=np.float32,
            )
            index = SimilarityIndex(pd.DataFrame(rows), {"timbre": vectors}, root)
            source = root / "audio" / "previews" / "Self.wav"
            upload = inspect_audio(source.name, source.read_bytes())
            upload.clip_bytes = whole_clip_audio(upload)
            upload.clip_start_seconds = 0.0

            results = index.search(
                "drums", {"timbre": vectors[0]}, upload=upload, limit=3
            )
            self.assertEqual([result.track_id for result in results], ["Good", "Medium", "Far"])
            self.assertTrue(all(result.stem == "drums" for result in results))
            self.assertTrue(all(result.mix_path.is_file() and result.stem_path.is_file() for result in results))
            self.assertEqual([result.similarity for result in results], sorted(
                (result.similarity for result in results), reverse=True
            ))

    def test_missing_saved_index_has_clear_error(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SearchError, "index is missing"):
                SimilarityIndex.load(Path(tmp))


if __name__ == "__main__":
    unittest.main()
