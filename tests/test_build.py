"""Checks that legacy feature caches become a usable facet index."""

import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import build
from dp3 import config
from dp3.cache import FeatureCache
from dp3.search import SimilarityIndex


class BuildTests(unittest.TestCase):
    def test_table_backfills_timbre_and_keeps_rows_aligned(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "features" / "previews" / "song" / "part"
            folder.mkdir(parents=True)
            (folder / "meta.json").write_text(json.dumps({
                "tempo": 120, "n_beats": 8,
                "stems": {"drums": {"status": "ok", "rms_db": -2},
                          "vocals": {"status": "silent", "rms_db": -50}},
            }))
            chroma = np.ones(config.CHROMA_DIM, dtype=np.float32)
            mfcc = np.ones(config.MFCC_DIM, dtype=np.float32) * 2
            np.savez_compressed(folder / "features.npz", beat_times=np.arange(8),
                                drums__chroma=chroma, drums__mfcc=mfcc,
                                drums__chroma_beat=np.ones((12, 8)),
                                drums__mfcc_beat=np.ones((19, 8)))
            stem = root / "stems" / "previews" / "song" / "part" / "drums.flac"
            stem.parent.mkdir(parents=True)
            stem.touch()

            class StubTimbre:
                def extract(self, _audio):
                    return np.arange(config.TIMBRE_DIM, dtype=np.float32)

            class StubExtractor:
                def __init__(self):
                    self.extractors = {
                        config.TIMBRE: StubTimbre(),
                        config.RHYTHM: SimpleNamespace(
                            context=lambda *_: SimpleNamespace(meter_beats=4, meter_confidence=0.8),
                            extract=lambda *_: np.arange(config.RHYTHM_DIM, dtype=np.float32),
                        ),
                    }

            with patch.object(config, "DATA", root), patch.object(config, "INDEX_DIR", root / "index"), \
                    patch.object(build, "FeatureExtractor", StubExtractor), \
                    patch.object(build, "load_mono", return_value=np.zeros(1)):
                build.stage_table()
                self.assertEqual(FeatureCache(root / "features" / "previews").missing_facets("song/part"), {})
                table = pd.read_csv(root / "index" / "embeddings_index.csv")
                with np.load(root / "index" / "embeddings.npz") as saved:
                    self.assertEqual(saved["timbre"].shape, (1, config.TIMBRE_DIM))
                    self.assertEqual(saved["rhythm"].shape, (1, config.RHYTHM_DIM))
                    np.testing.assert_array_equal(saved["timbre"][0], np.arange(config.TIMBRE_DIM))
                    np.testing.assert_array_equal(saved["rhythm"][0], np.arange(config.RHYTHM_DIM))
                    np.testing.assert_array_equal(saved["chroma"][0], chroma)
                    np.testing.assert_array_equal(saved["mfcc"][0], mfcc)
                self.assertEqual(table["row"].tolist(), [0, -1])
                self.assertEqual(table["track_id"].tolist(), ["song/part", "song/part"])
                self.assertEqual(table["meter_beats"].tolist(), [4, 4])
                self.assertEqual(SimilarityIndex.load(root).vectors["timbre"].shape,
                                 (1, config.TIMBRE_DIM))
                self.assertEqual(FeatureCache(root / "features" / "previews").load("song/part")["meter_beats"], 4)

    def test_empty_table_has_two_dimensional_timbre_array(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "features" / "previews").mkdir(parents=True)
            with patch.object(config, "DATA", root), patch.object(config, "INDEX_DIR", root / "index"):
                build.stage_table()
            with np.load(root / "index" / "embeddings.npz") as saved:
                self.assertEqual(saved["timbre"].shape, (0, config.TIMBRE_DIM))
                self.assertEqual(saved["rhythm"].shape, (0, config.RHYTHM_DIM))


if __name__ == "__main__":
    unittest.main()
