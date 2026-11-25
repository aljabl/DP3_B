"""Checks meter accents and beat-relative onset placement."""

import unittest

import numpy as np

from dp3 import config
from dp3.rhythm import GROOVE_END, PERIODICITY_END, RhythmExtractor


class RhythmTests(unittest.TestCase):
    @staticmethod
    def pulses(phases: list[float], *, accents: bool = False) -> np.ndarray:
        sample_rate = config.ANALYSIS_SR
        audio = np.zeros(16 * sample_rate, np.float32)
        for beat in range(32):
            for phase in phases:
                position = round((beat + phase) * 0.5 * sample_rate)
                strength = (1.0 if beat % 4 == 0 else 0.2) if accents else 1.0
                audio[position:position + 150] += strength * np.hanning(150)
        return audio

    def test_four_beat_accents_produce_confident_meter(self) -> None:
        extractor = RhythmExtractor()
        beats = np.arange(0, 16, 0.5)
        audio = self.pulses([0], accents=True)
        context = extractor.context(audio, beats, 120)
        vector = extractor.extract(audio, context)
        self.assertEqual(context.meter_beats, 4)
        self.assertGreater(context.meter_confidence, 0.5)
        self.assertEqual(vector.shape, (config.RHYTHM_DIM,))
        self.assertTrue(np.isfinite(vector).all())

    def test_groove_distinguishes_beat_from_offbeat(self) -> None:
        extractor = RhythmExtractor()
        beats = np.arange(0, 16, 0.5)
        on_beat = self.pulses([0])
        offbeat = self.pulses([0.5])
        context = extractor.context(on_beat, beats, 120)
        a = extractor.extract(on_beat, context)
        b = extractor.extract(offbeat, context)
        groove_a = a[PERIODICITY_END:GROOVE_END]
        groove_b = b[PERIODICITY_END:GROOVE_END]
        self.assertGreater(np.linalg.norm(groove_a - groove_b), 0.1)


if __name__ == "__main__":
    unittest.main()
