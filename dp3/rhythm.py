"""Beat-aligned, fixed-size rhythm features for stem similarity search.

The periodicity block is inspired by onset patterns: positive spectral changes
in frequency bands are measured on a logarithmic periodicity scale, then
compressed with a two-dimensional DCT. Groove and meter are separate blocks.
"""

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.fft import dctn

from . import config


METER_CANDIDATES = (2, 3, 4, 6)
PERIODICITY_END = config.RHYTHM_PERIODICITY_DIM
GROOVE_END = PERIODICITY_END + config.RHYTHM_GROOVE_DIM
RHYTHM_BLOCKS = (
    (slice(0, PERIODICITY_END), 0.45),
    (slice(PERIODICITY_END, GROOVE_END), 0.40),
    (slice(GROOVE_END, config.RHYTHM_DIM), 0.15),
)


@dataclass(frozen=True)
class RhythmContext:
    beat_times: np.ndarray
    tempo: float
    meter_scores: np.ndarray
    meter_confidence: float
    meter_beats: int | None


class RhythmExtractor:
    """Describe periodicity and beat-phase onset placement of one stem."""

    @staticmethod
    def band_onsets(audio: np.ndarray) -> np.ndarray:
        """Positive changes in 12 log-frequency bands, one value per STFT frame."""
        spectrum = np.abs(librosa.stft(audio, hop_length=config.HOP))
        frequencies = librosa.fft_frequencies(sr=config.ANALYSIS_SR)
        edges = np.geomspace(50, min(8000, config.ANALYSIS_SR / 2),
                             config.RHYTHM_BANDS + 1)
        bands = np.empty((config.RHYTHM_BANDS, spectrum.shape[1]), dtype=np.float32)
        for band in range(config.RHYTHM_BANDS):
            mask = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
            bands[band] = np.log1p(spectrum[mask].sum(axis=0))

        # Subtract the trailing 250 ms mean, as in the onset-pattern paper.
        width = max(1, round(0.25 * config.ANALYSIS_SR / config.HOP))
        cumulative = np.pad(np.cumsum(bands, axis=1), ((0, 0), (1, 0)))
        end = np.arange(1, bands.shape[1] + 1)
        start = np.maximum(0, end - width)
        baseline = (cumulative[:, end] - cumulative[:, start]) / (end - start)
        onsets = np.maximum(bands - baseline, 0)
        if float(np.max(onsets)) < 1e-5:
            return np.zeros_like(onsets)
        scale = np.percentile(onsets, 95, axis=1, keepdims=True)
        return np.clip(onsets / np.maximum(scale, 1e-5), 0, 4).astype(np.float32)

    @classmethod
    def context(cls, mix: np.ndarray, beat_times: np.ndarray, tempo: float) -> RhythmContext:
        """Estimate beat grouping from the full mix, shared by all stems."""
        times = np.asarray(beat_times, dtype=np.float64)
        onset = cls.band_onsets(mix).sum(axis=0)
        frame_times = np.arange(len(onset)) * config.HOP / config.ANALYSIS_SR
        beat_period = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
        half_window = max(0.04, 0.12 * beat_period)
        accents = np.array([
            onset[np.abs(frame_times - beat) <= half_window].sum()
            for beat in times
        ], dtype=np.float64)
        scores = np.zeros(len(METER_CANDIDATES), dtype=np.float32)
        if len(accents) >= 8 and float(np.std(accents)) > 1e-6:
            accents = (accents - accents.mean()) / (accents.std() + 1e-8)
            for i, meter in enumerate(METER_CANDIDATES):
                phase_means = np.array([accents[phase::meter].mean()
                                        for phase in range(meter)])
                contrast = max(0.0, float(phase_means.max() - phase_means.mean()))
                recurrence = max(0.0, float(np.mean(accents[:-meter] * accents[meter:])))
                scores[i] = contrast * recurrence
        total = float(scores.sum())
        if total > 1e-8:
            scores /= total
        ordered = np.sort(scores)
        confidence = float((ordered[-1] - ordered[-2]) * min(1, len(times) / 24))
        meter_beats = METER_CANDIDATES[int(scores.argmax())] if confidence >= 0.2 else None
        return RhythmContext(times, float(tempo), scores, confidence, meter_beats)

    @staticmethod
    def _periodicity(onsets: np.ndarray) -> np.ndarray:
        seconds_per_frame = config.HOP / config.ANALYSIS_SR
        width = round(2.63 / seconds_per_frame)
        step = max(1, round(0.25 / seconds_per_frame))
        if onsets.shape[1] < width:
            onsets = np.pad(onsets, ((0, 0), (0, width - onsets.shape[1])))
        starts = range(0, onsets.shape[1] - width + 1, step)
        windows = np.stack([onsets[:, start:start + width] for start in starts])
        windows *= np.hanning(width)[None, None, :]
        fft_size = round(6.0 / seconds_per_frame)
        spectrum = np.abs(np.fft.rfft(windows, n=fft_size, axis=2))
        frequencies = np.fft.rfftfreq(fft_size, d=seconds_per_frame)
        target = np.geomspace(40 / 60, 800 / 60, config.RHYTHM_PERIOD_BINS)
        upper = np.searchsorted(frequencies, target).clip(1, len(frequencies) - 1)
        lower = upper - 1
        blend = (target - frequencies[lower]) / (frequencies[upper] - frequencies[lower])
        patterns = spectrum[:, :, lower] * (1 - blend) + spectrum[:, :, upper] * blend
        patterns = np.log1p(patterns)
        coefficients = dctn(patterns, type=2, norm="ortho", axes=(1, 2))
        compact = coefficients[:, :config.RHYTHM_DCT_BANDS, :config.RHYTHM_DCT_PERIODS]
        return np.concatenate((compact.mean(axis=0).ravel(),
                               compact.std(axis=0).ravel())).astype(np.float32)

    @staticmethod
    def _groove(onsets: np.ndarray, beat_times: np.ndarray) -> tuple[np.ndarray, float, float]:
        beat_times = np.asarray(beat_times)
        hist = np.zeros((len(beat_times) - 1, 3, config.RHYTHM_SUBDIVISIONS), np.float32)
        frame_times = np.arange(onsets.shape[1]) * config.HOP / config.ANALYSIS_SR
        beat = np.searchsorted(beat_times, frame_times, side="right") - 1
        valid = (beat >= 0) & (beat < len(hist))
        beat = beat[valid]
        phase = (frame_times[valid] - beat_times[beat]) / (
            beat_times[beat + 1] - beat_times[beat]
        )
        subdivision = np.minimum((phase * config.RHYTHM_SUBDIVISIONS).astype(int),
                                 config.RHYTHM_SUBDIVISIONS - 1)
        for band, values in enumerate(np.array_split(onsets, 3, axis=0)):
            np.add.at(hist, (beat, band, subdivision), values.sum(axis=0)[valid])
        totals = hist.sum(axis=(1, 2), keepdims=True)
        hist = np.divide(hist, totals, out=np.zeros_like(hist), where=totals > 1e-8)
        groove = np.concatenate((hist.mean(axis=0).ravel(), hist.std(axis=0).ravel()))
        density = float(np.mean(onsets.sum(axis=0) > 0.5))
        syncopation = float(hist[:, :, 1::2].sum() / max(hist.sum(), 1e-8))
        return groove.astype(np.float32), density, syncopation

    def extract(self, stem: np.ndarray, context: RhythmContext) -> np.ndarray:
        onsets = self.band_onsets(stem)
        if not np.any(onsets):
            return np.zeros(config.RHYTHM_DIM, np.float32)
        periodicity = self._periodicity(onsets)
        groove, density, syncopation = self._groove(onsets, context.beat_times)
        tempo = float(np.clip(context.tempo, 40, 240))
        octave_phase = 2 * np.pi * np.log2(tempo / 60)
        context_values = np.array([
            np.sin(octave_phase), np.cos(octave_phase),
            np.log2(tempo / 40) / np.log2(240 / 40),
            *(context.meter_scores * context.meter_confidence),
            context.meter_confidence, density, syncopation,
        ], dtype=np.float32)
        vector = np.concatenate((periodicity, groove, context_values))
        if vector.shape != (config.RHYTHM_DIM,) or not np.isfinite(vector).all():
            raise ValueError("rhythm extraction produced an invalid vector")
        return vector.astype(np.float32)
