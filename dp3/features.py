"""
Extracts Beat-synchronous chroma and MFCC features per stem, and pools them
into one embedding per stem. The emdedding is stored in FeatureCache.

Chroma: shows which pitch classes are present (the notes and harmony)
MFCCs: describe timbre
"""

import time
import librosa
import numpy as np

from . import config


def load_mono(path) -> np.ndarray:
    """
    Load any audio file as mono at the analysis sample rate.
    Feature extraction doesn't need the same specs as the Demucs model;
    a smaller sample rate and mono audio makes the process faster.
    """

    y, _ = librosa.load(str(path), sr=config.ANALYSIS_SR, mono=True)
    return y


def to_analysis(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convert in-memory audio shaped (channels, samples) at sr to mono at the analysis rate."""

    return librosa.resample(librosa.to_mono(audio), orig_sr=sr, target_sr=config.ANALYSIS_SR)

def pool(beat_matrix: np.ndarray) -> np.ndarray:
    """
    Collapse a (n_features, n_beats) matrix into a single vector: the mean and
    std of each feature across all beats.

    Each stem becomes a fixed size embedding: 24 numbers for chroma, 38 numbers
    for MFCC), regardless of the stem's original length.
    """

    return np.concatenate([beat_matrix.mean(axis=1), beat_matrix.std(axis=1)]).astype(np.float32)

class FeatureExtractor:
    """
    Beat tracking on the full audio mix, plus beat-synchronous features per stem.
    """

    def __init__(self, silence_db:float = config.SILENCE_DB, min_beats:int = config.MIN_BEATS):
        self.silence_db = silence_db
        self.min_beats = min_beats

    def track_beats(self, mix:np.ndarray) -> tuple:
        """
        Beat tracking on the full mix. Returns (tempo, beat_frames, beat_times)

        Finds the tempo and position of every beat in the full mix.
        """

        tempo, frames = librosa.beat.beat_track(y=mix, sr=config.ANALYSIS_SR, hop_length=config.HOP)
        frames = np.asarray(frames)

        if len(frames) < self.min_beats:
            raise ValueError(f"only {len(frames)} beats found (min {self.min_beats})")

        times = librosa.frames_to_time(frames, sr=config.ANALYSIS_SR, hop_length=config.HOP)

        return float(np.atleast_1d(tempo)[0]), frames, times

    def is_silent(self, level: float) -> bool:
        """level: stem loudness relative to the mix, in dB."""

        return level < self.silence_db

    def analyze(self, mix: np.ndarray, stems: dict) -> dict:
        """
        Main entry point. Beat-track the mix, then compute features for every
        stem that is not silent.

        mix, stems: mono float arrays at config.ANALYSIS_SR (stems maps name -> array).
        Raises ValueError if no stems are given or too few beats are found.

        Returns a dict with tempo, beat_times, n_beats, beat_time_s, stems_time_s and
        stems: name -> {"status": "ok" | "silent", "rms_db", and for "ok" stems
        chroma_beat, mfcc_beat (beat-level) and chroma, mfcc (pooled)}.
        """
        if not stems:
            raise ValueError("no stems to analyze")

        t0 = time.perf_counter()
        # Track beats for entire mix
        tempo, frames, times = self.track_beats(mix)
        beat_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        chroma_time = 0.0
        mfcc_time = 0.0

        out = {}

        for name, y in stems.items():
            db = self.level_db(y, mix)

            if self.is_silent(db):
                # Skip silent stems, else they'll create noise in results.
                out[name] = {"status": "silent", "rms_db": db}
                continue

            # Turn stem into beat-level chroma and MFCC values, plus pooled vectors
            chroma_b, chroma_s = self._timed(self.chroma_beats, y, frames)
            mfcc_b, mfcc_s = self._timed(self.mfcc_beats, y, frames)

            chroma_time += chroma_s
            mfcc_time += mfcc_s

            out[name] = {"status": "ok", "rms_db": db,
                         "chroma_beat": chroma_b.astype(np.float32),
                         "mfcc_beat": mfcc_b.astype(np.float32),
                         "chroma": pool(chroma_b), "mfcc": pool(mfcc_b)}

        stems_time = time.perf_counter() - t0

        return {"tempo": tempo, "beat_times": times, "n_beats": int(len(frames)),
                "beat_time_s": beat_time, "stems_time_s": stems_time,
                "chroma_time_s": chroma_time, "mfcc_time_s": mfcc_time, "stems": out}

    @staticmethod
    def _timed(func, *args) -> tuple:
        """Calls func(*args) and returns (result, elapsed seconds)."""

        start = time.perf_counter()
        result = func(*args)
        return result, time.perf_counter() - start

    @staticmethod
    def chroma_beats(stem: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
        """
        Beat-synchronous chroma, shaped (12, n). Uses the median per beat
        (control for outliers).

        Averages the feature between consecutive beats, so we get one
        feature column per beat.

        librosa.util.sync pads the boundaries with the start and end of the
        audio, so n = n_beats + 1 and the material before the first / after the
        last beat is kept.
        """

        chroma = librosa.feature.chroma_cqt(y=stem, sr=config.ANALYSIS_SR, hop_length=config.HOP)
        return librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    @staticmethod
    def mfcc_beats(stem: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
        """
        Beat-synchronous MFCC without the 0th coefficient, shaped (N_MFCC - 1, n).
        Uses the mean per beat.

        Averages the feature between consecutive beats, so we get one
        feature column per beat.

        librosa.util.sync pads the boundaries with the start and end of the
        audio, so n = n_beats + 1 and the material before the first / after the
        last beat is kept.
        """

        mfcc = librosa.feature.mfcc(y=stem, sr=config.ANALYSIS_SR, hop_length=config.HOP,
                                    n_mfcc=config.N_MFCC)[1:]
        return librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)

    @staticmethod
    def _rms(y: np.ndarray) -> float:
        """Measures the average loudness of a signal."""
        return float(np.sqrt(np.mean(y ** 2)))

    def level_db(self, stem: np.ndarray, mix: np.ndarray) -> float:
        """Stem loudness relative to the mix, in dB."""

        return float(20 * np.log10(self._rms(stem) / (self._rms(mix) + 1e-12) + 1e-12))

