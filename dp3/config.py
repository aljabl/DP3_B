"""Constants shared by the offline index build and the app.

Anything that must be identical between building the index and processing an
uploaded song lives here (a different hop length, for instance, would silently
break retrieval).
"""

from pathlib import Path

# Paths
DATA = Path("data")
INDEX_DIR = DATA / "index"


def audio_dir(dataset: str) -> Path:
    return DATA / "audio" / dataset


def stems_dir(dataset: str) -> Path:
    return DATA / "stems" / dataset


def features_dir(dataset: str) -> Path:
    return DATA / "features" / dataset


# Separation
SEPARATION_MODEL = "htdemucs_6s"    # vocals, drums, bass, guitar, piano, other

# Features
ANALYSIS_SR = 22050                 # stems are stored at 44.1 kHz, analysed at this rate
HOP = 512                           # shared by beat tracking, chroma and MFCC
N_MFCC = 20                         # coefficient 0 (mostly loudness) is dropped
SILENCE_DB = -40.0                  # stems quieter than this (relative to the mix) are skipped
MIN_BEATS = 8                       # fewer detected beats -> the track fails

# Pooled vector sizes: mean + std over beats
CHROMA_DIM = 2 * 12
MFCC_DIM = 2 * (N_MFCC - 1)
