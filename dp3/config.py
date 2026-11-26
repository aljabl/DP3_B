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
SEPARATION_MODEL = "htdemucs_6s"    # default for offline builds and uploads
SEPARATION_MODELS = {
    "htdemucs_6s": ("vocals", "drums", "bass", "guitar", "piano", "other"),
    "htdemucs": ("vocals", "drums", "bass", "other"),
}

# Features
ANALYSIS_SR = 22050                 # stems are stored at 44.1 kHz, analysed at this rate
HOP = 512                           # shared by beat tracking, chroma and MFCC
N_MFCC = 20                         # coefficient 0 (mostly loudness) is dropped
SILENCE_DB = -40.0                  # stems quieter than this (relative to the mix) are skipped
MIN_BEATS = 8                       # fewer detected beats -> the track fails
TIMBRE_ACTIVE_RANGE_DB = 40.0       # frames quieter than (stem peak - this) are ignored when pooling

# Pooled vector sizes: mean + std over beats or analysis frames
CHROMA_DIM = 2 * 12
MFCC_DIM = 2 * (N_MFCC - 1)
TIMBRE_DIM = 2 * (2 * (N_MFCC - 1) + 7 + 4)

# Rhythm: pooled onset-pattern coefficients, beat-phase groove, and mix context.
RHYTHM_BANDS = 12
RHYTHM_PERIOD_BINS = 25
RHYTHM_DCT_BANDS = 3
RHYTHM_DCT_PERIODS = 17
RHYTHM_SUBDIVISIONS = 4
RHYTHM_PERIODICITY_DIM = 2 * RHYTHM_DCT_BANDS * RHYTHM_DCT_PERIODS
RHYTHM_GROOVE_DIM = 2 * 3 * RHYTHM_SUBDIVISIONS
RHYTHM_CONTEXT_DIM = 10
RHYTHM_DIM = RHYTHM_PERIODICITY_DIM + RHYTHM_GROOVE_DIM + RHYTHM_CONTEXT_DIM

# Facet names
TIMBRE = "timbre"
RHYTHM = "rhythm"
MELODY = "melody"

# Facets that define similarity for each stem type (drums are unpitched, so no melody).
# To support a new stem such as piano, add it here.
FACETS_BY_STEM = {
    "drums": (RHYTHM, TIMBRE),
    "vocals": (RHYTHM, TIMBRE, MELODY),
    "bass": (RHYTHM, TIMBRE, MELODY),
    "other": (RHYTHM, TIMBRE, MELODY),
    "guitar": (RHYTHM, TIMBRE, MELODY),
    "piano": (RHYTHM, TIMBRE, MELODY),
}

# Facets that have an extractor. Add MELODY when its extractor is ready.
IMPLEMENTED_FACETS = (TIMBRE, RHYTHM)
