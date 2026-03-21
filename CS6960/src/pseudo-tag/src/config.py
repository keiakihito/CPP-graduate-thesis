"""Project paths and small global constants."""

from pathlib import Path


# Assumption: this file lives in <project_root>/src/config.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
WAV_DIR = DATA_DIR / "wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
RETRIEVAL_DIR = OUTPUT_DIR / "retrieval"
METRICS_DIR = OUTPUT_DIR / "metrics"

SAMPLE_RATE = 16_000
TOP_K = 10
