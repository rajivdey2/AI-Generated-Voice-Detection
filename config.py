import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio processing settings
SAMPLE_RATE = 16000  # Standard sample rate for voice
DURATION = 5  # Maximum duration in seconds
N_MFCC = 40  # Number of MFCC coefficients

# Model settings
MODEL_PATH = MODEL_DIR / "voice_detector.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Supported languages
LANGUAGES = ["tamil", "english", "hindi", "malayalam", "telugu"]

# Feature extraction settings
FEATURE_CONFIG = {
    "n_mfcc": N_MFCC,
    "n_fft": 2048,
    "hop_length": 512,
    "sample_rate": SAMPLE_RATE
}