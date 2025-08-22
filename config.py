from pathlib import Path
import logging.config

# Project Root
PROJECT_ROOT = Path(__file__).parent

# Caching
ANALYSIS_CACHE_MAX_SIZE = 50



# ML Prediction
ML_RANDOM_FALLBACK_ENABLED = True # Set to False for production to force explicit ML prediction or error
