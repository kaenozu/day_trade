from pathlib import Path
import logging.config

# Project Root
PROJECT_ROOT = Path(__file__).parent

# Caching
ANALYSIS_CACHE_MAX_SIZE = 50

# Logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'daytrade.log',
            'maxBytes': 10485760, # 10 MB
            'backupCount': 5,
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'daytrade': { # specific logger for daytrade application
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# ML Prediction
ML_RANDOM_FALLBACK_ENABLED = True # Set to False for production to force explicit ML prediction or error
