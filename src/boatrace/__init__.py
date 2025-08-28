"""
Boatrace Open API System

競艇予想・舟券購入支援システム
BoatraceOpenAPIを利用した包括的な競艇分析・投票システム
"""

__version__ = "1.0.0"
__author__ = "Boatrace System Team"

from .core import api_client, data_models
from .prediction import prediction_engine
from .betting import ticket_manager
from .data import data_collector

__all__ = [
    "api_client",
    "data_models", 
    "prediction_engine",
    "ticket_manager",
    "data_collector"
]