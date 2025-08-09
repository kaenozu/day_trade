#!/usr/bin/env python3
"""
リアルタイム予測API モジュール
Phase F: 次世代機能拡張フェーズ

深層学習ベースのリアルタイム株価予測API
"""

from .realtime_prediction_api import (
    RealtimePredictionAPI,
    PredictionRequest,
    PredictionResponse
)

__all__ = [
    'RealtimePredictionAPI',
    'PredictionRequest', 
    'PredictionResponse'
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade API Team"
__description__ = "リアルタイム株価予測API - 深層学習統合システム"