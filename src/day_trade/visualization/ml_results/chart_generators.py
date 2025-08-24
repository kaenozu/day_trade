#!/usr/bin/env python3
"""
機械学習結果可視化システム - チャート生成統合モジュール

Issue #315: 高度テクニカル指標・ML機能拡張
分割されたチャート生成モジュールの統合インターフェース
"""

import warnings
from typing import Dict, Optional

from .lstm_charts import BaseChartGenerator, LSTMChartGenerator
from .volatility_charts import VolatilityChartGenerator

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# バックワード互換性のため、全クラスをエクスポート
__all__ = [
    "BaseChartGenerator",
    "LSTMChartGenerator", 
    "VolatilityChartGenerator"
]

logger.info("チャート生成統合モジュール初期化完了")