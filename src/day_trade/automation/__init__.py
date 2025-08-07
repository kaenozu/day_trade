"""
自動化モジュール（自動取引機能は無効化済み）

【重要】自動取引機能は完全に無効化されています
分析・情報提供・手動取引支援のみを提供します
"""

from .analysis_only_engine import AnalysisOnlyEngine
from .trading_engine import TradingEngine

__all__ = [
    "TradingEngine",  # 市場分析エンジン（旧：自動取引エンジン）
    "AnalysisOnlyEngine",  # 分析専用エンジン
]
