"""
分析専用ダッシュボードモジュール

【重要】自動取引機能は完全に無効化されています
分析・教育・研究専用のWebダッシュボードを提供します
"""

from .analysis_dashboard_server import app

__all__ = ["app"]
