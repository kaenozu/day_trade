#!/usr/bin/env python3
"""
投資機会アラートシステム - パッケージ初期化
後方互換性を保つための再エクスポート
"""

# 列挙型のエクスポート
from .enums import (
    OpportunityType,
    OpportunitySeverity,
    TradingAction,
    TimeHorizon,
)

# データモデルのエクスポート
from .models import (
    OpportunityConfig,
    InvestmentOpportunity,
    MarketCondition,
    AlertConfig,
)

# メインシステムのエクスポート
from .system import InvestmentOpportunityAlertSystem

# ハンドラーのエクスポート
from .handlers import (
    log_opportunity_handler,
    console_opportunity_handler,
    file_opportunity_handler,
)

# ユーティリティ関数のエクスポート
from .utils import setup_investment_opportunity_monitoring

# 検出器と分析器は内部使用のため、エクスポートしない
# from .detectors import OpportunityDetector
# from .analyzers import OpportunityAnalyzer

# 公開API
__all__ = [
    # 列挙型
    "OpportunityType",
    "OpportunitySeverity", 
    "TradingAction",
    "TimeHorizon",
    # データモデル
    "OpportunityConfig",
    "InvestmentOpportunity",
    "MarketCondition",
    "AlertConfig",
    # メインシステム
    "InvestmentOpportunityAlertSystem",
    # ハンドラー
    "log_opportunity_handler",
    "console_opportunity_handler",
    "file_opportunity_handler",
    # ユーティリティ
    "setup_investment_opportunity_monitoring",
]