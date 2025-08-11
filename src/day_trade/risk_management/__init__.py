"""
Risk Management Module - Refactored Architecture
リスク管理モジュール - リファクタリング済みアーキテクチャ

依存関係逆転の原則(DIP)を適用した新アーキテクチャ
"""

from .interfaces.risk_interfaces import (
    IAlertManager,
    ICacheManager,
    IMetricsCollector,
    IRiskAnalyzer,
)
from .models.unified_models import (
    RiskAnalysisContext,
    UnifiedRiskRequest,
    UnifiedRiskResult,
)

__all__ = [
    # インターフェース
    "IRiskAnalyzer",
    "IAlertManager",
    "ICacheManager",
    "IMetricsCollector",
    # 統一モデル
    "UnifiedRiskRequest",
    "UnifiedRiskResult",
    "RiskAnalysisContext",
]
