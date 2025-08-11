"""
Risk Management Interfaces
リスク管理インターフェース

依存関係逆転の原則(DIP)を適用したインターフェース定義
"""

from .alert_interfaces import AlertSeverity, IAlertRule, INotificationChannel
from .cache_interfaces import CacheStrategy, ICacheProvider, ICacheSerializer
from .risk_interfaces import (
    IAlertManager,
    ICacheManager,
    IConfigManager,
    IMetricsCollector,
    IRiskAnalyzer,
)

__all__ = [
    # メインインターフェース
    "IRiskAnalyzer",
    "IAlertManager",
    "ICacheManager",
    "IMetricsCollector",
    "IConfigManager",
    # キャッシュインターフェース
    "ICacheProvider",
    "ICacheSerializer",
    "CacheStrategy",
    # アラートインターフェース
    "INotificationChannel",
    "IAlertRule",
    "AlertSeverity",
]
