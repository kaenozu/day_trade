"""
Risk Management Interfaces
リスク管理インターフェース

依存関係逆転の原則(DIP)を適用したインターフェース定義
"""

from .risk_interfaces import (
    IRiskAnalyzer,
    IAlertManager,
    ICacheManager,
    IMetricsCollector,
    IConfigManager
)

from .cache_interfaces import (
    ICacheProvider,
    ICacheSerializer,
    CacheStrategy
)

from .alert_interfaces import (
    INotificationChannel,
    IAlertRule,
    AlertSeverity
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
    "AlertSeverity"
]
