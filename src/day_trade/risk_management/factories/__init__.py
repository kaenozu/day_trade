"""
Risk Management Factories
ファクトリーパターン実装

動的なコンポーネント生成とDIコンテナー機能を提供
"""

from .alert_factory import (
    AlertChannelFactory,
    NotificationChannelType,
    create_notification_channel,
)
from .cache_factory import (
    CacheProviderFactory,
    CacheProviderType,
    create_cache_provider,
)
from .config_factory import (
    ConfigProviderFactory,
    ConfigProviderType,
    create_config_provider,
)
from .dependency_injection import (
    DIContainer,
    ServiceLifetime,
    inject_dependencies,
    singleton,
    transient,
)
from .risk_analyzer_factory import (
    AnalyzerType,
    RiskAnalyzerFactory,
    create_analyzer,
    register_analyzer_type,
)

__all__ = [
    # アナライザーファクトリー
    "RiskAnalyzerFactory",
    "AnalyzerType",
    "create_analyzer",
    "register_analyzer_type",
    # キャッシュファクトリー
    "CacheProviderFactory",
    "CacheProviderType",
    "create_cache_provider",
    # アラートファクトリー
    "AlertChannelFactory",
    "NotificationChannelType",
    "create_notification_channel",
    # 設定ファクトリー
    "ConfigProviderFactory",
    "ConfigProviderType",
    "create_config_provider",
    # 依存性注入
    "DIContainer",
    "ServiceLifetime",
    "inject_dependencies",
    "singleton",
    "transient",
]
