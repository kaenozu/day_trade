"""
統合設定管理パッケージ

すべての設定管理を統合し、一貫性のある設定APIを提供します。
Phase 3: 設定管理統合とユーティリティ抽出の一環として、
分散していた設定クラスを統一し、保守性を向上させます。
"""

# 新しい統合設定（推奨）
# 従来の設定（後方互換性のため）
from .config_manager import (
    AlertSettings,
    BacktestSettings,
    ConfigManager,
    DatabaseSettings,
    ExecutionSettings,
    MarketHours,
    PatternRecognitionSettings,
    ReportSettings,
    SignalGenerationSettings,
    TechnicalIndicatorSettings,
    WatchlistSymbol,
)

# 後方互換性のためのレガシー設定
from .legacy_config import (
    ScreeningConfig as LegacyScreeningConfig,
)
from .legacy_config import (
    get_screening_config as get_legacy_screening_config,
)
from .legacy_config import (
    set_screening_config as set_legacy_screening_config,
)
from .unified_config import (
    APIConfig,
    DisplayConfig,
    LoggingConfig,
    PerformanceConfig,
    ScreeningConfig,
    SecurityConfig,
    # 各セクション設定
    TradingConfig,
    UnifiedAppConfig,
    UnifiedConfigManager,
    get_logging_config,
    get_performance_config,
    # 便利関数
    get_screening_config,
    get_unified_config_manager,
    set_unified_config_manager,
)

__all__ = [
    # 新しい統合設定（推奨）
    "UnifiedAppConfig",
    "UnifiedConfigManager",
    "get_unified_config_manager",
    "set_unified_config_manager",
    # 設定セクション
    "TradingConfig",
    "DisplayConfig",
    "APIConfig",
    "ScreeningConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "SecurityConfig",
    # 便利関数
    "get_screening_config",
    "get_performance_config",
    "get_logging_config",
    # 従来の設定（後方互換性）
    "ConfigManager",
    "WatchlistSymbol",
    "MarketHours",
    "TechnicalIndicatorSettings",
    "PatternRecognitionSettings",
    "SignalGenerationSettings",
    "AlertSettings",
    "BacktestSettings",
    "ReportSettings",
    "ExecutionSettings",
    "DatabaseSettings",
    # レガシー設定
    "LegacyScreeningConfig",
    "get_legacy_screening_config",
    "set_legacy_screening_config",
]
