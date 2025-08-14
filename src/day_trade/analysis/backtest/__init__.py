"""
高度バックテストシステム - Issue #753

包括的なバックテスト機能強化
- 高度リスク指標・リターン分析
- マルチタイムフレーム対応
- 機械学習アンサンブル統合
- 詳細レポート・可視化
- Issue #487の93%精度システム活用
"""

# 後方互換性のため、元のAPIを維持
from .core import BacktestEngine
from .strategies import bollinger_band_strategy, rsi_strategy, simple_sma_strategy
from .types import (
    BacktestConfig,
    BacktestMode,
    BacktestResult,
    MonteCarloResult,
    OptimizationObjective,
    OptimizationResult,
    Position,
    Trade,
    WalkForwardResult,
)

# Issue #753 新機能: 高度バックテストシステム
from .enhanced_backtest_engine import (
    EnhancedBacktestEngine,
    EnhancedBacktestConfig,
    EnhancedBacktestResult,
    create_quick_backtest_config
)

from .advanced_metrics import (
    AdvancedRiskMetrics,
    AdvancedReturnMetrics,
    MarketRegimeMetrics,
    AdvancedBacktestAnalyzer,
    MultiTimeframeAnalyzer
)

from .ml_integration import (
    MLBacktestConfig,
    MLPredictionResult,
    MLBacktestResult,
    MLEnsembleBacktester
)

from .reporting import BacktestReportGenerator

# バージョン情報
__version__ = "1.0.0"
__issue__ = "#753"

# 主要クラスのエクスポート
__all__ = [
    # 基本バックテスト（後方互換性）
    "BacktestEngine",
    "BacktestConfig",
    "BacktestMode",
    "BacktestResult",
    "MonteCarloResult",
    "OptimizationObjective",
    "OptimizationResult",
    "Position",
    "Trade",
    "WalkForwardResult",

    # Issue #753 高度バックテストシステム
    "EnhancedBacktestEngine",
    "EnhancedBacktestConfig",
    "EnhancedBacktestResult",

    # 高度分析
    "AdvancedRiskMetrics",
    "AdvancedReturnMetrics",
    "MarketRegimeMetrics",
    "AdvancedBacktestAnalyzer",
    "MultiTimeframeAnalyzer",

    # ML統合
    "MLBacktestConfig",
    "MLPredictionResult",
    "MLBacktestResult",
    "MLEnsembleBacktester",

    # レポート
    "BacktestReportGenerator",

    # 戦略関数
    "simple_sma_strategy",
    "rsi_strategy",
    "bollinger_band_strategy",

    # ヘルパー
    "create_quick_backtest_config"
]


def get_version_info():
    """Issue #753 バージョン情報取得"""
    return {
        "version": __version__,
        "issue": __issue__,
        "description": "高度バックテスト機能強化システム",
        "features": [
            "高度リスク指標分析",
            "詳細リターン分析",
            "市場レジーム分析",
            "マルチタイムフレーム対応",
            "機械学習アンサンブル統合",
            "包括的レポート生成",
            "インタラクティブ可視化"
        ]
    }
