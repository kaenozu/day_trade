"""
バックテストパッケージ

巨大だった backtest.py を機能別に分割し、保守性と可読性を向上
- types.py: 型定義とデータクラス
- core.py: バックテストエンジンのコア機能
- strategies.py: 取引戦略実装
- analytics.py: パフォーマンス分析
- optimization.py: パラメータ最適化
- visualization.py: レポート・可視化
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

# 主要クラスのエクスポート
__all__ = [
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
    # 戦略関数
    "simple_sma_strategy",
    "rsi_strategy",
    "bollinger_band_strategy",
]
