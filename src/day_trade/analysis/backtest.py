"""
バックテスト機能 (リファクタリング後)

巨大だったbacktest.pyを機能別パッケージに分割し、後方互換性を維持しながら
保守性と可読性を向上させたバックテストモジュール

⚠️ 重要: このファイルは新しいパッケージ構造への移行ブリッジです
実際の実装は backtest/ パッケージ内にあります
"""

# 新しいパッケージ構造からクラスをインポート
from .backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    BacktestResult,
    MonteCarloResult,
    OptimizationObjective,
    OptimizationResult,
    Position,
    Trade,
    WalkForwardResult,
)

# 後方互換性のため、すべてのクラスを再エクスポート
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
]
